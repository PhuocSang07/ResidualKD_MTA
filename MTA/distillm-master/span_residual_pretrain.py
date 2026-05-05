"""Stage 1 — Projector pretraining for SpanResidual KD.

Trains P_T->A (d_T->d_A) and P_A->T (d_A->d_T) by minimising
CE(W_T · P_A->T(P_T->A(h^T)), y) over Qwen-tokenised Dolly.
Teacher model and W_T are fully frozen.

Paper hyperparameters (On et al. ICLR 2026):
  Stage 1: epochs=10, lr=1e-3, weight_decay=1e-4, d_A=64, cosine schedule
"""
import os
import json
import random

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import deepspeed

from transformers import AutoModelForCausalLM, AutoConfig
from peft import PeftModel

from arguments import get_args
from data_utils.lm_datasets import LMTrainDataset
from utils import initialize, print_rank, save_rank, get_tokenizer
from span_residual_utils import ProjectorTA

torch.set_num_threads(1)


def get_teacher(args, device):
    config = AutoConfig.from_pretrained(args.teacher_model_path)
    config.is_model_parallel = False
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model_path, config=config,
            device_map={"": device}, torch_dtype=torch.bfloat16)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model_path, config=config,
            device_map={"": device}, torch_dtype=torch.float32).half()
    if args.peft is not None and args.teacher_peft_path is not None:
        if args.peft == "lora":
            model = PeftModel.from_pretrained(model, args.teacher_peft_path)
            model = model.merge_and_unload()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if dist.get_rank() == 0:
        print(f" > teacher params: {sum(p.numel() for p in model.parameters()):,}")
    return model


def prepare_dataset(args, tokenizer):
    rng = random.Random(args.seed)
    data = {
        "train": LMTrainDataset(args, tokenizer, args.data_dir, "train",
                                args.train_num, args.train_ratio, rng),
        "dev":   LMTrainDataset(args, tokenizer, args.data_dir, "valid",
                                args.dev_num, args.dev_ratio, rng),
    }
    print_rank(f"train={len(data['train'])}, dev={len(data['dev'])}")
    return data


def pretrain(args, projector_engine, dataset, device, teacher):
    """Main Stage-1 training loop."""
    dp_world = dist.get_world_size()
    dp_rank = dist.get_rank()

    sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True,
                                 rank=dp_rank, num_replicas=dp_world)
    train_dl = DataLoader(dataset["train"], sampler=sampler,
                          batch_size=args.batch_size,
                          collate_fn=dataset["train"].collate)

    best_val_loss = float("inf")
    teacher_dtype = next(teacher.parameters()).dtype
    lm_head = teacher.lm_head  # frozen LM head W_T

    for epoch in range(args.projector_pretrain_epochs):
        sampler.set_epoch(epoch)
        projector_engine.train()
        total_loss, n_steps = 0.0, 0

        for model_batch, no_model_batch, gen_data in train_dl:
            dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)

            with torch.no_grad():
                t_out = teacher(**model_batch, output_hidden_states=True, use_cache=False)
                h_T = t_out.hidden_states[-1].float()  # (B, L, d_T)

            # ProjectorTA: P_T->A -> P_A->T
            _z, h_recon = projector_engine(h_T)              # (B, L, d_T)
            # Paper Eq.3: CE summed over ALL positions i (prompt + response), only padding ignored.
            # Restricting to response tokens shrinks signal too much (~50 vs ~256 tokens/sample
            # on Dolly) and causes the projector to overfit.
            logits = lm_head(h_recon.to(teacher_dtype))      # (B, L, V_T)
            input_ids  = model_batch["input_ids"]             # (B, L)
            attn_mask  = model_batch["attention_mask"]        # (B, L) 1=real 0=pad
            logits_s   = logits[:, :-1].contiguous()          # (B, L-1, V)
            targets    = input_ids[:, 1:].contiguous()        # (B, L-1)
            targets    = targets.masked_fill(attn_mask[:, 1:] == 0, -100)
            loss = F.cross_entropy(
                logits_s.float().view(-1, logits_s.size(-1)),
                targets.view(-1), ignore_index=-100)

            projector_engine.backward(loss)
            projector_engine.step()

            dist.all_reduce(loss, dist.ReduceOp.SUM)
            total_loss += loss.item() / dp_world
            n_steps += 1

        avg_loss = total_loss / max(n_steps, 1)
        val_loss = validate(args, projector_engine, dataset["dev"],
                            device, teacher, teacher_dtype, lm_head)

        if dist.get_rank() == 0:
            log = (f"epoch {epoch+1}/{args.projector_pretrain_epochs} | "
                   f"train_loss={avg_loss:.4f} | val_loss={val_loss:.4f}")
            print(log)
            save_rank(log, os.path.join(args.save, "log.txt"))

            ckpt_latest = os.path.join(args.save, "projector_latest.pt")
            torch.save(projector_engine.module.state_dict(), ckpt_latest)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_best = os.path.join(args.save, "projector_best.pt")
                torch.save(projector_engine.module.state_dict(), ckpt_best)
                print(f"  -> new best saved ({best_val_loss:.4f})")
        dist.barrier()


def validate(args, projector_engine, dev_dataset, device, teacher, teacher_dtype, lm_head):
    projector_engine.eval()
    dp_world = dist.get_world_size()
    dp_rank = dist.get_rank()
    sampler = DistributedSampler(dev_dataset, shuffle=False, drop_last=False,
                                 rank=dp_rank, num_replicas=dp_world)
    dl = DataLoader(dev_dataset, sampler=sampler, batch_size=args.eval_batch_size,
                    collate_fn=dev_dataset.collate)
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for model_batch, no_model_batch, gen_data in dl:
            dev_dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
            t_out = teacher(**model_batch, output_hidden_states=True, use_cache=False)
            h_T = t_out.hidden_states[-1].float()
            _z, h_recon = projector_engine(h_T)
            logits = lm_head(h_recon.to(teacher_dtype))
            input_ids = model_batch["input_ids"]
            attn_mask = model_batch["attention_mask"]
            logits_s  = logits[:, :-1].contiguous()
            targets   = input_ids[:, 1:].contiguous()
            targets   = targets.masked_fill(attn_mask[:, 1:] == 0, -100)
            loss = F.cross_entropy(
                logits_s.float().view(-1, logits_s.size(-1)),
                targets.view(-1), ignore_index=-100)
            dist.all_reduce(loss, dist.ReduceOp.SUM)
            total_loss += loss.item() / dp_world
            n += 1
    return total_loss / max(n, 1)


def main():
    torch.backends.cudnn.enabled = False
    args = get_args()
    initialize(args)

    device = torch.cuda.current_device()
    os.makedirs(args.save, exist_ok=True)

    if dist.get_rank() == 0:
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10_000_000
    args.fp32 = not ds_config["fp16"]["enabled"]
    args.bf16 = "bf16" in ds_config and ds_config["bf16"]["enabled"]
    args.deepspeed_config = None

    # tokenizer == Qwen (teacher) tokenizer
    tokenizer = get_tokenizer(args)
    dataset = prepare_dataset(args, tokenizer)

    dp_world = dist.get_world_size()
    iters_per_epoch = int(
        len(dataset["train"]) / (args.batch_size * dp_world * args.gradient_accumulation_steps))
    total_iters = iters_per_epoch * args.projector_pretrain_epochs

    teacher = get_teacher(args, device)

    d_T = teacher.config.hidden_size  # Qwen1.5-1.8B: 2048
    d_A = args.d_bottleneck           # default 64
    projector = ProjectorTA(d_T, d_A).to(device)

    if dist.get_rank() == 0:
        print(f"ProjectorTA: d_T={d_T}, d_A={d_A}, "
              f"params={sum(p.numel() for p in projector.parameters()):,}")

    # Paper Stage 1: lr=1e-3, weight_decay=1e-4, cosine to lr_min
    optimizer = AdamW(projector.parameters(), lr=args.projector_lr,
                      weight_decay=args.weight_decay)
    # Pass scheduler via config to avoid DeepSpeed type-check issues
    # DeepSpeed will call scheduler.step() each micro-batch step
    ds_config["scheduler"] = {
        "type": "WarmupCosineLR",
        "params": {
            "total_num_steps": total_iters,
            "warmup_min_ratio": 0.0,
            "warmup_num_steps": 0,
        }
    }

    projector_engine, optimizer, _, _ = deepspeed.initialize(
        model=projector,
        optimizer=optimizer,
        args=args,
        mpu=None,
        config_params=ds_config,
    )

    pretrain(args, projector_engine, dataset, device, teacher)


if __name__ == "__main__":
    main()
