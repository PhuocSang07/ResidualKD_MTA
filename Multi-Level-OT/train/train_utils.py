import os
import time
import torch
import wandb
os.environ["WANDB_MODE"] = "dryrun"
import torch.distributed as dist

from tqdm import tqdm
from contextlib import nullcontext
from models.memory import MemoryTrace
from train.tools import clear_gpu_cache
from train.evaluations import evaluation
from train.save import save_train_params, save_model
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from models.distillation_model import DistillationLoss, preprocess_distillation_batch


class SpanExtractor:
    """
    Decodes token IDs back to text, re-tokenizes with offset mapping,
    and extracts word/phrase spans for span-based distillation loss.
    """
    def __init__(self, student_tokenizer, teacher_tokenizer, distil_config):
        self.student_tok = student_tokenizer
        self.teacher_tok = teacher_tokenizer
        self.config = distil_config
        self.nlp = None
        self.matcher = None

        if distil_config.use_phrase_spans:
            try:
                import spacy
                from spacy.matcher import Matcher
                self.nlp = spacy.load("en_core_web_sm")
                self.matcher = Matcher(self.nlp.vocab)
                # VP pattern: verb followed by optional adverbs
                self.matcher.add("VP", [[{"POS": "VERB"}, {"POS": "ADV", "OP": "*"}]])
            except Exception as e:
                print(f"[SpanExtractor] spaCy unavailable ({e}), falling back to word spans only.")

    def _word_char_offsets(self, text):
        offsets = []
        pos = 0
        for word in text.split():
            start = text.find(word, pos)
            if start == -1:
                continue
            offsets.append((start, start + len(word)))
            pos = start + len(word)
        return offsets

    def extract(self, student_input_ids, teacher_input_ids,
                student_attention_mask, teacher_attention_mask, device):
        student_texts = self.student_tok.batch_decode(
            student_input_ids.cpu(), skip_special_tokens=False)
        teacher_texts = self.teacher_tok.batch_decode(
            teacher_input_ids.cpu(), skip_special_tokens=False)

        s_enc = self.student_tok(
            student_texts, return_offsets_mapping=True,
            padding=True, add_special_tokens=False, return_tensors='pt')
        t_enc = self.teacher_tok(
            teacher_texts, return_offsets_mapping=True,
            padding=True, add_special_tokens=False, return_tensors='pt')

        s_offsets = s_enc['offset_mapping'].to(device)
        t_offsets = t_enc['offset_mapping'].to(device)

        if self.nlp is not None and self.matcher is not None:
            from models.span_utils import get_spans_offsets
            spans_offsets, words_offsets = get_spans_offsets(
                teacher_texts, self.nlp, self.matcher)
        else:
            spans_offsets = [[] for _ in teacher_texts]
            words_offsets = [self._word_char_offsets(t) for t in teacher_texts]

        return {
            's_attention_mask': student_attention_mask.to(device),
            't_attention_mask': teacher_attention_mask.to(device),
            's_offsets_mapping': s_offsets,
            't_offsets_mapping': t_offsets,
            'spans_offsets': spans_offsets,
            'words_offsets': words_offsets,
        }

def train(model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, distil_config, dataset_config, teacher_train_dataloader=None, teacher_eval_dataloader=None, fsdp_config=None, local_rank=None, rank=None, f=1, student_tokenizer=None, teacher_tokenizer=None):
    # Unwrap DDP/FSDP to access underlying DistillationModel attributes
    raw_model = model.module if hasattr(model, 'module') else model

    # Weights & Biases tracking system initialization.
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    if rank == 0:
        wandb.init(
            project=f"llm_distillation_{dataset_config.file.split('/')[-1][:-3]}",
            name=f"{train_config.model_name.split('/')[-1]}-{raw_model.teacher.name_or_path.split('/')[-1]}-d{distil_config.distil_factor}-t{distil_config.teacher_temperature}{distil_config.student_temperature}" if train_config.distillation else f"{train_config.model_name.split('/')[-1]}",
            config={
                "model_name": train_config.model_name.split('/')[-1],
                "dataset": dataset_config.file.split('/')[-1],
                "batch_size_training": train_config.batch_size_training,
                "val_batch_size": train_config.val_batch_size,
                "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
                "num_epochs": train_config.num_epochs,
                "lr": train_config.lr,
                "weight_decay": train_config.weight_decay,
                "pct_start": train_config.pct_start,
                "div_factor": train_config.div_factor,
                "final_div_factor": train_config.final_div_factor,
                "seed": train_config.seed,
                "use_fp16": train_config.use_fp16,
                "mixed_precision": train_config.mixed_precision,
                "peft_method": train_config.peft_method,
                "use_peft": train_config.use_peft,
                "freeze_layers": train_config.freeze_layers,
                "num_freeze_layers": train_config.num_freeze_layers,
                "quantization": train_config.quantization,
                "cross_entropy_factor": distil_config.cross_entropy_factor if train_config.distillation else -1,
                "distil_factor": distil_config.distil_factor if train_config.distillation else -1,
                "student_temperature": distil_config.student_temperature if train_config.distillation else -1,
                "teacher_temperature": distil_config.teacher_temperature if train_config.distillation else -1
            }
        )

    # Init distillation loss if distillation is enabled
    if train_config.distillation:
        distillation_loss = DistillationLoss(
            crossentropy_weight=distil_config.cross_entropy_factor,
            distillation_weight=distil_config.distil_factor,
            student_temperature=distil_config.student_temperature,
            teacher_temperature=distil_config.teacher_temperature,
            skip_student_eos=True, debug=False, debug_rank=0,
            tokenizer_student=raw_model.student.name_or_path,
            tokenizer_teacher=raw_model.teacher.name_or_path,
            f=f,
            span_loss_weight=distil_config.span_loss_weight,
            distil_config=distil_config,
        )

        span_extractor = None
        if distil_config.span_loss_weight > 0 and student_tokenizer is not None and teacher_tokenizer is not None:
            span_extractor = SpanExtractor(student_tokenizer, teacher_tokenizer, distil_config)
        
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp or distil_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext

    train_prep = []
    train_loss = []
    val_ppl = []
    val_loss = []
    epoch_times = []
    checkpoint_times = []
    results = {}
    steps_per_eval = len(eval_dataloader)
    steps_per_epoch = len(train_dataloader)
    best_val_loss = float("inf")
    for epoch in range(train_config.num_epochs):
        epoch_start_time = time.perf_counter()
        total_length = steps_per_epoch//gradient_accumulation_steps
        raw_model.student.train() if train_config.distillation else model.train()
        with MemoryTrace() as memtrace:
            total_loss = 0.0
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            for step, batch in enumerate(train_dataloader if not train_config.distillation else zip(train_dataloader, teacher_train_dataloader)):
                if train_config.distillation: batch = preprocess_distillation_batch(batch)
                # MTA-style: DistillationModel.forward() handles per-device placement.
                # Move all tensors to CPU here; the model moves them to the right GPU.
                for key in batch.keys():
                    if local_rank is not None:
                        batch[key] = batch[key].to(f"cuda:{local_rank}")
                    else:
                        batch[key] = batch[key].to('cpu')

                with autocast():
                    if train_config.distillation:
                        student_output, teacher_output = model(**batch)

                        span_data = None
                        if span_extractor is not None:
                            dev = raw_model.student_device
                            span_data = span_extractor.extract(
                                batch['student_input_ids'],
                                batch['teacher_input_ids'],
                                batch['student_attention_mask'],
                                batch['teacher_attention_mask'],
                                dev,
                            )

                        loss, cross_loss, dist_loss, span_loss = distillation_loss(
                            epoch, student_output, teacher_output,
                            batch['student_labels'], batch['teacher_labels'],
                            rank=rank, span_data=span_data,
                            projectors=getattr(raw_model, 'projectors', None),
                        )
                        teacher_loss = teacher_output.loss.detach().float()
                        # Free teacher outputs (no grad_fn) and span_data early to reduce peak VRAM
                        del teacher_output, span_data
                    else:
                        loss = model(**batch).loss

                loss = loss / gradient_accumulation_steps
                total_loss += loss.detach().float()
                if train_config.use_fp16:
                    scaler.scale(loss).backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == steps_per_epoch - 1:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        pbar.update()
                else:
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == steps_per_epoch - 1:
                        # Clip ALL trainable params (student + projectors), not just student
                        all_trainable = [p for p in model.parameters() if p.requires_grad]
                        torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
                        # Skip step if any gradient is NaN/Inf to prevent weight corruption
                        has_bad_grad = any(
                            p.grad is not None and not torch.isfinite(p.grad).all()
                            for p in all_trainable
                        )
                        if has_bad_grad:
                            if rank == 0:
                                print(f"[step {step}] NaN/Inf gradient detected — skipping optimizer step")
                            optimizer.zero_grad()
                        else:
                            optimizer.step()
                            optimizer.zero_grad()
                        pbar.update()

                if rank == 0:
                    if train_config.distillation:
                        wandb.log({
                            "train_loss": loss.detach().float(),
                            "cross_loss": cross_loss.detach().float(),
                            "distil_loss": dist_loss.detach().float(),
                            "span_loss": span_loss.detach().float(),
                            "teacher_loss": teacher_loss,
                            "lr": optimizer.param_groups[0]['lr']
                        })
                    else:
                        wandb.log({
                            "train_loss": loss.detach().float(),
                            "lr": optimizer.param_groups[0]['lr']
                        })

                lr_scheduler.step()
                pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{steps_per_epoch} completed (loss: {loss.detach().float()})")

                if (train_config.run_validation and ((step+1) % train_config.save_step == 0 or step+1 == steps_per_epoch)):
                    if rank == 0: print("Running evaluation...")
                    # model.eval()
                    # eval_ppl, eval_epoch_loss, eval_cross_loss, eval_dist_loss = evaluation(
                    #     model, train_config, distil_config, 
                    #     eval_dataloader if not train_config.distillation else zip(eval_dataloader, teacher_eval_dataloader),
                    #     steps_per_eval, local_rank, epoch)
                    # model.student.train() if train_config.distillation else model.train()
                    # val_loss.append(eval_epoch_loss)
                    # val_ppl.append(eval_ppl)
                    
                    # if rank == 0:
                    #     print(f"Perplexity {eval_ppl}, loss {eval_epoch_loss}")
                    #     if train_config.distillation:
                    #         wandb.log({
                    #             "eval_ppl": eval_ppl,
                    #             "eval_epoch_loss": eval_epoch_loss,
                    #             "eval_cross_loss": eval_cross_loss,
                    #             "eval_dist_loss": eval_dist_loss
                    #         })
                    #     else:
                    #         wandb.log({
                    #             "eval_ppl": eval_ppl,
                    #             "eval_epoch_loss": eval_epoch_loss,
                    #         })

                    # if eval_epoch_loss < best_val_loss or train_config.save_all:
                    if True:
                        # if eval_epoch_loss < best_val_loss:
                        #     best_val_loss = eval_epoch_loss
                            # if rank == 0:
                            #     print(f"best eval loss is {best_val_loss}")
                        if train_config.save_model:
                            checkpoint_start_time = time.perf_counter()
                            save_model(
                                model,
                                optimizer, ((steps_per_epoch*epoch)+step), train_config, distil_config, fsdp_config, rank
                            )
                            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                            checkpoint_times.append(checkpoint_end_time)
                    clear_gpu_cache(rank)
            pbar.close()
        
        if epoch == 0:
            distillation_loss.on_epoch_end()
        
        if rank == 0: print(memtrace)
        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)

        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            train_epoch_loss = total_loss / steps_per_epoch / dist.get_world_size()
        else:
            train_epoch_loss = total_loss / steps_per_epoch
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)

        if rank == 0:
            print(
                f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
            wandb.log({
                "train_perplexity": train_perplexity,
                "train_epoch_loss": train_epoch_loss,
                "train_epoch_time": epoch_end_time
            })

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = sum(
        checkpoint_times) / len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    # if train_config.run_validation:
    #     avg_eval_prep = sum(val_ppl)/len(val_ppl)
    #     avg_eval_loss = sum(val_loss)/len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    # if train_config.run_validation:
    #     results['avg_eval_prep'] = avg_eval_prep
    #     results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time

    if train_config.enable_fsdp and not train_config.use_peft:
        save_train_params(train_config, fsdp_config, rank)

    return results