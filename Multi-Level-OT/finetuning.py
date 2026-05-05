import os
import argparse
import random
import torch

from configs import dataset as DATA_CONFIG
from configs import fsdp_config as FSDP_CONFIG
from configs import train_config as TRAIN_CONFIG
from configs import distillation_config as DISTIL_CONFIG

from train.train_utils import train
from configs.configs_utils import update_config
from data.data_utils import (get_dataloader, get_distillation_dataloader)
from train.tools import (setup, setup_environ_flags, clear_gpu_cache)
from models.models_utils import (get_model, get_distillation_models, get_optimizer)

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Suppress the background safetensors auto-conversion thread that transformers
# spawns when loading sharded .bin models (e.g. VoCuc/Mistral7B_Dolly_SFT).
# The thread only tries to create a HF Hub PR — it has no effect on model loading.
try:
    import transformers.safetensors_conversion as _sc
    _sc.auto_conversion = lambda *args, **kwargs: None
except Exception:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning script")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset.file", type=str, required=True, help="Path to the dataset loader")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size_training", type=int, default=4, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=4, help="Validation batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--context_length", type=int, default=None, help="Max token length; sequences longer than this are filtered out")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory path")
    parser.add_argument("--distillation_config_model_name", type=str, help="Model name for distillation")
    parser.add_argument("--distillation", action="store_true", help="Enable distillation")
    parser.add_argument("--distillation_config_enable_fsdp", action="store_true", help="Enable FSDP for distillation")
    parser.add_argument("--distillation_config_pure_bf16", action="store_true", help="Use pure BF16 for teacher model")
    parser.add_argument("--distillation_config_distil_factor", type=float, default=1.5, help="Weight for distillation loss")
    parser.add_argument("--distillation_config_cross_entropy_factor", type=float, default=1.0, help="Weight for cross-entropy loss")
    parser.add_argument("--distillation_config_student_temperature", type=float, default=1.0, help="Student softmax temperature")
    parser.add_argument("--distillation_config_teacher_temperature", type=float, default=1.0, help="Teacher softmax temperature")
    parser.add_argument("--save_step", type=int, default=100, help="Save step")
    parser.add_argument("--f", type=int, default=1, help="Distillation method (1=sort, 2=greedy, other=raw)")
    # MTA Span + Entropy Weight
    parser.add_argument("--entropy_weight", action="store_true", help="Use teacher entropy as token weight in span loss")
    parser.add_argument("--span_loss_weight", type=float, default=0.0, help="Lambda for span loss (0=disabled)")
    parser.add_argument("--student_layer_mapping", type=str, default="", help="Comma-separated student hidden_states layer indices, e.g. '8,16,24'")
    parser.add_argument("--teacher_layer_mapping", type=str, default="", help="Comma-separated teacher hidden_states layer indices, e.g. '16,24,32'")
    parser.add_argument("--split_layer_mapping", type=str, default="0,1,2", help="Split indices for word vs phrase layers, e.g. '0,1,2'")
    parser.add_argument("--use_phrase_spans", action="store_true", default=True, help="Use spaCy NP/VP phrase spans for higher layers")
    parser.add_argument("--student_hidden_size", type=int, default=0, help="Student hidden size (0=auto-detect)")
    parser.add_argument("--teacher_hidden_size", type=int, default=0, help="Teacher hidden size (0=auto-detect)")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing for student (saves activation memory at ~30% speed cost)")
    # Multi-GPU: MTA-style model parallelism
    parser.add_argument("--student_device", type=str, default="cuda:0", help="Device for student model, e.g. cuda:0")
    parser.add_argument("--teacher_device", type=str, default="cuda:1", help="Device for teacher model, e.g. cuda:1 or auto")
    # LoRA / PEFT
    parser.add_argument("--use_peft", action="store_true", help="Enable LoRA PEFT for student model")
    parser.add_argument("--lora_r", type=int, default=None, help="LoRA rank (default: 8)")
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=None, help="LoRA dropout (default: 0.05)")
    return parser.parse_args()

def main():
    args = parse_args()

    train_config, fsdp_config, distil_config, data_config = TRAIN_CONFIG(), FSDP_CONFIG(), DISTIL_CONFIG(), DATA_CONFIG()
    update_config((train_config, fsdp_config, data_config), **vars(args))
    update_config((distil_config), isSubmodule=True, **vars(args))

    # Remap --lora_r → args.r vì lora_config dataclass dùng field tên "r"
    # update_config() sẽ tự match khi gọi generate_peft_config
    if args.lora_r is not None:
        args.r = args.lora_r

    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    setup()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    torch.cuda.set_device(local_rank)
    clear_gpu_cache(local_rank)
    setup_environ_flags(rank)

    # Load Model and Tokenizer
    if train_config.distillation:
        distil_config.model_name = args.distillation_config_model_name
        # Fix: update_config với isSubmodule=True không xử lý key không có dấu "."
        # nên phải gán thủ công tất cả distil_config fields từ args
        distil_config.pure_bf16 = args.distillation_config_pure_bf16
        distil_config.enable_fsdp = args.distillation_config_enable_fsdp
        distil_config.distil_factor = args.distillation_config_distil_factor
        distil_config.cross_entropy_factor = args.distillation_config_cross_entropy_factor
        distil_config.student_temperature = args.distillation_config_student_temperature
        distil_config.teacher_temperature = args.distillation_config_teacher_temperature
        # MTA Span + Entropy Weight config
        distil_config.entropy_weight = args.entropy_weight
        distil_config.span_loss_weight = args.span_loss_weight
        distil_config.student_layer_mapping = args.student_layer_mapping
        distil_config.teacher_layer_mapping = args.teacher_layer_mapping
        distil_config.split_layer_mapping = args.split_layer_mapping
        distil_config.use_phrase_spans = args.use_phrase_spans
        distil_config.student_hidden_size = args.student_hidden_size
        distil_config.teacher_hidden_size = args.teacher_hidden_size
        distil_config.student_device = args.student_device
        distil_config.teacher_device = args.teacher_device

        # DDP: each rank owns its own copy of both models on its local GPU
        if torch.distributed.get_world_size() > 1:
            distil_config.student_device = f"cuda:{local_rank}"
            distil_config.teacher_device = f"cuda:{local_rank}"

        student_tokenizer, teacher_tokenizer, model = get_distillation_models(
            train_config, distil_config, fsdp_config, rank, vars(args))
    else:
        tokenizer, model = get_model(train_config, fsdp_config, rank, vars(args))
    if rank == 0: print(model)

    # DDP: wrap model so student gradients are all-reduced across ranks
    if torch.distributed.get_world_size() > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Load Data
    data_config.encoder_decoder = train_config.encoder_decoder
    if train_config.distillation:
        train_dataloader, teacher_train_dataloader, eval_dataloader, teacher_eval_dataloader = get_distillation_dataloader(data_config, train_config, distil_config, student_tokenizer, teacher_tokenizer, rank)
    else:
        train_dataloader, eval_dataloader = get_dataloader(data_config, train_config, tokenizer, rank)

    # Get the optimizer and learning rate scheduler
    optimizer = get_optimizer(model, train_config, fsdp_config)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=train_config.lr, epochs=train_config.num_epochs, steps_per_epoch=len(train_dataloader),
                                                    pct_start=train_config.pct_start, div_factor=train_config.div_factor, final_div_factor=train_config.final_div_factor)

    f = train_config.f
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        distil_config,
        data_config,
        teacher_train_dataloader if train_config.distillation else None,
        teacher_eval_dataloader if train_config.distillation else None,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank,
        rank,
        f,
        student_tokenizer=student_tokenizer if train_config.distillation else None,
        teacher_tokenizer=teacher_tokenizer if train_config.distillation else None,
    )
    if rank == 0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

if __name__ == "__main__":
    main()
