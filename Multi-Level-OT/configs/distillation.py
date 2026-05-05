from dataclasses import dataclass
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

@dataclass
class distillation_config:
    model_name: str = "meta-llama/Llama-2-7b-hf"
    enable_fsdp: bool = False
    low_cpu_fsdp: bool = False
    quantization: bool = False
    use_fast_kernels: bool = False
    use_peft: bool = False
    freeze_layers: bool = False
    num_freeze_layers: int = 0
    cross_entropy_factor: float = 1
    distil_factor: float = 1.5
    student_temperature: float = 1
    teacher_temperature: float = 1
    encoder_decoder: bool = False

    # MTA Span + Entropy Weight Integration
    entropy_weight: bool = False          # use teacher entropy weight instead of attention centrality
    span_loss_weight: float = 0.0         # lambda weight for overall span loss (0 = disabled)
    student_layer_mapping: str = ""       # comma-separated hidden_states indices, e.g. "8,16,24"
    teacher_layer_mapping: str = ""       # comma-separated hidden_states indices, e.g. "16,24,32"
    split_layer_mapping: str = "0,1,2"   # split indices: [0:1]=word layers, [1:2]=phrase layers
    use_phrase_spans: bool = True         # use spaCy NP/VP for higher layers; False=word only
    student_hidden_size: int = 0          # auto-detected from model if 0
    teacher_hidden_size: int = 0          # auto-detected from model if 0

    # Multi-GPU: MTA-style model parallelism (teacher on one GPU, student on another)
    student_device: str = "cuda:0"
    teacher_device: str = "cuda:1"

    # FSDP Config
    mixed_precision: bool = False
    use_fp16: bool = False
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    checkpoint_type: StateDictType = StateDictType.SHARDED_STATE_DICT
    fsdp_activation_checkpointing: bool = True
    fsdp_cpu_offload: bool = False
    pure_bf16: bool = False
    optimizer: str = "AdamW"