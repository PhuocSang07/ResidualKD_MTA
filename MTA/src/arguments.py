from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class Arguments:
    train_data: str = field(default=None, metadata={"help": "Path to training data"})
    val_data: str = field(default=None, metadata={"help": "Path to validation data"})
    test_data: str = field(default=None, metadata={"help": "Path to test data"})

    num_labels: int = field(default=2, metadata={"help": "Number of labels"})

    batch_size: int = field(default=8)
    val_batch_size: int = field(default=32)

    max_len: int = field(
        default=256,
        metadata={"help": "Max total sequence length after tokenization."},
    )

    pad_to_multiple_of: int = field(default=2)

    temperature: Optional[float] = field(default=2.0)
    distill_temperature: Optional[float] = field(default=2.0)

    knowledge_distillation: bool = field(default=True)
    finetune_hidden_states: bool = field(default=True)
    output_attentions: bool = field(default=False)

    teach_device: str = field(default='cuda:1')
    student_device: str = field(default='cuda:0')

    num_train_epochs: int = field(default=1)

    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.1)

    hard_label_loss_weight: float = field(default=1.0)

    finetune_embedding: bool = field(default=False)

    p: float = field(default=1.0)

    # MTA span/feature distillation (mta_dskd_v2 style)
    MTA_mode: bool = field(default=False, metadata={"help": "Enable MTA span/feature distillation"})
    w_span_loss: float = field(default=1.0, metadata={"help": "Weight for MTA span loss"})
    entropy_weight: bool = field(default=False, metadata={"help": "Use teacher entropy as token weight"})
    projector_lr: float = field(default=5e-4, metadata={"help": "Learning rate for MTA projectors"})

    # Absolute layer indices into model.hidden_states tuple
    student_layer_mapping: List[int] = field(default_factory=list,
                                             metadata={"help": "Absolute student layer indices for MTA"})
    teacher_layer_mapping: List[int] = field(default_factory=list,
                                             metadata={"help": "Absolute teacher layer indices for MTA"})
    # [word_start, word_end / span_start, span_end] indices into the projectors / layer mappings
    split_layer_mapping: List[int] = field(default_factory=lambda: [0, 0, 0],
                                           metadata={"help": "Split between word- and span-level layers"})

    output_dir: Optional[str] = field(default=None)

    teacher_model: str = field(default='')
    teacher_tokenizer: str = field(default='')
    student_model: str = field(default='google-bert/bert-base-uncased')
    student_tokenizer: str = field(default='google-bert/bert-base-uncased')
    hf_token: str = field(default='')

    load_student_tokenizer_kwargs: dict = field(default_factory=dict)
    load_teacher_tokenizer_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.train_data is not None and not os.path.exists(self.train_data):
            raise FileNotFoundError(f"cannot find file: {self.train_data}")

        if self.MTA_mode:
            if len(self.teacher_layer_mapping) != len(self.student_layer_mapping):
                raise ValueError("teacher_layer_mapping and student_layer_mapping must have the same length when MTA_mode is on")
            if len(self.split_layer_mapping) < 3:
                raise ValueError("split_layer_mapping needs at least 3 values: [word_start, word_end/span_start, span_end]")
            if self.split_layer_mapping[-1] > len(self.student_layer_mapping):
                raise ValueError("split_layer_mapping[-1] cannot exceed len(student_layer_mapping)")
