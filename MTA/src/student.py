from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from torch import Tensor
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

import logging
logger = logging.getLogger(__name__)


@dataclass
class StudentOutput(ModelOutput):
    logits: Optional[Tensor] = None
    hidden_states: Any = None


class LLMModel(torch.nn.Module):
    def __init__(self, model_name, load_model_kwargs={}, lora_conf=None, sft_path=None):
        super().__init__()

        self.lora_config = lora_conf

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.output_hidden_states = load_model_kwargs.pop('output_hidden_states', False)
        config.output_attentions = load_model_kwargs.pop('output_attentions', False)
        load_model_kwargs['config'] = config

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_model_kwargs)

        if sft_path is not None:
            print("Loading adapter for student")
            self.model = PeftModel.from_pretrained(self.model, sft_path)
            self.model = self.model.merge_and_unload()

        if lora_conf is not None:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_conf.lora_rank,
                lora_alpha=lora_conf.lora_alpha,
                lora_dropout=lora_conf.lora_dropout
            )
            self.model = get_peft_model(self.model, lora_config).to(self.model.device)
            self.model.print_trainable_parameters()

        self.device = self.model.device

    def forward(self, inputs: Dict[str, Tensor] = None):
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        outputs = self.model(**inputs, use_cache=False, return_dict=True)

        if not self.training:
            return StudentOutput(logits=None)

        return StudentOutput(
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
        )

    def save(self, output_dir: str):
        self.model.save_pretrained(output_dir, state_dict=self.model.state_dict())
