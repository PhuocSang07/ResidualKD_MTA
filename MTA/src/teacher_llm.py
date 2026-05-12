from typing import Optional, Any
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from torch import Tensor
from peft import PeftModel

import logging
logger = logging.getLogger(__name__)


@dataclass
class TeacherOutput(ModelOutput):
    logits: Optional[Tensor] = None
    hidden_states: Any = None


class Teacher:
    def __init__(self, model_name, load_model_kwargs, sft_path=None):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_model_kwargs)
        if sft_path is not None:
            self.model = PeftModel.from_pretrained(self.model, sft_path)
            self.model = self.model.merge_and_unload()
        self.model = self.model.eval()
        self.device = self.model.device

    def decode(self, inputs) -> TeacherOutput:
        raise NotImplementedError


class TeacherMistral7B(Teacher):
    def __init__(self, model_name, load_model_kwargs, sft_path=None):
        print('TeacherMistral7B loading model ...')
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.output_hidden_states = load_model_kwargs.pop('output_hidden_states', True)
        config.output_attentions = load_model_kwargs.pop('output_attentions', False)
        load_model_kwargs['config'] = config
        super().__init__(model_name, load_model_kwargs, sft_path)

    def decode(self, inputs):
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return TeacherOutput(
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
        )


class TeacherQwen(Teacher):
    def __init__(self, model_name, load_model_kwargs, sft_path=None):
        print('TeacherQwen loading model ...')
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.output_hidden_states = load_model_kwargs.pop('output_hidden_states', True)
        config.output_attentions = load_model_kwargs.pop('output_attentions', False)
        load_model_kwargs['config'] = config
        super().__init__(model_name, load_model_kwargs, sft_path)

    def decode(self, inputs) -> TeacherOutput:
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return TeacherOutput(
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
        )


class TeacherGPT2(Teacher):
    def __init__(self, model_name, load_model_kwargs, sft_path=None):
        print('TeacherGPT2 loading model ...')
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.output_hidden_states = load_model_kwargs.pop('output_hidden_states', True)
        config.output_attentions = load_model_kwargs.pop('output_attentions', False)
        load_model_kwargs['config'] = config
        super().__init__(model_name, load_model_kwargs, sft_path)

    def decode(self, inputs) -> TeacherOutput:
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return TeacherOutput(
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
        )
