from typing import Optional, Tuple, Any
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from torch import Tensor
from utils import get_span_hidden_states, get_span_hidden_states_custom
from peft import PeftModel

import logging
logger = logging.getLogger(__name__)



class CustomsQwen3Attention(torch.nn.Module):
    def __init__(self, original_self_attn):
        super().__init__()
        self.original = original_self_attn

    def forward(self, **kwargs):
        kwargs['output_attentions'] = False
        return self.original(**kwargs)

class CustomsMistralAttention(torch.nn.Module):
    def __init__(self, original_self_attn):
        super().__init__()
        self.original = original_self_attn

    def forward(self, **kwargs):
        kwargs['output_attentions'] = False
        return self.original(**kwargs)
    
class CustomsGPT2Attention(torch.nn.Module):
    def __init__(self, original_self_attn):
        super().__init__()
        self.original = original_self_attn

    def forward(self, hidden_states, **kwargs):
        kwargs['output_attentions'] = False
        return self.original(hidden_states, **kwargs)


@dataclass
class TeacherOutput(ModelOutput):
    logits: Optional[Tensor] = None
    hidden_states: Any = None
    span_states: Any = None
    span_weights: Any = None


class Teacher:
    def __init__(self, model_name, load_model_kwargs, export_hidden_state_layers,
                 weight_pooling=True, span_weight=True, sft_path=None):

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_model_kwargs)
        if sft_path is not None:
            self.model = PeftModel.from_pretrained(self.model, sft_path)
            self.model = self.model.merge_and_unload()
        self.model = self.model.eval()
        # self.model.config.use_cache = False

        self.device = self.model.device

        self.export_hidden_state_layers = export_hidden_state_layers
    

        self.weight_pooling = weight_pooling
        self.span_weight = span_weight

        if weight_pooling and span_weight:
            self.get_span_hidden_states = get_span_hidden_states
        else:
            self.get_span_hidden_states = get_span_hidden_states_custom

    def decode(self, inputs) -> TeacherOutput:
        return None


class TeacherMistral7B(Teacher):
    def __init__(self, model_name, load_model_kwargs,
                 export_hidden_state_layers=[2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35], 
                 sentence_mean_pooling=False, weight_pooling=True, span_weight=True, sft_path=None):

        print('TeacherMistral7B loading model ...')

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.output_hidden_states = load_model_kwargs.pop('output_hidden_states', False)
        config.output_attentions = load_model_kwargs.pop('output_attentions', False)
        load_model_kwargs['config'] = config

        super().__init__(model_name, load_model_kwargs,
                         export_hidden_state_layers, 
                         weight_pooling, span_weight, sft_path)

        for i, layer in enumerate(self.model.model.layers):
                if (i + 1) in self.export_hidden_state_layers: 
                    continue
                layer.self_attn = CustomsMistralAttention(layer.self_attn)

        self.model = self.model.eval()

    def decode(self, inputs):
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        safe_idx = inputs.pop('pooler_safe_idx', None)
        pooler_mask = inputs.pop('pooler_mask', None)

        with torch.no_grad():
            outputs = self.model(**inputs)

        hidden_states = outputs.hidden_states
        attentions = outputs.attentions

        span_states, span_weights = None, None
        if safe_idx is not None and hidden_states is not None:
            span_states, span_weights = self.get_span_hidden_states(inputs, hidden_states, 
                                                                    attentions, safe_idx, pooler_mask,
                                                                    inputs['attention_mask'], 
                                                                    self.export_hidden_state_layers, 
                                                                    self.weight_pooling, self.span_weight, 
                                                                    is_causal=True)
        if hidden_states is not None:
            hidden_states = torch.stack([outputs.hidden_states[i] for i in self.export_hidden_state_layers])
        

        return TeacherOutput(
            logits = outputs.logits,
            hidden_states = hidden_states,
            span_states = span_states,
            span_weights = span_weights
        )


class TeacherQwen(Teacher):
    def __init__(self, model_name, load_model_kwargs,
                 export_hidden_state_layers=[2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35],
                   weight_pooling=True, span_weight=True, sft_path=None):

        print('TeacherQwen loading model ...')

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.output_hidden_states = load_model_kwargs.pop('output_hidden_states', False)
        config.output_attentions = load_model_kwargs.pop('output_attentions', False)
        load_model_kwargs['config'] = config

        super().__init__(model_name, load_model_kwargs,
                         export_hidden_state_layers, 
                         weight_pooling, span_weight, sft_path)
        
        for i, layer in enumerate(self.model.model.layers):
            if (i + 1) in self.export_hidden_state_layers: 
                continue
            layer.self_attn = CustomsQwen3Attention(layer.self_attn)


    def decode(self, inputs) -> TeacherOutput:
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        safe_idx = inputs.pop('pooler_safe_idx', None)
        pooler_mask = inputs.pop('pooler_mask', None)

        with torch.no_grad():
            outputs = self.model(**inputs)

        hidden_states = outputs.hidden_states
        attentions = outputs.attentions

        span_states, span_weights = None, None
        if safe_idx is not None and hidden_states is not None:
            span_states, span_weights = self.get_span_hidden_states(inputs, hidden_states, 
                                                                    attentions, safe_idx, pooler_mask,
                                                                    inputs['attention_mask'], 
                                                                    self.export_hidden_state_layers, 
                                                                    self.weight_pooling, self.span_weight, 
                                                                    is_causal=True)
        if hidden_states is not None:
            hidden_states = torch.stack([outputs.hidden_states[i] for i in self.export_hidden_state_layers])
        

        return TeacherOutput(
            logits = outputs.logits,
            hidden_states = hidden_states,
            span_states = span_states,
            span_weights = span_weights
        )


class TeacherGPT2(Teacher):
    def __init__(self, model_name, load_model_kwargs,
                 export_hidden_state_layers=[2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35],
                   weight_pooling=True, span_weight=True, sft_path=None):

        print('TeacherGPT2 loading model ...')

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.output_hidden_states = load_model_kwargs.pop('output_hidden_states', False)
        config.output_attentions = load_model_kwargs.pop('output_attentions', False)
        load_model_kwargs['config'] = config

        super().__init__(model_name, load_model_kwargs,
                         export_hidden_state_layers, 
                         weight_pooling, span_weight, sft_path)

        
        if config.output_attentions:
            for i, layer in enumerate(self.model.transformer.h):
                if (i + 1) in self.export_hidden_state_layers: 
                    continue
                layer.attn = CustomsGPT2Attention(layer.attn)


    def decode(self, inputs) -> TeacherOutput:
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        safe_idx = inputs.pop('pooler_safe_idx', None)
        pooler_mask = inputs.pop('pooler_mask', None)

        with torch.no_grad():
            outputs = self.model(**inputs)

        hidden_states = outputs.hidden_states
        attentions = outputs.attentions

        span_states, span_weights = None, None
        if safe_idx is not None and hidden_states is not None:
            span_states, span_weights = self.get_span_hidden_states(inputs, hidden_states, 
                                                                    attentions, safe_idx, pooler_mask,
                                                                    inputs['attention_mask'], 
                                                                    self.export_hidden_state_layers, 
                                                                    self.weight_pooling, self.span_weight, 
                                                                    is_causal=True)
        if hidden_states is not None:
            hidden_states = torch.stack([outputs.hidden_states[i] for i in self.export_hidden_state_layers])
        

        return TeacherOutput(
            logits = outputs.logits,
            hidden_states = hidden_states,
            span_states = span_states,
            span_weights = span_weights
        )