from typing import Optional, Dict, Any
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoConfig
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from torch import nn, Tensor
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from utils import get_span_hidden_states, get_span_hidden_states_custom
    
import os

import logging
logger = logging.getLogger(__name__)



@dataclass
class StudentOutput(ModelOutput):
    logits: Optional[Tensor] = None
    embeddings: Optional[Tensor] = None
    hidden_states: Any = None
    span_states: Any = None
    span_weights: Any = None

class LLMModel(torch.nn.Module):
    def __init__(self, model_name, load_model_kwargs = {}, hidden_layer_fineturn=[23], 
                 weight_pooling=True, span_weight=True, lora_conf=None, sft_path=None):
        super().__init__()

        self.hidden_layer_fineturn = hidden_layer_fineturn
        self.weight_pooling = weight_pooling
        self.span_weight = span_weight
        self.lora_config = lora_conf

        if weight_pooling and span_weight:
            self.get_span_hidden_states = get_span_hidden_states
        else:
            self.get_span_hidden_states = get_span_hidden_states_custom

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
        
        safe_idx = inputs.pop('pooler_safe_idx', None)
        pooler_mask = inputs.pop('pooler_mask', None)

        # outputs = self.model(**inputs, output_attentions=True, output_hidden_states=True, return_dict=True)
        outputs = self.model(**inputs, use_cache=False, return_dict=True)

        if not self.training:
            return StudentOutput(logits=None)
        
        if outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states
        else:
            hidden_states = None

        attentions = outputs.attentions

        span_states, span_weights = None, None
        if safe_idx is not None and hidden_states is not None:
            span_states, span_weights = self.get_span_hidden_states(inputs, hidden_states, 
                                                                    attentions, safe_idx, 
                                                                    pooler_mask, inputs['attention_mask'],
                                                                    self.hidden_layer_fineturn,
                                                                    self.weight_pooling, self.span_weight,
                                                                    is_causal=True)

        if hidden_states is not None:
            hidden_states = torch.stack([outputs.hidden_states[i] for i in self.hidden_layer_fineturn])
        
            
        return StudentOutput(
            logits=outputs.logits,
            hidden_states=hidden_states,
            span_states=span_states,
            span_weights=span_weights
        )
        

    def save(self, output_dir: str):
        self.model.save_pretrained(output_dir, state_dict=self.model.state_dict())