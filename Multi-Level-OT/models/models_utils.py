import torch
import dataclasses
import torch.nn as nn
import torch.optim as optim

from policies import AnyPrecisionAdamW
from policies import apply_fsdp_checkpointing
from models.fsdp import fsdp_auto_wrap_policy
from configs import fsdp_config as FSDP_CONFIG
from models.distillation_model import DistillationModel
try:
    from optimum.bettertransformer import BetterTransformer
except ImportError:
    BetterTransformer = None
from transformers import AutoModelForCausalLM, MT5ForConditionalGeneration, AutoTokenizer
from configs.configs_utils import generate_peft_config, update_config
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)

from models.tools import (
    freeze_transformer_layers,
    print_model_size,
    get_policies
)

def load_tokenizer(name, encoder_decoder):
    import transformers.tokenization_utils_base as _tub

    # transformers >= 5.x: SpecialTokensMixin was merged into PreTrainedTokenizerBase
    _token_base_cls = getattr(_tub, "SpecialTokensMixin", None) or _tub.PreTrainedTokenizerBase
    _orig_set = _token_base_cls._set_model_specific_special_tokens

    def _patched_set(self, special_tokens):
        # Some older model configs store extra_special_tokens as a list;
        # transformers expects a dict — convert gracefully.
        if isinstance(special_tokens, list):
            special_tokens = {}
        return _orig_set(self, special_tokens)

    _token_base_cls._set_model_specific_special_tokens = _patched_set
    try:
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    finally:
        _token_base_cls._set_model_specific_special_tokens = _orig_set

    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"
    if not encoder_decoder:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def load_model(train_config, rank, device=None):
    use_cache = False if train_config.enable_fsdp else True
    def load():
        if train_config.quantization:
            return AutoModelForCausalLM.from_pretrained(
                train_config.model_name,
                use_cache=use_cache,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True,
            )
        # MTA-style: load directly onto target device via device_map
        device_map = device if device is not None else None
        if getattr(train_config, 'pure_bf16', False):
            dtype = torch.bfloat16
        elif getattr(train_config, 'use_fp16', False):
            dtype = torch.float16
        else:
            dtype = torch.float32
        if "mt0" in train_config.model_name:
            return MT5ForConditionalGeneration.from_pretrained(
                train_config.model_name,
                use_cache=use_cache,
                device_map=device_map,
            )
        elif "Qwen" in train_config.model_name:
            return AutoModelForCausalLM.from_pretrained(
                train_config.model_name,
                use_cache=use_cache,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map=device_map,
            )
        else:
            return AutoModelForCausalLM.from_pretrained(
                train_config.model_name,
                use_cache=use_cache,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map=device_map,
            )

    if not train_config.enable_fsdp:
        model = load()
        
    elif train_config.enable_fsdp:
        if train_config.low_cpu_fsdp:
            if rank == 0:
                model = load()
            else:
                model_config = AutoModelForCausalLM.from_pretrained(
                    train_config.model_name, torch_dtype=torch.float32, trust_remote_code=True)
                model_config.use_cache = use_cache
                with torch.device("meta"):
                    model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
        else:
            model = load()

        if train_config.use_fast_kernels:
            """
            For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
            using of Flash Attention or Xformer memory-efficient kernels
            based on the hardware being used. This would speed up fine-tuning.
            """
            if BetterTransformer is not None:
                model = BetterTransformer.transform(model)
            else:
                print("Warning: optimum.bettertransformer is not available; skipping BetterTransformer optimization.")
            
    print_model_size(model, train_config, rank)
    return model

def _auto_detect_lora_targets(model):
    leaf_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    for candidates in [
        ["q_proj", "v_proj"],   # LLaMA, Mistral, Qwen, OPT
        ["c_attn"],             # GPT-2
        ["query_key_value"],    # Falcon, GPT-NeoX
        ["query", "value"],     # BERT-style
    ]:
        if all(c in leaf_names for c in candidates):
            return candidates
    return ["q_proj", "v_proj"]

def set_model(model, train_config, fsdp_config, rank, kwargs, device=None):
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        # Auto-fix target_modules when defaults don't match this architecture
        model_leaf_names = {name.split(".")[-1] for name, _ in model.named_modules()}
        if not any(m in model_leaf_names for m in peft_config.target_modules):
            peft_config.target_modules = set(_auto_detect_lora_targets(model))
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif train_config.freeze_layers:
        freeze_transformer_layers(train_config.num_freeze_layers)

    if train_config.enable_fsdp:
        if fsdp_config.pure_bf16: model.to(torch.bfloat16)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, [LlamaDecoderLayer, GPTNeoXLayer, MistralDecoderLayer, FalconDecoderLayer])

        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )

        if fsdp_config.fsdp_activation_checkpointing: apply_fsdp_checkpointing(model)
        return model
    else:
        if train_config.quantization: return model
        elif device is not None: return model  # already on device via device_map in load()
        else:
            model = model.to(f"cuda:{rank}")
            # Gradient checkpointing: recomputes activations during backward instead of
            # storing them → trades ~30% speed for significant activation memory savings.
            if getattr(train_config, 'gradient_checkpointing', False):
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
                    print(f"[set_model] gradient_checkpointing enabled for {train_config.model_name}")
            return model

def get_model(train_config, fsdp_config, rank, kwargs, device=None):
    model = load_model(train_config, rank, device=device)
    model = set_model(model, train_config, fsdp_config, rank, kwargs, device=device)
    tokenizer = load_tokenizer(train_config.model_name, train_config.encoder_decoder)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model

def get_distillation_models(train_config, distil_config, fsdp_config, rank, kwargs):
    student_device = getattr(distil_config, 'student_device', None)
    teacher_device = getattr(distil_config, 'teacher_device', None)

    student_tokenizer, student_model = get_model(train_config, fsdp_config, rank, kwargs, device=student_device)

    teacher_fsdp_config = FSDP_CONFIG()
    update_config((teacher_fsdp_config), **dataclasses.asdict(distil_config))
    teacher_tokenizer, teacher_model = get_model(distil_config, distil_config, rank, kwargs, device=teacher_device)

    use_span_loss = getattr(distil_config, 'span_loss_weight', 0.0) > 0
    model = DistillationModel(
        student_model, teacher_model, teacher_tokenizer, student_tokenizer,
        use_span_loss=use_span_loss,
        student_device=student_device,
        teacher_device=teacher_device,
    )

    # Create projectors on the model (part of model.parameters() → included in optimizer)
    model.projectors = None
    if use_span_loss:
        layer_str = getattr(distil_config, 'student_layer_mapping', '')
        n_layers = len([x for x in layer_str.split(',') if x.strip()])
        if n_layers > 0:
            s_h = getattr(distil_config, 'student_hidden_size', 0) or student_model.config.hidden_size
            t_h = getattr(distil_config, 'teacher_hidden_size', 0) or teacher_model.config.hidden_size
            distil_config.student_hidden_size = s_h
            distil_config.teacher_hidden_size = t_h
            device = next(student_model.parameters()).device
            model.projectors = nn.ModuleList([
                nn.Linear(s_h, t_h).to(device) for _ in range(n_layers)
            ])

    return student_tokenizer, teacher_tokenizer, model

def get_optimizer(model, train_config, fsdp_config):
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        return AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        return optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )