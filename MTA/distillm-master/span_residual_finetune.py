import time
import os
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import deepspeed

import random
import json
from tqdm import tqdm
import math
import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GenerationConfig)

from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

from arguments import get_args

from data_utils.lm_datasets import LMTrainDataset
from data_utils.data_utils import LLMDataset
from utils import get_optimizer_params, get_optimizer_params_peft, print_args, initialize
from utils import print_rank, get_rank
from utils import save_rank
from utils import all_gather
from utils import load_parallel, save_parallel
from utils import get_tokenizer, get_model

from distillm import forward_kl, reverse_kl, js_distance, tv_distance
from distillm import skewed_forward_kl, skewed_reverse_kl, csd
from distillm import SampleGenerator, ReplayBuffer

from rouge_metric import compute_metrics

from peft import PeftModel
import spacy
from spacy.matcher import Matcher

from span_residual_utils import (
    ProjectorTA, ProjectorSA,
    cross_model_attention, compute_beta_seq, compute_residual_mask, load_projectors,
)

torch.set_num_threads(1)


def get_teacher_model(args, device):
    config = AutoConfig.from_pretrained(args.teacher_model_path)
    if args.model_parallel:
        raise NotImplementedError
    else:
        config.is_model_parallel = False
        try: model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, config=config, device_map={"": device}, torch_dtype=torch.bfloat16)
        except:
            model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, config=config, device_map={"": device}, torch_dtype=torch.float32)
            model = model.half()
        
        if args.peft is not None and args.teacher_peft_path is not None:
            if args.peft == "lora":
                model = PeftModel.from_pretrained(model, args.teacher_peft_path)
                model = model.merge_and_unload()
            else:
                raise NotImplementedError
        else:
            if dist.get_rank() == 0:
                print(' > number of parameters: {}'.format(
                    sum([p.nelement() for p in model.parameters()])), flush=True)

    model.eval()
    
    return model


def get_optimizer(args, model):
    """Set up the optimizer."""
    while isinstance(model, DDP):
        model = model.module

    # Collect projector param ids BEFORE building main group so we can exclude them.
    # projectors are registered as nn.Module submodules → they appear in model.parameters()
    # and would be double-counted if also added to a separate param group.
    projector_param_ids = set()
    for attr in ("projectors", "projector_SA", "projector_AS"):
        m = getattr(model, attr, None)
        if m is not None:
            for p in m.parameters():
                projector_param_ids.add(id(p))

    if args.peft is not None:
        param_groups = get_optimizer_params_peft(args, model)
    else:
        param_groups = get_optimizer_params(args, model)

    # Strip projector params from every main group to avoid double-counting.
    for group in param_groups:
        group["params"] = [p for p in group["params"] if id(p) not in projector_param_ids]

    # Span projectors (MTA) — explicit group with lower LR
    if model.projectors is not None:
        param_groups.append({
            "params": list(model.projectors.parameters()),
            "lr": 5e-4,
        })
    # Residual projectors P_S->A and P_A->S — learnable in Stage 2
    for attr in ("projector_SA", "projector_AS"):
        proj = getattr(model, attr, None)
        if proj is not None:
            param_groups.append({"params": list(proj.parameters()), "lr": 5e-4})

    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    print_rank(f'Optimizer = {optimizer.__class__.__name__}')
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.total_iters,
            eta_min=args.lr_min)
    elif args.lr_decay_style == "noam":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters,
            num_training_steps=args.total_iters,
            power=0.5)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler


def setup_model_and_optimizer(args, model, ds_config, set_optim=True):
    # get the optimizer and lr_scheduler
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
        
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=None,
        config_params=ds_config
    )
    
    # get the memory usage
    print_rank("Model mem\n", torch.cuda.memory_summary())
    return model, optimizer, lr_scheduler


def prepare_dataset(args, tokenizer, teacher_tokenizer=None):
    data = {}
    rng_sample = random.Random(args.seed)
    if args.do_train:
        data["train"] = LMTrainDataset(args, tokenizer, args.data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["train"]))
        data["dev"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
    elif args.do_eval:
        data["test"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
    else:
        raise ValueError("Do train and do eval must set one")

    # Cross-tokenizer: separate teacher dataset tokenised with Qwen tokenizer
    if args.do_train and args.teacher_data_dir is not None and teacher_tokenizer is not None:
        args_t = copy.copy(args)
        args_t.model_type = args.teacher_model_type if args.teacher_model_type else "qwen"
        data["teacher_train"] = LMTrainDataset(args_t, teacher_tokenizer, args.teacher_data_dir,
                                               "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("teacher train num", len(data["teacher_train"]))
    return data


def get_distil_loss(args, teacher_logits, no_model_batch, logits):
    if args.model_parallel:
        raise NotImplementedError
    else:
        if "sfkl" in args.type:
            distil_loss = skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
        elif "srkl" in args.type:
            distil_loss = skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
        elif "jsd" in args.type:
            distil_loss = js_distance(logits, teacher_logits, no_model_batch)
        elif "tvd" in args.type:
            distil_loss = tv_distance(logits, teacher_logits, no_model_batch)
        elif "fkl" in args.type or args.type == "kd":
            distil_loss = forward_kl(logits, teacher_logits, no_model_batch)
        elif "rkl" in args.type:
            distil_loss = reverse_kl(logits, teacher_logits, no_model_batch)
        elif "csd" in args.type:
            distil_loss = csd(logits, teacher_logits, no_model_batch)
        else:
            raise NotImplementedError
    return distil_loss

def compute_token_weights(hidden_state, attention_mask):
    std = hidden_state.std(dim=-1, keepdim=True) + 1e-5
    Q = hidden_state / std
    K = hidden_state / std
    scores = torch.matmul(Q, K.transpose(-1, -2)) / (hidden_state.size(-1) ** 0.5)

    mask = attention_mask.unsqueeze(1).expand(-1, scores.size(-2), -1)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    diag_mask = torch.eye(scores.size(-1), device=scores.device, dtype=torch.bool)
    scores = scores.masked_fill(diag_mask.unsqueeze(0), float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)  # [1, L, L]
    attn_weights = attn_weights * mask
    attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)

    token_weights = attn_weights.mean(dim=1).squeeze(0)  # [L]
    return token_weights.detach()

def prepare_span_indices_and_weights(t_layer_weights, s_layer_weights, 
                                     attention_mask, offsets_mapping, spans_offsets):
    device = attention_mask.device
    B_size, SeqLen = attention_mask.shape

    max_spans = max(len(s) for s in spans_offsets)
    if max_spans == 0:
        print(f"No spans found in the batch.")
        return torch.tensor(0.0, device=device)

    # (B_size, max_spans)
    padded_span_starts = torch.zeros(B_size, max_spans, dtype=torch.long, device=device)
    padded_span_ends = torch.zeros(B_size, max_spans, dtype=torch.long, device=device)
    padded_span_mask = torch.zeros(B_size, max_spans, dtype=torch.bool, device=device)

    for i in range(B_size):
        num_spans_i = len(spans_offsets[i])
        if num_spans_i > 0:
            spans_i = torch.tensor(spans_offsets[i], device=device, dtype=torch.long)
            padded_span_starts[i, :num_spans_i] = spans_i[:, 0]
            padded_span_ends[i, :num_spans_i] = spans_i[:, 1]
            padded_span_mask[i, :num_spans_i] = True
    
    if offsets_mapping.shape[1] != SeqLen:
        current_offsets_mapping = offsets_mapping[:, :SeqLen, :]
    else:
        current_offsets_mapping = offsets_mapping

    # (B_size, SeqLen, 1)
    offsets_start_expanded = current_offsets_mapping[..., 0].unsqueeze(2).to(device)
    offsets_end_expanded = current_offsets_mapping[..., 1].unsqueeze(2).to(device)
    
    # (B_size, 1, max_spans)
    span_starts_expanded = padded_span_starts.unsqueeze(1)
    span_ends_expanded = padded_span_ends.unsqueeze(1)

    token_in_span_map = (offsets_start_expanded + 1 >= span_starts_expanded) & \
                        (offsets_end_expanded <= span_ends_expanded)

    attention_mask_expanded = attention_mask.unsqueeze(2).bool()
    span_mask_expanded = padded_span_mask.unsqueeze(1) 

    final_token_to_span_map = token_in_span_map & attention_mask_expanded & span_mask_expanded

    if not final_token_to_span_map.any():
        print(f"No valid tokens found for any spans in the batch.")
        return torch.tensor(0.0, device=device)

    nonzero_indices = final_token_to_span_map.nonzero(as_tuple=False)
    
    batch_indices = nonzero_indices[:, 0] # (T_total)
    token_indices = nonzero_indices[:, 1] # (T_total)
    local_span_indices = nonzero_indices[:, 2] # (T_total)

    All_Indices = batch_indices * SeqLen + token_indices

    global_span_ids_flat = batch_indices * max_spans + local_span_indices
    _, Span_IDs = torch.unique(global_span_ids_flat, return_inverse=True) # (T_total)
    Max_Spans = Span_IDs.max().item() + 1 # Tổng số span duy nhất

    Batch_ID_for_Spans = torch.empty(Max_Spans, device=device, dtype=torch.long)
    Batch_ID_for_Spans.scatter_(0, Span_IDs, batch_indices)

    def gather_layer_weights(layer_weights):
        B_size, SeqLen = attention_mask.shape
        num_layers = layer_weights.shape[0]
        layer_weights_flat = layer_weights.view(num_layers, B_size * SeqLen)
        token_weights_unnorm = layer_weights_flat[:, All_Indices].float()
        batch_indices_expanded = batch_indices.unsqueeze(0).expand(num_layers, -1)
        sample_weight_sums = torch.zeros(num_layers, B_size, device=device, dtype=torch.float)
        sample_weight_sums.scatter_add_(1, batch_indices_expanded, token_weights_unnorm)
        sample_weight_sums = sample_weight_sums.clamp(min=1e-5)
        sample_weight_sums_gathered = torch.gather(sample_weight_sums, 1, batch_indices_expanded)
        Token_Weights_all = token_weights_unnorm / sample_weight_sums_gathered

        return Token_Weights_all

    T_Token_Weights_all = gather_layer_weights(t_layer_weights)
    S_Token_Weights_all = gather_layer_weights(s_layer_weights)

    return All_Indices, T_Token_Weights_all, S_Token_Weights_all, Span_IDs, Max_Spans, Batch_ID_for_Spans

def get_span_loss(projectors, attention_mask, s_hidden_states, t_hidden_states, 
                  offsets_mapping, spans_offsets, teacher_layer_mapping, student_layer_mapping):
    
    t_layer_weights = []
    s_layer_weights = []
    for i in teacher_layer_mapping:
        weights = compute_token_weights(t_hidden_states[i], attention_mask)  # (B, SeqLen)
        t_layer_weights.append(weights)
    for i in student_layer_mapping:
        weights = compute_token_weights(s_hidden_states[i], attention_mask)  # (B, SeqLen)
        s_layer_weights.append(weights)

    t_layer_weights = torch.stack(t_layer_weights)  # (num_layers, B, SeqLen)
    s_layer_weights = torch.stack(s_layer_weights)  # (num_layers, B, SeqLen)

    (All_Indices, T_Token_Weights_all, S_Token_Weights_all, 
     Span_IDs, Max_Spans, Batch_ID_for_Spans) =  prepare_span_indices_and_weights(t_layer_weights, s_layer_weights, 
                                                                                  attention_mask, offsets_mapping, spans_offsets)
    final_loss = 0.0
    for i, (s_idx, t_idx, projector) in enumerate(zip(student_layer_mapping, teacher_layer_mapping, projectors)):
        s_hidden = s_hidden_states[s_idx]
        t_hidden = t_hidden_states[t_idx]
        span_loss = compute_hidden_span_loss(projector, s_hidden, t_hidden, All_Indices,
                                             S_Token_Weights_all[i], T_Token_Weights_all[i], 
                                             Span_IDs, Max_Spans, Batch_ID_for_Spans)
        final_loss += span_loss

    return final_loss

def get_token_loss(attention_mask, s_hidden_states, t_hidden_states, 
                   teacher_layer_mapping, student_layer_mapping):
    t_layer_weights = []
    for i in teacher_layer_mapping:
        weights = compute_token_weights(t_hidden_states[i], attention_mask)  # (B, SeqLen)
        t_layer_weights.append(weights)
    N = attention_mask.size(-1)
    final_loss = 0.0
    for i, (s_idx, t_idx) in enumerate(zip(student_layer_mapping, teacher_layer_mapping)):
        pair_weights = t_layer_weights[i].unsqueeze(2) * t_layer_weights[i].unsqueeze(1)
        mask = torch.eye(N, device=pair_weights.device).bool()  # (N, N)
        pair_weights[:, mask] = 0.0
        pair_weights = pair_weights / pair_weights.sum(dim=(1, 2), keepdim=True).clamp(min=1e-5)

        s_tokens = F.normalize(s_hidden_states[s_idx], dim=-1, eps=1e-5)
        t_tokens = F.normalize(t_hidden_states[t_idx], dim=-1, eps=1e-5)
        student_scores = torch.matmul(s_tokens, s_tokens.transpose(-1, -2))
        teacher_scores = torch.matmul(t_tokens, t_tokens.transpose(-1, -2))
        span_loss = F.mse_loss(student_scores, teacher_scores, reduction='none')
        span_loss = (span_loss * pair_weights).sum() / pair_weights.sum()

        final_loss += span_loss

    final_loss = final_loss / len(student_layer_mapping)
    return final_loss

def compute_overall_span_loss(projectors, attention_mask, s_hidden_states, t_hidden_states, 
                              offsets_mapping, prases_offsets, spans_offsets, words_offsets, args):
    
    # s_token_mapping = args.student_layer_mapping[:args.split_layer_mapping[0]]
    # t_token_mapping = args.teacher_layer_mapping[:args.split_layer_mapping[0]]
    # token_loss = get_token_loss(attention_mask, s_hidden_states, 
    #                             t_hidden_states, t_token_mapping, s_token_mapping)
    
    s_word_mapping = args.student_layer_mapping[args.split_layer_mapping[0]:args.split_layer_mapping[1]]
    t_word_mapping = args.teacher_layer_mapping[args.split_layer_mapping[0]:args.split_layer_mapping[1]]
    word_projectors = projectors[args.split_layer_mapping[0]:args.split_layer_mapping[1]]
    word_loss = get_span_loss(word_projectors, attention_mask, s_hidden_states, t_hidden_states, 
                              offsets_mapping, words_offsets, t_word_mapping, s_word_mapping)
    
    s_span_mapping = args.student_layer_mapping[args.split_layer_mapping[1]:args.split_layer_mapping[2]]
    t_span_mapping = args.teacher_layer_mapping[args.split_layer_mapping[1]:args.split_layer_mapping[2]]
    span_projectors = projectors[args.split_layer_mapping[1]:args.split_layer_mapping[2]]
    span_loss = get_span_loss(span_projectors, attention_mask, s_hidden_states, t_hidden_states, 
                              offsets_mapping, spans_offsets, t_span_mapping, s_span_mapping)
    
    # s_prase_mapping = args.student_layer_mapping[args.split_layer_mapping[2]:]
    # t_prase_mapping = args.teacher_layer_mapping[args.split_layer_mapping[2]:]
    # prases_loss = get_span_loss(attention_mask, s_hidden_states, t_hidden_states, 
    #                             offsets_mapping, prases_offsets, t_prase_mapping, s_prase_mapping)

    # overall_loss = (token_loss + word_loss + span_loss + prases_loss) / 4
    overall_loss = (word_loss + span_loss) / len(args.student_layer_mapping)
    # overall_loss = (word_loss + span_loss + prases_loss) / 3
    return overall_loss

def compute_hidden_span_loss(projector, s_hidden_state, t_hidden_state, All_Indices, 
                             S_Token_Weights_all, T_Token_Weights_all, Span_IDs, Max_Spans, Batch_ID_for_Spans):
    D_hidden_s = s_hidden_state.size(-1)
    D_hidden_t = t_hidden_state.size(-1)
    device = t_hidden_state.device
    B_size = s_hidden_state.size(0)

    T_Hidden_Flat = t_hidden_state.flatten(0, 1) # (B*SeqLen, D_hidden_t)
    S_Hidden_Flat = s_hidden_state.flatten(0, 1) # (B*SeqLen, D_hidden_s)

    # 1. Trích xuất và Áp dụng Trọng số
    T_span_all = T_Hidden_Flat[All_Indices] # (T_total, D_hidden_t)
    S_span_all = S_Hidden_Flat[All_Indices] # (T_total, D_hidden_s)
    
    T_Token_Weights_expanded = T_Token_Weights_all.unsqueeze(-1) 
    S_Token_Weights_expanded = S_Token_Weights_all.unsqueeze(-1)
    
    T_span_weighted = T_span_all * T_Token_Weights_expanded # (T_total, D_hidden_t)
    S_span_weighted = S_span_all * S_Token_Weights_expanded # (T_total, D_hidden_s)

    Span_IDs_expanded_t = Span_IDs.unsqueeze(-1).expand(-1, D_hidden_t) 
    Span_IDs_expanded_s = Span_IDs.unsqueeze(-1).expand(-1, D_hidden_s) 

    T_span_sum = torch.zeros(Max_Spans, D_hidden_t, device=device)
    S_span_sum = torch.zeros(Max_Spans, D_hidden_s, device=device)
    T_Weight_sum_1d = torch.zeros(Max_Spans, device=device)
    S_Weight_sum_1d = torch.zeros(Max_Spans, device=device)

    T_span_sum.scatter_add_(0, Span_IDs_expanded_t, T_span_weighted)
    S_span_sum.scatter_add_(0, Span_IDs_expanded_s, S_span_weighted)

    T_Weight_sum_1d.scatter_add_(0, Span_IDs, T_Token_Weights_all) 
    T_Weight_sum = T_Weight_sum_1d.clamp(min=1e-5).unsqueeze(-1) # (Max_Spans, 1)
    S_Weight_sum_1d.scatter_add_(0, Span_IDs, S_Token_Weights_all)
    S_Weight_sum = S_Weight_sum_1d.clamp(min=1e-5).unsqueeze(-1) # (Max_Spans, 1)

    # Tính Trung bình (Mean)
    T_span_hidden_mean = T_span_sum / T_Weight_sum 
    S_span_hidden_mean = S_span_sum / S_Weight_sum

    S_normalized = F.normalize(S_span_hidden_mean, p=2, dim=-1)
    T_normalized = F.normalize(T_span_hidden_mean, p=2, dim=-1)
    S_Full_Sim_Matrix = S_normalized @ S_normalized.T
    T_Full_Sim_Matrix = T_normalized @ T_normalized.T

    Batch_IDs_col = Batch_ID_for_Spans.unsqueeze(1)
    Batch_IDs_row = Batch_ID_for_Spans.unsqueeze(0)
    Same_Batch_Mask = (Batch_IDs_col == Batch_IDs_row)
    Not_Self_Mask = ~torch.eye(Max_Spans, dtype=torch.bool, device=device)
    Final_Mask = Same_Batch_Mask & Not_Self_Mask

    S_intra_batch_similarities_flat = torch.masked_select(S_Full_Sim_Matrix, Final_Mask)
    T_intra_batch_similarities_flat = torch.masked_select(T_Full_Sim_Matrix, Final_Mask)

    Pair_Weights_Matrix = T_Weight_sum_1d.unsqueeze(1) * T_Weight_sum_1d.unsqueeze(0)
    Valid_Pair_Weights = torch.masked_select(Pair_Weights_Matrix, Final_Mask)

    span_loss = F.mse_loss(S_intra_batch_similarities_flat, T_intra_batch_similarities_flat, reduction='none')
    span_loss = (span_loss * Valid_Pair_Weights).sum() / Valid_Pair_Weights.sum().clamp(min=1e-5)

    s_hidden_expand = projector(S_span_all)
    token_cos = F.cosine_similarity(s_hidden_expand, T_span_all, dim=-1, eps=1e-5)
    token_loss = 1 - token_cos
    token_loss = (token_loss * T_Token_Weights_all).sum() / T_Token_Weights_all.sum().clamp(min=1e-5)

    return span_loss + token_loss / 10.0


def filter_overlapping_spans(spans):
    sorted_spans = sorted(spans, key=lambda s: (s[0], -s[1]))
    filtered = []
    words = []
    if not sorted_spans:
        return filtered

    current_span = sorted_spans[0]
    for next_span in sorted_spans[1:]:
        _, current_end, p = current_span
        _, next_end, _ = next_span
        if next_end <= current_end:
            continue
        filtered.append((current_span[0], current_span[1]))

        n_token = len(p)
        words.extend([(p[idx - 1].idx, p[idx].idx) for idx in range(1, n_token)])
        words.append((p[n_token - 1].idx, p[n_token - 1].idx + len(p[n_token - 1])))

        current_span = next_span
    filtered.append((current_span[0], current_span[1]))

    p = current_span[2]
    n_token = len(p)
    words.extend([(p[idx - 1].idx, p[idx].idx) for idx in range(1, n_token)])
    words.append((p[n_token - 1].idx, p[n_token-1].idx + len(p[n_token-1])))
    
    return filtered, words

def get_spans_offsets(texts, nlp, matcher):
    disabled_components = ["ner", "lemmatizer"]

    spans = []
    words = []
    prases = []

    for doc in nlp.pipe(texts, disable=disabled_components, n_process=4):
        spans_with_offsets = []
        
        vps = matcher(doc)
        for _, start, end in vps:
            vp = doc[start:end]
            spans_with_offsets.append((vp.start_char, vp.end_char, vp))
            
        ncs = doc.noun_chunks
        spans_with_offsets.extend([(nc.start_char, nc.end_char, nc) for nc in ncs])

        unique_spans, unique_words = filter_overlapping_spans(spans_with_offsets)
        spans.append(unique_spans)
        words.append(unique_words)

        # doc_prases = []
        # for sent in doc.sents:
        #     doc_prases.append((sent.start_char, sent.root.idx))
        #     doc_prases.append((sent.root.idx, sent.end_char + 1))
        # prases.append(doc_prases)
    
    return prases, spans, words

def soft_label_distill_loss(student_logits, teacher_logits, mask, distill_temperature = 2.0):
    student_probs = F.log_softmax(student_logits / distill_temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / distill_temperature, dim=-1)

    loss = F.kl_div(student_probs, teacher_probs, reduction='none').sum(dim=-1)
    loss = (loss * mask).sum() / mask.sum()

    return loss

def get_fdd_loss(args, t_hiddens, s_hiddens, mask, student, teacher):
    i = 0
    traj_loss, der_loss = 0.0, 0.0
    pre_s_hidden_logs, pre_t_hidden_logs = None, None
    # mask = (no_model_batch["label"] != -100).int()

    for s_idx, t_idx in zip(args.student_layer_mapping, args.teacher_layer_mapping):
        s_hidden = s_hiddens[s_idx]
        t_hidden = t_hiddens[t_idx]
        if args.model_type == 'opt':
            s_decoder_proj = student.module.model.model.decoder.project_out
            if s_decoder_proj is not None:
                s_hidden = s_decoder_proj(s_hidden)

            t_decoder_proj = teacher.model.decoder.project_out
            if t_decoder_proj is not None:
                t_hidden = t_decoder_proj(t_hidden)

        s_hidden_logits = student.module.lm_head(s_hidden)
        t_hidden_logits = teacher.lm_head(t_hidden)
        # traj_loss += forward_kl(s_hidden_logits, t_hidden_logits, no_model_batch)
        traj_loss += soft_label_distill_loss(s_hidden_logits, t_hidden_logits, mask)

        s_hidden_logs = F.log_softmax(s_hidden_logits, dim=-1)
        t_hidden_logs = F.log_softmax(t_hidden_logits, dim=-1)

        if i > 0:
            delta_hidden_student = s_hidden_logs - pre_s_hidden_logs
            delta_hidden_teacher = t_hidden_logs - pre_t_hidden_logs
            cos_sim = F.cosine_similarity(delta_hidden_student, delta_hidden_teacher, dim=-1, eps=1e-5)
            cos_sim_loss = 1 - cos_sim
            cos_sim_loss = (cos_sim_loss * mask).sum() / mask.sum()

            der_loss +=  cos_sim_loss

        pre_s_hidden_logs, pre_t_hidden_logs = s_hidden_logs, t_hidden_logs

        i += 1

    return traj_loss / i +  der_loss / (i - 1)


def _align_teacher_hiddens(teacher_hiddens, n_S):
    """Interpolate all teacher hidden layers to student sequence length for span loss."""
    aligned = []
    for h in teacher_hiddens:
        # h: (B, n_T, d_T) -> (B, n_S, d_T)
        h_aligned = F.interpolate(
            h.float().permute(0, 2, 1),   # (B, d_T, n_T)
            size=n_S, mode='linear', align_corners=False
        ).permute(0, 2, 1).to(h.dtype)   # (B, n_S, d_T)
        aligned.append(h_aligned)
    return tuple(aligned)


def finetune(args, tokenizer, model: deepspeed.DeepSpeedEngine, optimizer: AdamW,
             lr_scheduler, dataset, device, teacher_model=None, projectors=None):
    """SpanResidual Stage-2 training loop (cross-tokenizer).

    Loss: L = (1-lambda_res)*L_SFT + lambda_res*L_res + gamma_span*L_span
    """
    print_rank("Start SpanResidual Fine-tuning")

    nlp = spacy.load("en_core_web_sm")
    matcher = Matcher(nlp.vocab)
    matcher.add("VERB_PHRASE", [[
        {"POS": "AUX", "OP": "*"}, {"POS": "ADV", "OP": "*"},
        {"POS": "VERB", "OP": "+"}, {"POS": "ADV", "OP": "*"},
    ]])

    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    dp_group = None
    loss_func = nn.CrossEntropyLoss()

    sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True,
                                 rank=dp_rank, num_replicas=dp_world_size)
    train_dataloader = DataLoader(
        dataset["train"], sampler=sampler, batch_size=args.batch_size,
        collate_fn=dataset["train"].collate)

    # Cross-tokenizer teacher dataloader (separate Qwen-tokenised batches)
    cross_tokenizer = "teacher_train" in dataset
    if cross_tokenizer:
        t_sampler = DistributedSampler(dataset["teacher_train"], shuffle=True, drop_last=True,
                                       rank=dp_rank, num_replicas=dp_world_size)
        teacher_dataloader = DataLoader(
            dataset["teacher_train"], sampler=t_sampler, batch_size=args.batch_size,
            collate_fn=dataset["teacher_train"].collate)
        print_rank("Cross-tokenizer mode: separate teacher batches enabled")
    else:
        teacher_dataloader = None
        print_rank("Same-tokenizer mode")

    # Residual projector handles — all accessed via model.module (the HF model under DS engine)
    projector_TA = getattr(model.module, "projector_TA", None)  # frozen P_T->A
    projector_SA = getattr(model.module, "projector_SA", None)  # learnable P_S->A
    projector_AS = getattr(model.module, "projector_AS", None)  # learnable P_A->S
    d_A = args.d_bottleneck

    step, global_step = 1, 1
    total_loss, total_res_loss, total_span_loss, total_time = 0.0, 0.0, 0.0, 0.0

    prev_avg_loss = evaluate(args, tokenizer, model, dataset["dev"], "dev", 0, device)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        if cross_tokenizer:
            t_sampler.set_epoch(epoch)
            teacher_iter = iter(teacher_dataloader)

        model.train()
        for model_batch, no_model_batch, gen_data in train_dataloader:
            dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)

            # Fetch teacher batch (cross-tokenizer)
            if cross_tokenizer:
                try:
                    t_mb, t_nmb, t_gd = next(teacher_iter)
                except StopIteration:
                    teacher_iter = iter(teacher_dataloader)
                    t_mb, t_nmb, t_gd = next(teacher_iter)
                dataset["teacher_train"].move_to_device(t_mb, t_nmb, t_gd, device)
                teacher_batch = t_mb
            else:
                teacher_batch = model_batch

            torch.cuda.synchronize()
            st_time = time.time()

            # Student forward
            outputs = model(**model_batch, output_hidden_states=True, use_cache=False)
            s_logits = outputs.logits
            label = no_model_batch["label"]
            lm_loss = loss_func(s_logits.float().view(-1, s_logits.shape[-1]), label.view(-1))

            res_loss_val = torch.tensor(0.0, device=device)
            span_loss_val = torch.tensor(0.0, device=device)
            lambda_eff = 0.0

            if teacher_model is not None:
                with torch.no_grad():
                    teacher_model.eval()
                    teacher_outputs = teacher_model(**teacher_batch,
                                                   output_hidden_states=True, use_cache=False)

                # ── Residual loss ──────────────────────────────────────────────
                h_S = outputs.hidden_states[-1]          # (B, n_S, d_S)
                h_T = teacher_outputs.hidden_states[-1]  # (B, n_T, d_T)
                d_S = h_S.size(-1)

                if projector_TA is not None and projector_SA is not None and projector_AS is not None:
                    with torch.no_grad():
                        h_T_A = projector_TA.encode(h_T)  # (B, n_T, d_A) frozen
                    h_S_A = projector_SA(h_S)              # (B, n_S, d_A)

                    # Section 3.4: cross-model attention only needed when sequence lengths differ.
                    if cross_tokenizer:
                        h_T_A_aligned = cross_model_attention(h_S_A, h_T_A.to(h_S_A.dtype))
                    else:
                        h_T_A_aligned = h_T_A.to(h_S_A.dtype)
                    proj_to_S = projector_AS(h_T_A_aligned)  # (B, n_S, d_S)

                    resp_mask = (label != -100)
                    lm_dtype = next(model.module.lm_head.parameters()).dtype

                    # Eq.4 indicator: 1[argmax P_T(x_i|x_:i-1) ≠ x_i]
                    if not cross_tokenizer:
                        teacher_wrong = compute_residual_mask(
                            teacher_outputs.logits, label, resp_mask)
                    else:
                        # Cross-tokenizer: teacher logits live in teacher vocab and don't
                        # align with student labels. Use the projected teacher hidden state
                        # routed through the student LM head to get a prediction in student
                        # vocab, then compare with student labels.
                        with torch.no_grad():
                            proj_logits = model.module.lm_head(proj_to_S.detach().to(lm_dtype))
                            teacher_pred = proj_logits.argmax(dim=-1)
                        teacher_wrong = (teacher_pred != label) & resp_mask

                    # Eq.5: β = sqrt(d_S/d_A) * mean(||h_S|| / ||proj_to_S||).
                    # Paper formula has no clamp, but with zero-init P_A->S the ratio is
                    # ill-defined at step 1 (||proj_to_S|| ≈ 0 ⇒ β → ∞).  A loose static
                    # clamp [0.05, 10.0] keeps β in the empirically observed paper range
                    # without distorting normal training dynamics.
                    beta = compute_beta_seq(h_S.detach(), proj_to_S.detach(), resp_mask, d_S, d_A)
                    beta = beta.clamp(0.05, 10.0)

                    h_S_res = h_S - beta * proj_to_S * teacher_wrong.unsqueeze(-1).float()
                    res_logits = model.module.lm_head(h_S_res.to(lm_dtype))
                    res_loss_val = loss_func(res_logits.float().view(-1, res_logits.shape[-1]), label.view(-1))

                # ── Span loss (MTA) ───────────────────────────────────────────
                if args.gamma_span > 0.0 and projectors is not None:
                    input_texts = tokenizer.batch_decode(model_batch["input_ids"], skip_special_tokens=False)
                    offsets_mapping = tokenizer(
                        input_texts, return_offsets_mapping=True, padding="max_length",
                        max_length=model_batch["attention_mask"].shape[1],
                        truncation=True, add_special_tokens=False, return_tensors="pt"
                    )["offset_mapping"].to(device)
                    prases_offsets, spans_offsets, words_offsets = get_spans_offsets(input_texts, nlp, matcher)

                    # Align teacher hiddens to student seq length for span loss
                    n_S = outputs.hidden_states[0].shape[1]
                    n_T = teacher_outputs.hidden_states[0].shape[1]
                    if cross_tokenizer and n_T != n_S:
                        t_hiddens_aligned = _align_teacher_hiddens(teacher_outputs.hidden_states, n_S)
                    else:
                        t_hiddens_aligned = teacher_outputs.hidden_states

                    span_loss_val = compute_overall_span_loss(
                        projectors, model_batch["attention_mask"],
                        outputs.hidden_states, t_hiddens_aligned,
                        offsets_mapping, prases_offsets, spans_offsets, words_offsets, args)

                # ── Compose total loss ────────────────────────────────────────
                # Warm up lambda_res: ramp 0 → target over first N steps so that
                # randomly-initialized projectors (SA, AS) don't corrupt the model.
                warmup = args.lambda_res_warmup_steps
                if warmup > 0 and global_step <= warmup:
                    lambda_eff = args.lambda_res * (global_step / warmup)
                else:
                    lambda_eff = args.lambda_res
                loss = ((1 - lambda_eff) * lm_loss
                        + lambda_eff * res_loss_val
                        + args.gamma_span * span_loss_val)
            else:
                loss = lm_loss

            model.backward(loss)
            model.step()

            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            global_loss = loss.item() / dp_world_size
            dist.all_reduce(res_loss_val, dist.ReduceOp.SUM, group=dp_group)
            dist.all_reduce(span_loss_val, dist.ReduceOp.SUM, group=dp_group)

            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time
            total_loss += global_loss
            total_res_loss += res_loss_val.item() / dp_world_size
            total_span_loss += span_loss_val.item() / dp_world_size
            total_time += elapsed_time

            # Logging
            def get_log(log_loss, log_res, log_span, log_time):
                lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else args.lr
                scale = optimizer.cur_scale if hasattr(optimizer, "cur_scale") else 0
                return ("train | epoch {:3d} | g_step: {:6d}/{:6d} | "
                        "loss: {:.4f} | res: {:.4f} | span: {:.4f} | λ: {:.3f} | "
                        "lr: {:.4e} | scale: {:.1f} | t: {:.3f}").format(
                    epoch, global_step, args.total_iters,
                    log_loss, log_res, log_span, lambda_eff, lr, scale, log_time)

            # Per-step detailed log for first 100 steps
            if global_step <= 100 or global_step % args.log_interval == 0:
                if step % args.gradient_accumulation_steps == 0:
                    log_str = get_log(global_loss, res_loss_val.item() / dp_world_size,
                                      span_loss_val.item() / dp_world_size, elapsed_time)
                    print_rank(log_str)
                    save_rank(log_str, os.path.join(args.save, "log.txt"))
                    total_loss = total_res_loss = total_span_loss = total_time = 0.0

            elif global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                n = args.log_interval * args.gradient_accumulation_steps
                log_str = get_log(total_loss / n, total_res_loss / n,
                                  total_span_loss / n, total_time / args.log_interval)
                print_rank("*" * 80)
                print_rank(log_str)
                print_rank("*" * 80)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                total_loss = total_res_loss = total_span_loss = total_time = 0.0

            # Checkpoint
            if (args.save and args.save_interval
                    and global_step % args.save_interval == 0
                    and step % args.gradient_accumulation_steps == 0):
                save_dir_path = os.path.join(args.save, str(global_step))
                if dist.get_rank() == 0:
                    os.makedirs(save_dir_path, exist_ok=True)
                    tokenizer.save_pretrained(save_dir_path)
                    model.module.save_pretrained(save_dir_path, safe_serialization=False)
                dist.barrier()

            # Eval
            if (args.eval_interval
                    and global_step % args.eval_interval == 0
                    and step % args.gradient_accumulation_steps == 0):
                evaluate(args, tokenizer, model, dataset["dev"], "dev", epoch, device)
                model.train()

            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1
            if global_step > args.total_iters:
                break

    return model


def evaluate(args, tokenizer, model, dataset: LMTrainDataset, split, epoch, device, adaptive_threshold=None):
    
    collate_fn = dataset.collate

    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss()

    print_rank("dp size", dp_world_size)

    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        min_length=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False
    )

    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    model.eval()
    all_loss = 0.0
    step = 0

    # --eval-gen: generate on ALL dev samples (standalone eval script).
    # Otherwise: always generate on a small subset so ROUGE-L is shown during training
    # without the full-generation overhead that slows down later epochs.
    eval_gen_num = getattr(args, "eval_gen_num", 200)  # default 200 samples for training eval
    do_full_gen = getattr(args, "eval_gen", False)      # True only when --eval-gen passed

    all_response_ids = []
    gen_sample_count = 0

    with torch.no_grad():
        for it, (model_batch, no_model_batch, gen_data) in enumerate(tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0))):
            print_rank(f"{it}/{len(dataloader)}")
            dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
            logits = model(**model_batch).logits
            if args.model_parallel:
                raise NotImplementedError
            else:
                loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))

            # Generate when: full eval requested OR still within subset quota.
            # Skip batches where the prompt already fills max_length (no room for response).
            prompt_len = gen_data["input_ids"].size(1)
            # Fixed response length: max_length - max_prompt_length (e.g. 512 - 256 = 256).
            # All response tensors must share this width for torch.cat to work.
            fixed_resp_len = args.max_length - getattr(args, "max_prompt_length", prompt_len)
            fixed_resp_len = max(fixed_resp_len, 1)
            should_gen = (do_full_gen or (gen_sample_count < eval_gen_num)) and (fixed_resp_len > 0)
            if should_gen:
                max_new_tokens = max(1, fixed_resp_len)
                gen_out = model.generate(
                    **gen_data,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens)

                full_ids = gen_out.sequences              # (B, prompt_len + generated)
                response_ids = full_ids[:, prompt_len:]   # (B, generated_len)

                # Pad / clip to fixed_resp_len so all batches share the same width
                pad_len = fixed_resp_len - response_ids.size(1)
                if pad_len > 0:
                    response_ids = F.pad(response_ids, (0, pad_len), value=tokenizer.pad_token_id)
                elif pad_len < 0:
                    response_ids = response_ids[:, :fixed_resp_len]

                all_response_ids.append(response_ids)
                gen_sample_count += response_ids.size(0)

            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            loss = loss / dp_world_size
            all_loss += loss.item()
            step += 1

    if all_response_ids:
        all_response_ids = torch.cat(all_response_ids, dim=0)
        all_response_ids = all_gather(all_response_ids, dim=1, world_size=dp_world_size, group=dp_group, op="stack")
        all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))
        responses = tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)
    else:
        responses = []

    if get_rank() == 0:
        if responses:
            references = dataset.answers
            responses = responses[:len(references)]
            res = compute_metrics(responses, references)

            eval_dir = os.path.join(args.save, "eval", str(epoch))
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "answers.jsonl"), "w") as f:
                for resp in responses:
                    f.write(json.dumps({"text": resp}) + "\n")
        else:
            res = {}
    
        avg_loss = all_loss / step
        
        if "adaptive" in args.type:
            log_str = f"{split} | avg_loss: {avg_loss} | {res} | threshold: {adaptive_threshold}"
        else:
            log_str = f"{split} | avg_loss: {avg_loss} | {res}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))
        
    return all_loss / step


def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0
    
    args.fp32 = not ds_config["fp16"]["enabled"]   
    args.bf16 = "bf16" in ds_config and ds_config["bf16"]["enabled"]
    args.deepspeed_config = None
    
    # Student tokenizer (GPT2) — used for dataset + eval generation
    tokenizer = get_tokenizer(args)

    # Teacher tokenizer (Qwen) — separate for cross-tokenizer dual dataloader
    if args.teacher_data_dir is not None:
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path)
        if teacher_tokenizer.pad_token_id is None:
            teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id
        assert tokenizer.vocab_size != teacher_tokenizer.vocab_size, (
            "Teacher and student tokenizers have same vocab — use same-tokenizer script instead.")
    else:
        teacher_tokenizer = None

    dataset = prepare_dataset(args, tokenizer, teacher_tokenizer)
    
    dp_world_size = dist.get_world_size()
    
    if args.do_train:
        args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size * args.gradient_accumulation_steps))
        print_rank("Train iters per epoch", args.train_iters_per_epoch)
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.epochs
        if args.epochs is None:
            args.epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)
        print_rank("total_iters", args.total_iters)
        
        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch
    
    # get the model
    model = get_model(args, device)
       
    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type
    
    if args.teacher_model_path is not None:
        teacher_model = get_teacher_model(args, device)
        # Cross-tokenizer: do NOT resize teacher embeddings — vocabularies must stay separate
    else:
        teacher_model = None

    if teacher_model is not None:
        # Hidden sizes
        d_T = teacher_model.config.hidden_size
        d_S = model.config.n_embd if args.model_type == "gpt2" else model.config.hidden_size
        d_A = args.d_bottleneck

        # Span projectors (student_dim -> teacher_dim), one per layer pair
        projector_list = nn.ModuleList([
            nn.Linear(d_S, d_T).to(device)
            for _ in range(len(args.teacher_layer_mapping))
        ])
        # Residual projectors P_S->A and P_A->S (learnable).
        # Zero-init P_A->S so proj_to_S = 0 at step 1 → h_S_res = h_S → res_loss ≈ lm_loss.
        # Without this, random init produces huge corrections that explode CE in the first
        # iterations (proj_to_S is noise, ||proj_to_S|| tiny → β blows up via Eq.5).
        projector_SA = ProjectorSA(d_S, d_A).to(device)
        projector_AS = nn.Linear(d_A, d_S, bias=False).to(device)
        nn.init.zeros_(projector_AS.weight)
    else:
        projector_list = None
        projector_SA = None
        projector_AS = None
        d_S = d_T = d_A = None

    model.projectors = projector_list
    model.projector_SA = projector_SA
    model.projector_AS = projector_AS

    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, model, ds_config, set_optim=args.do_train)

    # Attach frozen P_T->A AFTER deepspeed.initialize (so DS doesn't track its params)
    if args.projector_load_path is not None and teacher_model is not None:
        proj_TA = load_projectors(args.projector_load_path, d_T, d_A, device)
        model.module.projector_TA = proj_TA
        print_rank(f"Loaded frozen P_T->A from {args.projector_load_path}")
    else:
        model.module.projector_TA = None

    if args.do_train:
        model = finetune(args, tokenizer, model, optimizer, lr_scheduler,
                         dataset, device, teacher_model=teacher_model, projectors=projector_list)

    if args.do_eval:
        evaluate(args, tokenizer, model, dataset["test"], "test", 0, device)
        
    
if __name__ == "__main__":
    main()