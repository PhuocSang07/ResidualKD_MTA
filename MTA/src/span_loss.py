"""MTA span/feature distillation losses.

Ported from mta_dskd_v2/code/criterions/dual_space_kd_v2_with_exact_token_alignment.py
with the only change being that `compute_overall_span_loss` takes explicit layer
mapping / entropy params instead of an `args` namespace.
"""
import math
import torch
import torch.nn.functional as F
import spacy
from spacy.matcher import Matcher


def compute_token_weights(hidden_state, attention_mask):
    std = hidden_state.std(dim=-1, keepdim=True) + 1e-5
    Q = hidden_state / std
    K = hidden_state / std
    scores = torch.matmul(Q, K.transpose(-1, -2)) / (hidden_state.size(-1) ** 0.5)

    mask = attention_mask.unsqueeze(1).expand(-1, scores.size(-2), -1)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    diag_mask = torch.eye(scores.size(-1), device=scores.device, dtype=torch.bool)
    scores = scores.masked_fill(diag_mask.unsqueeze(0), float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)  # [B, L, L]
    attn_weights = attn_weights * mask
    attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-5)

    token_weights = attn_weights.mean(dim=1)  # [B, L]
    return token_weights.detach()


def prepare_span_indices_and_weights(t_layer_weights, s_layer_weights, attention_mask,
                                     offsets_mapping, spans_offsets, w_t_entropy=None):
    device = attention_mask.device
    B_size, SeqLen = attention_mask.shape

    max_spans = max(len(s) for s in spans_offsets)
    if max_spans == 0:
        return torch.tensor(0.0, device=device)

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

    # Align offsets_mapping to attention_mask's seq dim
    M = offsets_mapping.shape[1]
    seq_dim = min(M, SeqLen)
    current_offsets_mapping = offsets_mapping[:, :seq_dim, :].to(device)
    attention_mask_subset = attention_mask[:, :seq_dim]

    offsets_start_expanded = current_offsets_mapping[..., 0].unsqueeze(2)  # (B, seq_dim, 1)
    offsets_end_expanded = current_offsets_mapping[..., 1].unsqueeze(2)    # (B, seq_dim, 1)

    span_starts_expanded = padded_span_starts.unsqueeze(1)  # (B, 1, max_spans)
    span_ends_expanded = padded_span_ends.unsqueeze(1)      # (B, 1, max_spans)

    token_in_span_map = (offsets_start_expanded + 1 >= span_starts_expanded) & \
                        (offsets_end_expanded <= span_ends_expanded)

    attention_mask_expanded = attention_mask_subset.unsqueeze(2).bool()  # (B, seq_dim, 1)
    span_mask_expanded = padded_span_mask.unsqueeze(1)                   # (B, 1, max_spans)

    final_token_to_span_map = token_in_span_map & attention_mask_expanded & span_mask_expanded

    if not final_token_to_span_map.any():
        return torch.tensor(0.0, device=device)

    nonzero_indices = final_token_to_span_map.nonzero(as_tuple=False)

    batch_indices = nonzero_indices[:, 0]
    token_indices = nonzero_indices[:, 1]
    local_span_indices = nonzero_indices[:, 2]

    # Indices flattened over (B, SeqLen) to match hidden states / per-layer weight tensors
    All_Indices = batch_indices * SeqLen + token_indices

    global_span_ids_flat = batch_indices * max_spans + local_span_indices
    _, Span_IDs = torch.unique(global_span_ids_flat, return_inverse=True)
    Max_Spans = Span_IDs.max().item() + 1

    Batch_ID_for_Spans = torch.empty(Max_Spans, device=device, dtype=torch.long)
    Batch_ID_for_Spans.scatter_(0, Span_IDs, batch_indices)

    def gather_layer_weights(layer_weights):
        num_layers = layer_weights.shape[0]
        layer_weights_flat = layer_weights.reshape(num_layers, B_size * SeqLen)
        token_weights_unnorm = layer_weights_flat[:, All_Indices].float()
        batch_indices_expanded = batch_indices.unsqueeze(0).expand(num_layers, -1)
        sample_weight_sums = torch.zeros(num_layers, B_size, device=device, dtype=torch.float)
        sample_weight_sums.scatter_add_(1, batch_indices_expanded, token_weights_unnorm)
        sample_weight_sums = sample_weight_sums.clamp(min=1e-5)
        sample_weight_sums_gathered = torch.gather(sample_weight_sums, 1, batch_indices_expanded)
        return token_weights_unnorm / sample_weight_sums_gathered

    T_Token_Weights_all = gather_layer_weights(t_layer_weights)
    S_Token_Weights_all = gather_layer_weights(s_layer_weights)
    T_Entropy_Weight_all = None
    if w_t_entropy is not None:
        T_Entropy_Weight_all = gather_layer_weights(w_t_entropy.unsqueeze(0)).squeeze(0)

    return (All_Indices, T_Token_Weights_all, S_Token_Weights_all,
            Span_IDs, Max_Spans, Batch_ID_for_Spans, T_Entropy_Weight_all)


def compute_hidden_span_loss(projector, s_hidden_state, t_hidden_state, All_Indices,
                             S_Token_Weights_all, T_Token_Weights_all, Span_IDs,
                             Max_Spans, Batch_ID_for_Spans, T_Entropy_Weight_all=None):
    D_hidden_s = s_hidden_state.size(-1)
    D_hidden_t = t_hidden_state.size(-1)
    device = t_hidden_state.device

    T_Hidden_Flat = t_hidden_state.flatten(0, 1)  # (B*SeqLen, D_t)
    S_Hidden_Flat = s_hidden_state.flatten(0, 1)  # (B*SeqLen, D_s)

    T_span_all = T_Hidden_Flat[All_Indices]
    S_span_all = S_Hidden_Flat[All_Indices]

    T_Token_Weights_expanded = T_Token_Weights_all.unsqueeze(-1)
    S_Token_Weights_expanded = S_Token_Weights_all.unsqueeze(-1)

    T_span_weighted = T_span_all * T_Token_Weights_expanded
    S_span_weighted = S_span_all * S_Token_Weights_expanded

    Span_IDs_expanded_t = Span_IDs.unsqueeze(-1).expand(-1, D_hidden_t)
    Span_IDs_expanded_s = Span_IDs.unsqueeze(-1).expand(-1, D_hidden_s)

    T_span_sum = torch.zeros(Max_Spans, D_hidden_t, device=device, dtype=T_span_weighted.dtype)
    S_span_sum = torch.zeros(Max_Spans, D_hidden_s, device=device, dtype=S_span_weighted.dtype)
    T_Weight_sum_1d = torch.zeros(Max_Spans, device=device, dtype=T_Token_Weights_all.dtype)
    S_Weight_sum_1d = torch.zeros(Max_Spans, device=device, dtype=S_Token_Weights_all.dtype)
    T_Entropy_Weight_sum_1d = torch.zeros(Max_Spans, device=device)

    T_span_sum.scatter_add_(0, Span_IDs_expanded_t, T_span_weighted)
    S_span_sum.scatter_add_(0, Span_IDs_expanded_s, S_span_weighted)

    T_Weight_sum_1d.scatter_add_(0, Span_IDs, T_Token_Weights_all)
    T_Weight_sum = T_Weight_sum_1d.clamp(min=1e-5).unsqueeze(-1)
    S_Weight_sum_1d.scatter_add_(0, Span_IDs, S_Token_Weights_all)
    S_Weight_sum = S_Weight_sum_1d.clamp(min=1e-5).unsqueeze(-1)
    if T_Entropy_Weight_all is not None:
        T_Entropy_Weight_sum_1d.scatter_add_(0, Span_IDs, T_Entropy_Weight_all)

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

    w_sum_1d = T_Entropy_Weight_sum_1d if T_Entropy_Weight_all is not None else T_Weight_sum_1d
    Pair_Weights_Matrix = w_sum_1d.unsqueeze(1) * w_sum_1d.unsqueeze(0)
    Valid_Pair_Weights = torch.masked_select(Pair_Weights_Matrix, Final_Mask)

    span_loss = F.mse_loss(S_intra_batch_similarities_flat, T_intra_batch_similarities_flat, reduction='none')
    span_loss = (span_loss * Valid_Pair_Weights).sum() / Valid_Pair_Weights.sum().clamp(min=1e-5)

    s_hidden_expand = projector(S_span_all)
    token_cos = F.cosine_similarity(s_hidden_expand, T_span_all, dim=-1, eps=1e-5)
    token_loss = 1 - token_cos
    token_weight = T_Entropy_Weight_all if T_Entropy_Weight_all is not None else T_Token_Weights_all
    token_loss = (token_loss * token_weight).sum() / token_weight.sum().clamp(min=1e-5)

    return span_loss + token_loss / 10.0


def get_span_loss(projectors, attention_mask, s_hidden_states, t_hidden_states, offsets_mapping,
                  spans_offsets, teacher_layer_mapping, student_layer_mapping, w_t_entropy=None):
    t_layer_weights = []
    s_layer_weights = []
    for i in teacher_layer_mapping:
        weights = compute_token_weights(t_hidden_states[i], attention_mask)
        t_layer_weights.append(weights)
    for i in student_layer_mapping:
        weights = compute_token_weights(s_hidden_states[i], attention_mask)
        s_layer_weights.append(weights)

    t_layer_weights = torch.stack(t_layer_weights)  # (num_layers, B, SeqLen)
    s_layer_weights = torch.stack(s_layer_weights)

    result = prepare_span_indices_and_weights(
        t_layer_weights, s_layer_weights, attention_mask,
        offsets_mapping, spans_offsets, w_t_entropy
    )
    if not isinstance(result, tuple):
        return result  # scalar zero tensor – no valid spans

    (All_Indices, T_Token_Weights_all, S_Token_Weights_all,
     Span_IDs, Max_Spans, Batch_ID_for_Spans, T_Entropy_Weight_all) = result

    final_loss = 0.0
    for i, (s_idx, t_idx, projector) in enumerate(zip(student_layer_mapping, teacher_layer_mapping, projectors)):
        s_hidden = s_hidden_states[s_idx]
        t_hidden = t_hidden_states[t_idx]
        span_loss = compute_hidden_span_loss(
            projector, s_hidden, t_hidden, All_Indices,
            S_Token_Weights_all[i], T_Token_Weights_all[i],
            Span_IDs, Max_Spans, Batch_ID_for_Spans, T_Entropy_Weight_all
        )
        final_loss = final_loss + span_loss

    return final_loss


def compute_overall_span_loss(projectors, attention_mask, s_logits, t_logits,
                              s_hidden_states, t_hidden_states,
                              offsets_mapping, spans_offsets, words_offsets,
                              student_layer_mapping, teacher_layer_mapping,
                              split_layer_mapping, entropy_weight=False):
    w_t_entropy = None
    if entropy_weight:
        t_probs = torch.softmax(t_logits.float().detach(), dim=-1)
        t_entropy = -(t_probs * torch.log(t_probs + 1e-8)).sum(dim=-1)
        w_t_entropy = 1 - t_entropy / math.log(t_logits.size(-1))

    s_word_mapping = student_layer_mapping[split_layer_mapping[0]:split_layer_mapping[1]]
    t_word_mapping = teacher_layer_mapping[split_layer_mapping[0]:split_layer_mapping[1]]
    word_projectors = projectors[split_layer_mapping[0]:split_layer_mapping[1]]
    word_loss = get_span_loss(word_projectors, attention_mask, s_hidden_states, t_hidden_states,
                              offsets_mapping, words_offsets, t_word_mapping, s_word_mapping, w_t_entropy)

    s_span_mapping = student_layer_mapping[split_layer_mapping[1]:split_layer_mapping[2]]
    t_span_mapping = teacher_layer_mapping[split_layer_mapping[1]:split_layer_mapping[2]]
    span_projectors = projectors[split_layer_mapping[1]:split_layer_mapping[2]]
    span_loss = get_span_loss(span_projectors, attention_mask, s_hidden_states, t_hidden_states,
                              offsets_mapping, spans_offsets, t_span_mapping, s_span_mapping, w_t_entropy)

    overall_loss = (word_loss + span_loss) / max(len(student_layer_mapping), 1)
    return overall_loss


def filter_overlapping_spans(spans):
    sorted_spans = sorted(spans, key=lambda s: (s[0], -s[1]))
    filtered = []
    words = []
    if not sorted_spans:
        return filtered, words

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
    words.append((p[n_token - 1].idx, p[n_token - 1].idx + len(p[n_token - 1])))

    return filtered, words


def get_spans_offsets(texts, nlp, matcher):
    disabled_components = ["ner", "lemmatizer"]
    spans = []
    words = []
    for doc in nlp.pipe(texts, disable=disabled_components, n_process=1):
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
    return spans, words


def build_spacy_matcher():
    """Build the spaCy NLP pipeline and verb-phrase matcher used by MTA."""
    nlp = spacy.load("en_core_web_sm")
    matcher = Matcher(nlp.vocab)
    VERB_PHRASE_PATTERN = [
        {"POS": "AUX", "OP": "*"},
        {"POS": "ADV", "OP": "*"},
        {"POS": "VERB", "OP": "+"},
        {"POS": "ADV", "OP": "*"},
    ]
    matcher.add("VERB_PHRASE", [VERB_PHRASE_PATTERN])
    return nlp, matcher
