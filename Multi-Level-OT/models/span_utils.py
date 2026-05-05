import math
import torch
import torch.nn.functional as F


# ============================================================
# Part A: MTA ORIGINALS (same-tokenizer, preserved as-is)
# ============================================================

def compute_token_weights(hidden_state, attention_mask):
    hidden_state = hidden_state.float()  # BF16 std can be 0 → Q/K overflow → NaN
    std = hidden_state.std(dim=-1, keepdim=True).clamp(min=1e-6)
    Q = hidden_state / std
    K = hidden_state / std
    scores = torch.matmul(Q, K.transpose(-1, -2)) / (hidden_state.size(-1) ** 0.5)

    mask = attention_mask.unsqueeze(1).expand(-1, scores.size(-2), -1)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    diag_mask = torch.eye(scores.size(-1), device=scores.device, dtype=torch.bool)
    scores = scores.masked_fill(diag_mask.unsqueeze(0), float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = attn_weights * mask
    attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-10)

    token_weights = attn_weights.mean(dim=1).squeeze(0)  # [L]
    return token_weights.nan_to_num(nan=0.0).detach()


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

        if spans_with_offsets:
            unique_spans, unique_words = filter_overlapping_spans(spans_with_offsets)
            spans.append(unique_spans)
            words.append(unique_words)
        else:
            spans.append([])
            words.append([])

    return spans, words


# ============================================================
# Part B: CROSS-TOKENIZER ADAPTATION (Multi-Level-OT)
# ============================================================

def _prepare_one_side_indices(attention_mask, offsets_mapping, spans_offsets, device):
    """
    Map character-level span offsets to token indices for ONE tokenizer side.

    Returns (All_Indices, Span_IDs, Max_Spans, Batch_ID_for_Spans, batch_indices)
    or None if no valid spans found.

    All_Indices   : flat token positions in [B*SeqLen]
    Span_IDs      : global span index for each entry in All_Indices
    Max_Spans     : total number of unique spans across batch
    Batch_ID_for_Spans : batch index for each unique span
    batch_indices : batch index for each entry in All_Indices
    """
    B_size, SeqLen = attention_mask.shape
    max_spans = max((len(s) for s in spans_offsets), default=0)
    if max_spans == 0:
        return None

    padded_span_starts = torch.zeros(B_size, max_spans, dtype=torch.long, device=device)
    padded_span_ends   = torch.zeros(B_size, max_spans, dtype=torch.long, device=device)
    padded_span_mask   = torch.zeros(B_size, max_spans, dtype=torch.bool,  device=device)

    for i in range(B_size):
        n = len(spans_offsets[i])
        if n > 0:
            s = torch.tensor(spans_offsets[i], device=device, dtype=torch.long)
            padded_span_starts[i, :n] = s[:, 0]
            padded_span_ends[i, :n]   = s[:, 1]
            padded_span_mask[i, :n]   = True

    cur_offsets = offsets_mapping[:, :SeqLen, :].to(device)
    offsets_start = cur_offsets[..., 0].unsqueeze(2)   # (B, SeqLen, 1)
    offsets_end   = cur_offsets[..., 1].unsqueeze(2)

    span_starts = padded_span_starts.unsqueeze(1)       # (B, 1, max_spans)
    span_ends   = padded_span_ends.unsqueeze(1)

    token_in_span = (offsets_start + 1 >= span_starts) & (offsets_end <= span_ends)
    token_in_span = token_in_span & attention_mask.unsqueeze(2).bool()
    token_in_span = token_in_span & padded_span_mask.unsqueeze(1)

    if not token_in_span.any():
        return None

    nz = token_in_span.nonzero(as_tuple=False)
    batch_indices      = nz[:, 0]
    token_indices      = nz[:, 1]
    local_span_indices = nz[:, 2]

    All_Indices = batch_indices * SeqLen + token_indices
    global_span_ids = batch_indices * max_spans + local_span_indices
    _, Span_IDs = torch.unique(global_span_ids, return_inverse=True)
    Max_Spans   = int(Span_IDs.max().item()) + 1

    Batch_ID_for_Spans = torch.empty(Max_Spans, device=device, dtype=torch.long)
    Batch_ID_for_Spans.scatter_(0, Span_IDs, batch_indices)

    return All_Indices, Span_IDs, Max_Spans, Batch_ID_for_Spans, batch_indices


def _gather_layer_weights(layer_weights, All_Indices, batch_indices_raw, B_size, SeqLen):
    """
    Gather and normalize token weights for the tokens indexed by All_Indices.

    layer_weights : (num_layers, B, SeqLen)
    All_Indices   : (T_total,) flat indices in [B*SeqLen]
    batch_indices_raw : (T_total,) batch index per token-in-span entry
    Returns: (num_layers, T_total) normalized per-sample token weights
    """
    num_layers = layer_weights.shape[0]
    flat = layer_weights.view(num_layers, B_size * SeqLen)  # (L, B*SeqLen)
    w = flat[:, All_Indices].float()                         # (L, T_total)

    b_exp = batch_indices_raw.unsqueeze(0).expand(num_layers, -1)  # (L, T_total)
    sums = torch.zeros(num_layers, B_size, device=layer_weights.device, dtype=torch.float)
    sums.scatter_add_(1, b_exp, w)
    sums = sums.clamp(min=1e-5)
    sums_g = torch.gather(sums, 1, b_exp)                  # (L, T_total)
    return w / sums_g


def prepare_span_data_cross_tokenizer(
    t_layer_weights, s_layer_weights,
    t_attention_mask, s_attention_mask,
    t_offsets_mapping, s_offsets_mapping,
    spans_offsets, w_t_entropy=None
):
    """
    Cross-tokenizer span preparation.
    Processes teacher and student sides independently using shared character-level span offsets.

    t_layer_weights : (num_layers, B, T_SeqLen) — teacher token-centrality weights per layer
    s_layer_weights : (num_layers, B, S_SeqLen) — student token-centrality weights per layer
    w_t_entropy     : (B, T_SeqLen) optional entropy weights from teacher logits

    Returns tuple (or scalar 0 tensor on failure):
        T_All_Indices, S_All_Indices,
        T_Span_IDs, S_Span_IDs, Max_Spans,
        T_Batch_IDs,
        T_Token_Weights,    # (num_layers, T_total_t) normalized
        S_Token_Weights,    # (num_layers, T_total_s) normalized
        T_Entropy_Weights   # (T_total_t,) or None
    """
    device = t_attention_mask.device
    T_B, T_SeqLen = t_attention_mask.shape
    S_B, S_SeqLen = s_attention_mask.shape

    t_side = _prepare_one_side_indices(t_attention_mask, t_offsets_mapping, spans_offsets, device)
    s_side = _prepare_one_side_indices(s_attention_mask, s_offsets_mapping, spans_offsets, device)

    if t_side is None or s_side is None:
        return torch.tensor(0.0, device=device)

    T_All_Indices, T_Span_IDs, T_Max_Spans, T_Batch_IDs, T_batch_raw = t_side
    S_All_Indices, S_Span_IDs, S_Max_Spans, S_Batch_IDs, S_batch_raw = s_side

    # Use min to guard against rare cross-tokenizer edge cases
    Max_Spans = min(T_Max_Spans, S_Max_Spans)
    T_Span_IDs = T_Span_IDs.clamp(max=Max_Spans - 1)
    S_Span_IDs = S_Span_IDs.clamp(max=Max_Spans - 1)
    T_Batch_IDs = T_Batch_IDs[:Max_Spans]

    T_Token_Weights = _gather_layer_weights(t_layer_weights, T_All_Indices, T_batch_raw, T_B, T_SeqLen)
    S_Token_Weights = _gather_layer_weights(s_layer_weights, S_All_Indices, S_batch_raw, S_B, S_SeqLen)

    T_Entropy_Weights = None
    if w_t_entropy is not None:
        t_ent_flat = w_t_entropy.float().reshape(-1)         # (B*T_SeqLen,)
        T_ent_unnorm = t_ent_flat[T_All_Indices]             # (T_total_t,)
        T_e_sums = torch.zeros(T_B, device=device, dtype=torch.float)
        T_e_sums.scatter_add_(0, T_batch_raw, T_ent_unnorm)
        T_e_sums = T_e_sums.clamp(min=1e-5)
        T_Entropy_Weights = T_ent_unnorm / T_e_sums[T_batch_raw]  # (T_total_t,)

    return (
        T_All_Indices, S_All_Indices,
        T_Span_IDs, S_Span_IDs, Max_Spans,
        T_Batch_IDs,
        T_Token_Weights,
        S_Token_Weights,
        T_Entropy_Weights,
    )


def compute_hidden_span_loss_cross(
    projector, s_hidden, t_hidden,
    T_All_Indices, S_All_Indices,
    T_Span_IDs, S_Span_IDs, Max_Spans,
    T_Batch_IDs,
    T_Token_Weights_1layer,  # (T_total_t,)
    S_Token_Weights_1layer,  # (T_total_s,)
    T_Entropy_Weights=None   # (T_total_t,) or None
):
    """
    Compute L_DSA + L_Hid/10 for a single layer pair (cross-tokenizer variant).

    Unlike MTA original, student and teacher span representations are aggregated
    independently (separate token indices) then compared at span-mean level.
    This handles cross-tokenizer differences where span token counts differ.
    """
    D_s = s_hidden.size(-1)
    D_t = t_hidden.size(-1)
    device = t_hidden.device

    T_flat = t_hidden.flatten(0, 1)  # (B*T_SeqLen, D_t)
    S_flat = s_hidden.flatten(0, 1)  # (B*S_SeqLen, D_s)

    T_span_all = T_flat[T_All_Indices]  # (T_total_t, D_t)
    S_span_all = S_flat[S_All_Indices]  # (T_total_s, D_s)

    # --- Aggregate teacher spans (weighted mean) ---
    T_w_exp = T_Token_Weights_1layer.unsqueeze(-1)
    T_span_sum = torch.zeros(Max_Spans, D_t, device=device)
    T_w_sum    = torch.zeros(Max_Spans, device=device)
    T_span_sum.scatter_add_(0, T_Span_IDs.unsqueeze(-1).expand(-1, D_t), T_span_all * T_w_exp)
    T_w_sum.scatter_add_(0, T_Span_IDs, T_Token_Weights_1layer)
    T_span_mean = T_span_sum / T_w_sum.clamp(min=1e-5).unsqueeze(-1)  # (Max_Spans, D_t)

    # --- Aggregate student spans (weighted mean) ---
    S_w_exp = S_Token_Weights_1layer.unsqueeze(-1)
    S_span_sum = torch.zeros(Max_Spans, D_s, device=device)
    S_w_sum    = torch.zeros(Max_Spans, device=device)
    S_span_sum.scatter_add_(0, S_Span_IDs.unsqueeze(-1).expand(-1, D_s), S_span_all * S_w_exp)
    S_w_sum.scatter_add_(0, S_Span_IDs, S_Token_Weights_1layer)
    S_span_mean = S_span_sum / S_w_sum.clamp(min=1e-5).unsqueeze(-1)  # (Max_Spans, D_s)

    # --- Span-level entropy weight sum (for weighting L_DSA and L_Hid) ---
    T_ent_span_sum = None
    if T_Entropy_Weights is not None:
        T_ent_span_sum = torch.zeros(Max_Spans, device=device)
        T_ent_span_sum.scatter_add_(0, T_Span_IDs, T_Entropy_Weights)

    # Select which weight to use for pair weighting
    span_w = T_ent_span_sum if T_ent_span_sum is not None else T_w_sum

    # --- L_DSA: pairwise structural alignment (span-mean cosine similarity) ---
    S_norm = F.normalize(S_span_mean, p=2, dim=-1)
    T_norm = F.normalize(T_span_mean, p=2, dim=-1)
    S_sim  = S_norm @ S_norm.T   # (Max_Spans, Max_Spans)
    T_sim  = T_norm @ T_norm.T

    same_batch = T_Batch_IDs.unsqueeze(1) == T_Batch_IDs.unsqueeze(0)
    not_self   = ~torch.eye(Max_Spans, dtype=torch.bool, device=device)
    mask = same_batch & not_self

    pair_w    = span_w.unsqueeze(1) * span_w.unsqueeze(0)
    valid_pw  = torch.masked_select(pair_w, mask)

    dsa_loss = F.mse_loss(
        torch.masked_select(S_sim, mask),
        torch.masked_select(T_sim, mask),
        reduction='none'
    )
    dsa_loss = (dsa_loss * valid_pw).sum() / valid_pw.sum().clamp(min=1e-5)

    # --- L_Hid: projected cosine alignment at span-mean level (cross-tokenizer safe) ---
    # Paper Eq.13 uses per-token; here we use span-mean to handle differing token counts.
    s_proj   = projector(S_span_mean)   # (Max_Spans, D_t)
    cos      = F.cosine_similarity(s_proj, T_span_mean, dim=-1, eps=1e-5)
    hid_loss = 1 - cos                  # (Max_Spans,)
    hid_loss = (hid_loss * span_w).sum() / span_w.sum().clamp(min=1e-5)

    return dsa_loss + hid_loss / 10.0


def get_span_loss_cross(
    projectors, t_attention_mask, s_attention_mask,
    t_hidden_states, s_hidden_states,
    t_offsets_mapping, s_offsets_mapping,
    spans_offsets, t_layer_mapping, s_layer_mapping,
    w_t_entropy=None
):
    """Compute combined span loss for one span type (word or phrase) across all layer pairs."""
    if not t_layer_mapping:
        return torch.tensor(0.0, device=t_attention_mask.device)

    device = t_attention_mask.device
    T_B, T_SeqLen = t_attention_mask.shape
    S_B, S_SeqLen = s_attention_mask.shape

    # Compute token-centrality weights for each layer pair
    t_weights_list = []
    for i in t_layer_mapping:
        t_weights_list.append(compute_token_weights(t_hidden_states[i], t_attention_mask))
    t_layer_weights = torch.stack(t_weights_list)  # (num_layers, B, T_SeqLen)

    s_weights_list = []
    for i in s_layer_mapping:
        s_weights_list.append(compute_token_weights(s_hidden_states[i], s_attention_mask))
    s_layer_weights = torch.stack(s_weights_list)  # (num_layers, B, S_SeqLen)

    result = prepare_span_data_cross_tokenizer(
        t_layer_weights, s_layer_weights,
        t_attention_mask, s_attention_mask,
        t_offsets_mapping, s_offsets_mapping,
        spans_offsets, w_t_entropy
    )
    if isinstance(result, torch.Tensor):  # 0 scalar returned on failure
        return result

    (T_All_Indices, S_All_Indices,
     T_Span_IDs, S_Span_IDs, Max_Spans,
     T_Batch_IDs,
     T_Token_Weights, S_Token_Weights,
     T_Entropy_Weights) = result

    final_loss = torch.tensor(0.0, device=device)
    for i, (s_idx, t_idx, proj) in enumerate(zip(s_layer_mapping, t_layer_mapping, projectors)):
        span_loss = compute_hidden_span_loss_cross(
            proj,
            s_hidden_states[s_idx],
            t_hidden_states[t_idx],
            T_All_Indices, S_All_Indices,
            T_Span_IDs, S_Span_IDs, Max_Spans,
            T_Batch_IDs,
            T_Token_Weights[i],
            S_Token_Weights[i],
            T_Entropy_Weights,
        )
        final_loss = final_loss + span_loss

    return final_loss


def compute_overall_span_loss_cross(
    projectors,
    t_attention_mask, s_attention_mask,
    s_logits, t_logits,
    s_hidden_states, t_hidden_states,
    t_offsets_mapping, s_offsets_mapping,
    spans_offsets, words_offsets,
    distil_config
):
    """
    Entry point for Multi-Level-OT span loss.

    Mirrors MTA's compute_overall_span_loss but supports cross-tokenizer distillation.
    Lower layers use word spans; higher layers use phrase spans (NP/VP).

    Loss = (word_loss + phrase_loss) / num_total_layers
    """
    # Entropy weight from teacher logits (MTA Eq. line 180-184)
    # Computed in chunks along the sequence dimension to avoid OOM on large vocabs.
    w_t_entropy = None
    if distil_config.entropy_weight:
        with torch.no_grad():
            t_log = t_logits.float()          # (B, T, V)
            vocab_size = t_log.size(-1)
            # Use log-softmax for numerical stability; avoid materialising
            # a second full (B, T, V) tensor by computing entropy directly.
            log_probs = torch.log_softmax(t_log, dim=-1)  # (B, T, V)
            # entropy = -sum(p * log_p) = -sum(exp(log_p) * log_p)
            # Process in seq-length chunks to cap peak memory usage.
            chunk = 64
            B, T, V = log_probs.shape
            entropy_parts = []
            for start in range(0, T, chunk):
                lp = log_probs[:, start:start+chunk, :]          # (B, c, V)
                p  = lp.exp()
                entropy_parts.append(-(p * lp).sum(dim=-1))     # (B, c)
            t_entropy = torch.cat(entropy_parts, dim=1)          # (B, T)
            del log_probs
            w_t_entropy = 1 - t_entropy / math.log(vocab_size)  # (B, T), [0,1]

    s_mapping = [int(x) for x in distil_config.student_layer_mapping.split(',') if x.strip()]
    t_mapping = [int(x) for x in distil_config.teacher_layer_mapping.split(',') if x.strip()]
    split_idx = [int(x) for x in distil_config.split_layer_mapping.split(',') if x.strip()]

    if len(split_idx) < 3 or not s_mapping:
        return torch.tensor(0.0, device=t_attention_mask.device)

    # Word spans → lower layers [split_idx[0]:split_idx[1]]
    s_word = s_mapping[split_idx[0]:split_idx[1]]
    t_word = t_mapping[split_idx[0]:split_idx[1]]
    word_proj = projectors[split_idx[0]:split_idx[1]]
    word_loss = get_span_loss_cross(
        word_proj, t_attention_mask, s_attention_mask,
        t_hidden_states, s_hidden_states,
        t_offsets_mapping, s_offsets_mapping,
        words_offsets, t_word, s_word, w_t_entropy
    )

    # Phrase spans → higher layers [split_idx[1]:split_idx[2]]
    s_phrase = s_mapping[split_idx[1]:split_idx[2]]
    t_phrase = t_mapping[split_idx[1]:split_idx[2]]
    phrase_proj = projectors[split_idx[1]:split_idx[2]]
    phrase_loss = get_span_loss_cross(
        phrase_proj, t_attention_mask, s_attention_mask,
        t_hidden_states, s_hidden_states,
        t_offsets_mapping, s_offsets_mapping,
        spans_offsets, t_phrase, s_phrase, w_t_entropy
    )

    total_layers = max(len(s_mapping), 1)
    return (word_loss + phrase_loss) / total_layers
