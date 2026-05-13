"""
Utilities for SpanResidual KD (On et al. ICLR 2026 + MTA span supervision).
Stage 1: ProjectorTA (P_T->A, P_A->T) for bottleneck reconstruction pretrain.
Stage 2: ProjectorSA (P_S->A) + cross_model_attention for cross-tokenizer residual.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectorTA(nn.Module):
    """Bottleneck projectors P_T->A (d_T->d_A) and P_A->T (d_A->d_T).

    Used in Stage 1 pretraining: teacher hidden -> bottleneck -> reconstruct.
    In Stage 2 the P_T->A half is loaded frozen; P_A->T is discarded.
    """
    def __init__(self, d_T: int, d_A: int):
        super().__init__()
        self.P_TA = nn.Linear(d_T, d_A, bias=False)
        self.P_AT = nn.Linear(d_A, d_T, bias=False)

    def forward(self, h: torch.Tensor):
        """Returns (z, h_recon) where z in R^(...,d_A), h_recon in R^(...,d_T)."""
        h = h.to(self.P_TA.weight.dtype)
        z = self.P_TA(h)
        return z, self.P_AT(z)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """Return anchor-space embedding z = P_T->A(h)."""
        h = h.to(self.P_TA.weight.dtype)
        return self.P_TA(h)


class ProjectorSA(nn.Module):
    """Student-to-anchor projector P_S->A (d_S->d_A).

    Trained from scratch in Stage 2.  Maps student hiddens into the same
    anchor space as P_T->A so cross-model attention can be applied.
    """
    def __init__(self, d_S: int, d_A: int):
        super().__init__()
        self.P_SA = nn.Linear(d_S, d_A, bias=False)

    def forward(self, h_S: torch.Tensor) -> torch.Tensor:
        return self.P_SA(h_S.to(self.P_SA.weight.dtype))


def cross_model_attention(
    h_S_A: torch.Tensor,
    h_T_A: torch.Tensor,
    return_attn: bool = False,
) -> torch.Tensor:
    """Cross-model attention (Eq. 9-10, On et al. 2026).

    Reindexes teacher anchor-space hiddens to student token positions so that
    residual correction can be applied position-wise despite n_S != n_T.

    Args:
        h_S_A: (B, n_S, d_A) student hidden in anchor space
        h_T_A: (B, n_T, d_A) teacher hidden in anchor space
        return_attn: if True, return (h_T_aligned, A) instead of just h_T_aligned
    Returns:
        h_T_aligned: (B, n_S, d_A), or tuple (h_T_aligned, A) if return_attn=True
    """
    d_A = h_S_A.size(-1)
    Q = h_S_A / h_S_A.std(dim=-1, keepdim=True).clamp(min=1e-5)  # (B, n_S, d_A)
    K = h_T_A / h_T_A.std(dim=-1, keepdim=True).clamp(min=1e-5)  # (B, n_T, d_A)
    A = torch.matmul(Q, K.transpose(-1, -2)) / (d_A ** 0.5)  # (B, n_S, n_T)
    A = F.softmax(A, dim=-1)
    out = torch.matmul(A, h_T_A)   # (B, n_S, d_A)
    if return_attn:
        return out, A
    return out


def compute_residual_correction(
    projector_ta: ProjectorTA,
    projector_sa: ProjectorSA,
    projector_at: nn.Linear,
    h_T: torch.Tensor,
    h_S: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Compute residual correction term for cross-tokenizer setup.

    h_T and h_S may have different sequence lengths (n_T != n_S).
    Cross-model attention aligns teacher positions to student positions before
    computing the residual correction vector.

    Returns:
        correction: (B, n_S, d_S)  to be subtracted from h_S
    """
    # anchor-space projections
    h_T_A = projector_ta.encode(h_T.float())    # (B, n_T, d_A)
    h_S_A = projector_sa(h_S.float())           # (B, n_S, d_A)
    # align teacher to student positions
    h_T_A_aligned = cross_model_attention(h_S_A, h_T_A)  # (B, n_S, d_A)
    # project back to student hidden space via P_A->T (shape [d_A, d_S] analogue)
    correction = projector_at(h_T_A_aligned)    # (B, n_S, d_S) if projector_at: d_A->d_S
    return beta * correction


def compute_beta_seq(
    h_S: torch.Tensor,
    proj_to_S: torch.Tensor,
    response_mask: torch.Tensor,
    d_S: int,
    d_A: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Sequence-level beta per paper: sqrt(d_S/d_A) * mean(||h_S|| / ||proj_to_S||).

    Detached — beta is a fixed scaling factor, not backpropagated.
    """
    s_norm = h_S.float().norm(dim=-1)               # (B, L)
    p_norm = proj_to_S.float().norm(dim=-1).clamp(min=eps)  # (B, L)
    ratio = s_norm / p_norm                          # (B, L)
    mask_f = response_mask.float()
    denom = mask_f.sum(dim=-1).clamp(min=1.0)       # (B,)
    beta_per_seq = (ratio * mask_f).sum(dim=-1) / denom  # (B,)
    beta = (d_S / d_A) ** 0.5 * beta_per_seq.mean()
    return beta.detach()


def compute_residual_mask(
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Returns bool mask: positions where teacher is wrong AND in response.

    Shape: (B, L)
    """
    pred = teacher_logits.argmax(dim=-1)          # (B, L)
    wrong = (pred != labels) & (labels != -100)
    return wrong & response_mask.bool()


def load_projectors(
    path: str,
    d_T: int,
    d_A: int,
    device: torch.device,
) -> ProjectorTA:
    """Load Stage 1 projector checkpoint."""
    proj = ProjectorTA(d_T, d_A)
    if isinstance(device, int):
        device = torch.device("cuda", device)
    state = torch.load(path, map_location=device, weights_only=False)
    proj.load_state_dict(state)
    proj.to(device)
    proj.eval()
    for p in proj.parameters():
        p.requires_grad_(False)
    return proj
