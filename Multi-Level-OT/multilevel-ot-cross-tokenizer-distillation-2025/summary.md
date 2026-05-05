# Paper Summary: Multi-Level Optimal Transport for Universal Cross-Tokenizer Knowledge Distillation on Language Models

## Metadata
- **Title**: Multi-Level Optimal Transport for Universal Cross-Tokenizer Knowledge Distillation on Language Models
- **Authors**: Xiao Cui, Mo Zhu, Yulei Qin, Liang Xie, Wengang Zhou, Houqiang Li
- **Year**: 2025
- **Venue**: AAAI 2025
- **arXiv ID**: 2412.14528v2 (updated 18 Jan 2025)
- **DOI**: [NOT FOUND IN PAPER]
- **Source**: `paper/2412.14528v2.pdf` (local)
- **Code Available**: https://github.com/2018cx/Multi-Level-OT

---

## Problem Statement

Hầu hết các phương pháp knowledge distillation (KD) yêu cầu teacher và student dùng chung tokenizer (cùng vocabulary). Khi hai model thuộc các "family" khác nhau (LLaMA, Qwen, Mistral, OPT, Pythia…), tokenizer khác nhau → vocabulary size khác nhau → không thể tính divergence (KL, RKL, JS) theo chiều dimension. Bài báo giải quyết bài toán **cross-tokenizer KD**: chuyển knowledge từ teacher sang student khi hai bên có tokenizer hoàn toàn khác nhau, không cần dimensional hay token-by-token correspondence.

---

## Key Contributions

1. **MultiLevelOT**: framework KD cross-tokenizer dùng optimal transport (OT) ở **hai cấp độ** — token-level (sequence-aware) và sequence-level — để align logit distribution giữa teacher và student.
2. **Sequence-aware token-level OT**: thay vì optimize từng token độc lập (như ULD), joint-optimize toàn bộ token trong sequence qua sequence-level ranking + top-k truncation + hai loại cost matrix (absolute difference và logarithmic).
3. **Sequence-level OT via Sinkhorn distance**: dùng token-to-token OT distance để xây cost matrix ở cấp sequence, giải token order misalignment do tokenization khác nhau; dùng Sinkhorn distance thay Wasserstein để giảm chi phí tính toán.
4. Không thêm module phụ, không thay đổi output format — áp dụng được cho mọi kiến trúc chỉ dựa trên logits.

---

## Methodology

### Problem Statement (formal)

Cho sample **x** và ground-truth label **y**:
- Teacher logits: **t** ∈ ℝ^{T×m} (m = vocab size teacher)
- Student logits: **s** ∈ ℝ^{T×n} (n = vocab size student)
- T = số token trong sequence, τ = temperature

Mục tiêu: minimize OT distance giữa distribution của teacher và student output.

### Pipeline (Figure 2)

```
Teacher fT ──┐
              ├──► Sequence-Level Ranking & Truncation
Student fS ──┘         │
                        ▼
              Token-Level Cost Matrix ──► OT Matrix ──► LHAD + LSL
              Sequence-Level Cost Matrix ──► Sinkhorn Normalization ──► LSD
```

### Token-Level: Holistic Absolute Difference Loss (LHAD)

**Sequence-level ranking**: thay vì rank độc lập từng token, cộng tổng logits qua T token rồi sort một lần duy nhất → dùng chung một OT matrix cho tất cả token (Eq. 10–11).

**Top-k truncation**: chỉ giữ k chiều top có logit lớn nhất (Eq. 12), đảm bảo: (1) support size đồng nhất giữa teacher và student, (2) loại nhiễu từ near-zero logits, (3) cho phép dùng logarithmic cost matrix.

**LHAD** (Eq. 14):
$$\mathcal{L}_\text{HAD} = \sum_{t=1}^{T}\sum_{i=1}^{k} |t^k_{\text{SR,Tr},i}(t) - s^k_{\text{SR,Tr},i}(t)|$$

### Token-Level: Sequential Logarithmic Loss (LSL)

Cost matrix dạng logarithmic: C^tok_ij(t) = −t_i(t) log s_j(t). Sau top-k truncation và sequence-level ranking:

**LSL** (Eq. 16):
$$\mathcal{L}_\text{SL} = -\sum_{t=1}^{T}\sum_{i=1}^{k} t^k_i(t)\log s^k_i(t)$$

### Sequence-Level: Sinkhorn Distance Loss (LSD)

Xây cost matrix **C** ∈ ℝ^{T×T}, mỗi entry C^seq_ij = OT distance giữa token i (teacher) và token j (student) tính từ PHAD. Dùng Sinkhorn distance (entropy-regularized OT, Eq. 17–19) với kernel matrix K⁰ = exp(−C/λ), iterate N lần Sinkhorn normalization.

**LSD** (Eq. 20):
$$\mathcal{L}_\text{SD} = \langle P^\lambda, C \rangle = \sum_{i,j} K^N_{i,j} C_{i,j}$$

### Total Loss (Eq. 21)
$$\mathcal{L} = \sum_{t=1}^{T}\mathcal{L}_\text{CE}(y(t), s(t)) + \alpha(\mathcal{L}_\text{HAD} + \beta\mathcal{L}_\text{SL} + \gamma\mathcal{L}_\text{SD})$$

---

## Experimental Setup

### Datasets
| Dataset | Task | Metric |
|---------|------|--------|
| **QED** (Lamm et al. 2021) | Extractive QA | F1 score |
| **FairytaleQA** (Xu et al. 2022) | Generative QA | Rouge-LSum |
| **DIALOGSum** (Chen et al. 2021) | Summarization | Rouge-LSum |

### Teacher Models
| Model | Size | Family |
|-------|------|--------|
| LLaMA2 7B Chat | 7B | LLaMA2 |
| Mistral 7B Instruct | 7B | Mistral |
| Qwen 7B Chat | 7B | Qwen |
| LLaMA3 8B Instruct | 8B | LLaMA3 |

### Student Models
| Model | Size | Family/Architecture |
|-------|------|---------------------|
| OPT-350M | 350M | OPT (decoder-only) |
| Pythia-160M | 160M | Pythia (decoder-only) |
| Pythia-410M | 410M | Pythia (decoder-only) |
| Pythia-1B | 1B | Pythia (decoder-only) |
| Bloomz-560M | 560M | BLOOM (decoder-only) |
| mT0-300M | 300M | mT0 (encoder-decoder) |

Tất cả student được khởi tạo từ pretrained weights.

### Distillation Settings
- **Labeled distillation**: có ground-truth label, teacher + ground-truth cùng supervise.
- **Unlabeled distillation**: không có ground-truth, chỉ dùng teacher-generated text làm pseudo-target.

### Baselines
- **SFT** (Supervised Fine-Tuning): chỉ CE loss với ground-truth
- **SeqKD** (Kim & Rush 2016): SFT trên teacher outputs
- **MinED** (Wan et al. 2024): align logits bằng dynamic programming
- **ULD** (Boizard et al. 2024): token-wise OT với zero-padding
- *(DSKD bị loại vì thêm learnable projector → tham số không bằng nhau)*

### Hyperparameters
| Param | Value | Ý nghĩa |
|-------|-------|---------|
| lr | 1e-6 | Learning rate |
| α | 0.15 | Weight cho distillation loss |
| β | 0.1 | Weight cho LSL |
| γ | 0.1 | Weight cho LSD |
| τ_SL | 1 | Temperature token-level |
| τ_SD | 2 | Temperature sequence-level |
| λ | 0.1 | Entropy regularization (Sinkhorn) |
| N | 20 | Số Sinkhorn iterations |
| k | 50 | Top-k truncation threshold |

Hyperparameters giữ cố định trên tất cả task để chứng minh robustness.

### Compute
GPU cluster của MCC Lab, USTC và Supercomputing Center USTC. Không báo cáo số GPU cụ thể hay training time trong paper.

---

## Results

### Main Findings

**Labeled Distillation (Table 1, teacher = LLaMA2-7B):**

| Student | Method | QED (F1) | FairytaleQA | DIALOGSum |
|---------|--------|-----------|-------------|-----------|
| OPT-350M | SFT | 55.71 | 46.04 | 35.59 |
| OPT-350M | ULD | 56.76 | 45.82 | 36.05 |
| OPT-350M | **Ours** | **58.97** | **46.96** | **37.61** |
| Pythia-410M | SFT | 59.03 | 47.23 | 36.06 |
| Pythia-410M | ULD | 59.71 | 47.81 | 36.07 |
| Pythia-410M | **Ours** | **61.79** | **49.10** | **37.45** |
| Bloomz-560M | ULD | 61.22 | 49.87 | 36.40 |
| Bloomz-560M | **Ours** | **62.58** | **50.94** | **37.68** |

- MultiLevelOT giảm performance gap giữa student và teacher **hơn 71%** trên QED (labeled) so với ULD.

**Unlabeled Distillation (Table 2):** MultiLevelOT vẫn outperform ULD và Raw Text baseline trên tất cả 3 dataset và 3 student model, đặc biệt DIALOGSum tăng mạnh (ULD: 32.03→34.21, Ours: 36.88→37.10).

**Generalizability across teacher models (Table 6, student = OPT-350M):**

| Teacher | ULD | Ours |
|---------|-----|------|
| LLaMA2 | 50.71 | 51.96 |
| Mistral3 | 52.08 | 52.96 |
| Qwen | 52.89 | 53.99 |
| LLaMA3 | 52.81 | **54.38** |

**Generalizability across architectures (Table 5, encoder-decoder mT0-300M):**

| Method | QED | FairytaleQA | DIALOGSum |
|--------|-----|-------------|-----------|
| ULD | 37.25 | 31.52 | 30.04 |
| **Ours** | **41.37** | **34.01** | **33.01** |

### Ablation Studies (Table 3, QED task)

| Components | OPT | Pythia | Bloomz |
|-----------|-----|--------|--------|
| CE only (SFT) | 55.71 | 59.03 | 60.48 |
| CE + AD + TR (≈ ULD) | 56.76 | 59.71 | 61.22 |
| CE + AD + SR + Tr (LHAD) | 58.02 | 60.18 | 61.56 |
| CE + AD + SR + Tr + SL | 58.17 | 61.10 | 61.87 |
| CE + AD + SR + Tr + SD | 58.15 | 61.20 | 61.90 |
| CE + AD + SR + Tr + SL + SD | **58.97** | **61.79** | **62.58** |

Mỗi component đóng góp tích lũy; SD (Sinkhorn Distance Loss) là cần thiết nhất cho bước cuối.

**Sequence-level vs Token-level Sinkhorn (Table 4):**

| | OPT | Pythia | Bloomz |
|-|-----|--------|--------|
| w/o SD | 58.17 | 61.10 | 61.87 |
| w token-level SD | 58.32 | 61.22 | 61.95 |
| w sequence-level SD | **58.97** | **61.79** | **62.58** |

→ Sequence-level SD vượt trội token-level SD.

**Hyperparameter sensitivity:**
- N (Sinkhorn iterations): N=20 tối ưu, N>20 cho diminishing returns (Table 7).
- k (truncation): k=50 tối ưu; k nhỏ → miss sentence structure, k lớn → mode-averaging noise (Table 8).

---

## Author-Stated Limitations

[AUTHORS DID NOT STATE LIMITATIONS]

*(Broader Impact section đề cập tiềm năng mở rộng sang multi-teacher, cross-language, multi-modal — nhưng không liệt kê hạn chế cụ thể.)*

---

## Key Terms

| Term | Definition (as used in this paper) |
|------|-----------------------------------|
| Cross-Tokenizer KD (CTKD) | KD giữa teacher và student dùng tokenizer khác nhau (vocabulary khác nhau) |
| Optimal Transport (OT) | Framework toán học đo khoảng cách giữa hai distribution bằng cách tính chi phí tối thiểu để biến đổi distribution này thành distribution kia |
| Wasserstein distance | OT distance chuẩn, đo geometric structure giữa hai distribution |
| Sinkhorn distance | Xấp xỉ Wasserstein bằng entropy regularization, giải được hiệu quả hơn qua iterative normalization |
| Sequence-level ranking | Sắp xếp chiều logit dựa trên tổng cộng qua toàn bộ T token trong sequence, dùng chung một permutation Q cho tất cả token |
| Top-k truncation | Giữ lại k chiều logit có giá trị lớn nhất, loại bỏ near-zero logits (noise) |
| LHAD | Holistic Absolute Difference Loss — token-level OT loss dùng |t-s| cost matrix sau sequence ranking + truncation |
| LSL | Sequential Logarithmic Loss — token-level OT loss dùng cross-entropy-like cost matrix |
| LSD | Sinkhorn Distance Loss — sequence-level OT loss dùng Sinkhorn distance giữa các token vectors |
| Labeled distillation | Distillation có ground-truth label, cả teacher output và label đều làm supervision |
| Unlabeled distillation | Distillation không có ground-truth, chỉ dùng teacher-generated text làm pseudo-target |

---

## Notable Quotes

> "Unlike strict token-wise distillation methods that may lead to token misalignment, we employ sequence-level and sequence-aware token-level optimal transport to facilitate effective knowledge transfer." — Figure 1 caption

> "MultiLevelOT reduces the performance gap between the student and the teacher by over 71% in the QED task on labeled distillation." — Results and Discussions

> "Although further tuning may enhance performance, we maintain a consistent set of hyper-parameters across all tasks to underscore the robustness of our approach." — Implementation Details

---

## References Worth Following

- **ULD** (Boizard et al. 2024, arXiv:2402.12030) — baseline CTKD đầu tiên dùng token-wise OT với zero-padding; paper này cải tiến trực tiếp từ ULD
- **DSKD** (Zhang et al. 2024b, arXiv:2406.17328) — Dual-Space KD dùng learnable projectors để align hidden states cross-tokenizer
- **SinKD** (Cui et al. 2024a/b, TNNLS + COLING) — tiền thân của MultiLevelOT, dùng Sinkhorn distance cho same-tokenizer KD
- **SeqKD** (Kim & Rush 2016, arXiv:1606.07947) — sequence-level KD kinh điển dùng teacher output làm SFT target
- **Sinkhorn Distances** (Cuturi 2013, NeurIPS) — paper gốc đề xuất Sinkhorn distance như xấp xỉ OT hiệu quả
