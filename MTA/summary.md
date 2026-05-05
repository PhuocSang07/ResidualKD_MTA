# Paper Summary: Explain in Your Own Words: Improving Reasoning via Token-Selective Dual Knowledge Distillation

## Metadata
- **Title**: Explain in Your Own Words: Improving Reasoning via Token-Selective Dual Knowledge Distillation
- **Authors**: Minsang Kim (Korea University, SK Telecom), Seung Jun Baek* (Korea University)
- **Year**: 2026
- **Venue**: ICLR 2026 (The International Conference on Learning Representations)
- **arXiv ID**: [NOT FOUND IN PAPER]
- **DOI**: [NOT FOUND IN PAPER]
- **Source**: `paper/4228_Explain_in_Your_Own_Words.pdf` (local)
- **Code Available**: https://github.com/kmswin1/TSD-KD

---

## Problem Statement

Knowledge Distillation (KD) truyền khả năng lập luận từ mô hình lớn (teacher) sang mô hình nhỏ (student). Các phương pháp KD hiện tại — đặc biệt là **on-policy KD** — yêu cầu student bắt chước phân phối của teacher trên **toàn bộ** output token, gây ra hai vấn đề chính:

1. **Distribution mismatch**: Student có năng lực hạn chế bị áp đặt quá nhiều supervision, dẫn đến lệch phân phối (distribution shift), đặc biệt nghiêm trọng trong các tác vụ lập luận phức tạp (complex reasoning).
2. **Teacher-forcing quá mức**: Student không được phép tự do lập luận theo cách riêng, cản trở khả năng tổng quát hóa.

Mục tiêu: thiết kế một framework KD **student-centric** — tập trung supervision vào các token quan trọng, khuyến khích student tự giải thích suy luận theo ngôn ngữ của chính nó.

---

## Key Contributions

1. **Indirect Distillation by Teacher's Preference**: Student tự sinh các candidate token; teacher chỉ cung cấp *preference ranking* (thứ hạng ưu tiên) thay vì ép student khớp toàn bộ phân phối — đây là dạng feedback gián tiếp và nhẹ nhàng hơn KD truyền thống.
2. **Direct Distillation based on Uncertainty Gap**: Áp dụng KL/JSD distillation có chọn lọc, chỉ tại các token mà student **không chắc chắn** (high entropy) trong khi teacher **tự tin** (low entropy) — gating bằng hàm sigmoid của entropy gap.
3. **Entropy Regularization (L_EM)**: Tối thiểu hóa entropy tại top-10% token không chắc chắn nhất, duy trì độ tự tin của student trong quá trình distillation.
4. **Token-selective mechanism**: Cả indirect và direct distillation đều chỉ áp dụng trên tập token được chọn lọc, không phải toàn bộ chuỗi — nguyên lý cốt lõi của TSD-KD.
5. **State-of-the-art** trên 10 reasoning benchmarks; student **vượt trội teacher** trong 4 tác vụ với biên độ tới 20.3%.

---

## Methodology

### Tổng quan kiến trúc

TSD-KD kết hợp **ba thành phần** hoạt động theo cơ chế token-selective:

```
[Student] ──(self-generate)──► Candidate tokens
                                    │
                              Teacher re-ranks (Preference)
                                    │
                         ◄── Indirect Distillation (Opener)
                         ◄── Direct Distillation (Uncertainty Gap)
                         ◄── Entropy Regularization (Top-10% uncertain)
```

---

### Preliminaries

**On-policy KD**: Student sinh output từ chính nó, teacher cung cấp tín hiệu supervision trên output đó — tránh distribution shift so với off-policy.

**Model distribution**: Với autoregressive LM có input `x`, output `y = (y_1, ..., y_L)`:
```
p(y_t | x, y_{<t}) = softmax(z_t)  tại vị trí token t
```

**GKD objective** (baseline): Tối thiểu hóa JSD(β) giữa teacher p_T và student p_S tại mỗi token:

> **Eq. (2):** D(p_T || p_S^θ)(y|x) := (1/L) Σ_{t=1}^{L} D_{JSD(β)}(p_T(·|x, y_{<t}) || p_S^θ(·|x, y_{<t}))

trong đó JSD(β) cho β → 0 hội tụ về forward KL, β → 1 hội tụ về reverse KL.

---

### 3.1 Indirect Distillation by Teacher's Preference (L_Indirect)

**Ý tưởng**: Thay vì ép student khớp phân phối đầy đủ của teacher, chỉ cần teacher đóng vai *ranking oracle*: student đề xuất top-k candidate token, teacher sắp xếp lại theo thứ tự ưu tiên.

**Plackett-Luce (PL) Model** — Eq. (3):

> P_PL(π_t | x, y_{<t}) = ∏_{j=1}^{k} exp(z_t[π_t(j)]) / Σ_{ℓ=j}^{k} exp(z_t[π_t(ℓ)])

Trong đó `z_t` là logits của student, `π_t` là thứ tự ưu tiên do teacher xác định.

**Loss**:

> **Eq. (5):** L_Indirect = -Σ_t log P_PL(π_t | x, y_{<t})

**Proposition 1**: Với k=2 (BT model), L_Indirect tương đương với negative log-likelihood của preference distribution dựa trên log-probability reward của student, với preference label được xác định bởi teacher reward. Tức là:
- Student's reward: `r_S(x, y) = log p_S(y|x) = Σ_t log p_S(y_t|y_{<t}, x)` — **Eq. (7)**
- Teacher's preference label: `y_{→t}^(1) ≻ y_{→t}^(2)` khi `r_T(x, y_{→t}^(1)) ≥ r_T(x, y_{→t}^(2))` — **Eq. (8)**
- Preference probability: `p(y_{→t}^(1) ≻ y_{→t}^(2)|x) = exp(r_S(x,y_{→t}^(1))) / [exp(r_S(x,y_{→t}^(1))) + exp(r_S(x,y_{→t}^(2)))]` — **Eq. (9)**

**Cơ chế Opener (Token-selective)**:
- Định nghĩa *opener* = phần đầu của response, tính từ đầu đến vị trí `m` sao cho cumulative entropy đạt `c%` tổng entropy:
  - **Eq. (10):** `H_t(p) = -Σ_{v∈V} p(v|x, y_{<t}) log p(v|x, y_{<t})`
  - `m = min integer: Σ_{t=1}^{m} H_t(p_S) / Σ_{t=1}^{L} H_t(p_S) ≥ c%`
- Sử dụng `c = 10%` (mặc định); cũng hỗ trợ **adaptive c** dựa trên uncertainty gap.
- Lý do: High-entropy token tập trung ở đầu response (Fig. 1) — đây là điểm phân nhánh quan trọng của lập luận.

---

### 3.2 Direct Distillation based on Uncertainty Gap (L_Direct)

**Ý tưởng**: Direct distillation (JSD) nhưng **có chọn lọc** — chỉ áp dụng tại các token student không chắc chắn nhưng teacher tự tin.

**Gating function** — **Eq. (11)**:

> L_Direct = (1/L) Σ_{t=1}^{L} **σ_τ(H_t(p_S) - H_t(p_T))** · D_{JSD(β)}(p_T(·|x,y_{<t}) || p_S^θ(·|x,y_{<t}))

trong đó `σ_τ(u) = (1 + exp(-u/τ))^{-1}` là sigmoidal gating, τ > 0 kiểm soát độ sắc nét.

- Khi `H_t(p_S) >> H_t(p_T)`: student không chắc, teacher tự tin → σ lớn → distillation mạnh.
- Khi student tự tin (H thấp): σ nhỏ → ít distillation → student tự do sinh reasoning.

**Phân tích gradient** (Appendix A.2): Token-selective KD có *rescaling factor* C so với conventional KD:
- Early training (student yếu): C > 1 → **reinforced gradient** giúp hội tụ nhanh.
- Late training (student mạnh): C < 1 → hiệu ứng **label smoothing** khuyến khích đa dạng hóa.

---

### 3.3 Entropy Regularization (L_EM)

**Ý tưởng**: Giảm entropy tại các token không chắc chắn nhất (top-10%) để tăng tự tin của student.

**Eq. (12)**:

> L_EM = E_{x~X} [ (1/|I|) Σ_{t∈I} H_t(p_S^θ) ]

trong đó `I ⊆ {1,...,T}` là tập index của top-10% token có entropy cao nhất.

- Chỉ áp dụng trên token khó → tránh overconfidence trên toàn bộ chuỗi.
- Động lực: "Entropy minimization improves reasoning" (Agarwal et al., 2025; Wang et al., 2025).

---

### Final Loss — Eq. (13)

> min_θ E_{x~X} [α · L_Indirect + L_Direct + L_EM]

- `α > 0` kiểm soát trọng số tương đối của indirect distillation (α = 0.1).
- Direct distillation và entropy minimization có trọng số bằng nhau (cùng liên quan đến entropy).

---

## Experimental Setup

- **Datasets (Training)**: UltraInteract prompts (Yuan et al., 2025)
- **Evaluation Benchmarks**:
  - *Mathematical reasoning*: GSM8K, GSM-Plus, MATH, MMLU-Pro-Math
  - *STEM & Scientific*: MMLU-STEM, ScienceQA (SciQ)
  - *Program synthesis*: MBPP
  - *Broad reasoning*: BBH (Big-Bench Hard)
  - *Multi-step soft reasoning*: MuSR
  - *Instruction following*: IFEval
- **Models**:
  - Qwen2.5-14B (teacher) → Qwen2.5-1.5B (student) [main]
  - Gemma2-9B (teacher) → Gemma2-2B (student)
  - Qwen3-8B (teacher) → Qwen3-1.7B (student)
- **Baselines**:
  - *On-policy*: DistilLLM, MiniLLM, GKD (β=0.9), Speculative KD
  - *Off-policy*: Supervised-KD (Hinton et al., 2015), Sequence-Level KD (Kim & Rush, 2016)
- **Metrics**: Accuracy theo chuẩn evaluation protocol của từng benchmark
- **Compute**: 8× A100 GPUs (80GB VRAM)
- **Key hyperparameters**:
  - Batch size: 128
  - Learning rate: 5e-6 (Qwen2.5), 1e-7 (Gemma2)
  - LR scheduler: Cosine; Optimizer: AdamW; Warmup ratio: 0.1
  - Epochs: 3; Max sequence length: 1024
  - k (top-k for L_Indirect): 10; β (JSD): 0.9; α: 0.1
  - Selection ratio cho L_Indirect và L_EM: 0.1

---

## Results

### Main Findings — Qwen2.5 (14B → 1.5B) [Table 1]

| Method | GSM8K | GSM-Plus | MATH | MBPP | IFEval | MMLU-STEM |
|---|---|---|---|---|---|---|
| Teacher (14B) | 80.3 | 59.7 | 21.7 | 78.9 | 85.9 | 70.5 |
| Student (1.5B) baseline | 57.1 | 38.8 | 16.9 | 38.4 | 53.1 | 49.5 |
| GKD (β=0.9) | 57.9 | 39.9 | 18.1 | 41.8 | 52.3 | 47.7 |
| MiniLLM | 57.7 | 39.7 | 17.8 | **42.2*** | 54.7 | 48.2 |
| **TSD-KD** | **60.1** | **40.5** | **26.1*** | 42.1 | **55.2** | **50.0** |

(*) = student vượt trội teacher

- TSD-KD (26.1) vượt runner-up (18.5 — Speculative KD) trên MATH: **+40.3%**
- TSD-KD (26.1) vượt teacher (21.7) trên MATH: **+20.3%** ← student surpasses teacher

### Advanced Benchmarks — Qwen2.5 [Table 2]

| Method | MMLU-Pro-Math | SciQ | BBH | MuSR |
|---|---|---|---|---|
| Teacher (14B) | 77.6 | 86.4 | 60.8 | 60.8 |
| **TSD-KD** | **36.9** | **93.0*** | **40.2** | **39.6** |

- SciQ: TSD-KD (93.0%) vượt teacher (86.4%) — **+7.6%** ← student surpasses teacher

### Generalization — Gemma2 (9B → 2B) [Table 4]

- TSD-KD đạt performance cao nhất trên phần lớn benchmarks.
- Student Gemma2-2B vượt teacher Gemma2-9B trên **IFEval và MBPP**.

### Generalization — Qwen3 (8B → 1.7B) [Table 5]

- TSD-KD đạt MATH = **28.0*** (teacher = 22.4) → student vượt teacher **+25.0%**.

### Ablation Study [Table 3]

| σ_τ(·) | L_Indirect | L_EM | MATH | MMLU-STEM | GSM8K |
|---|---|---|---|---|---|
| ✗ | ✗ | ✗ | 18.1 | 47.7 | 57.9 |
| ✔ | ✗ | ✗ | 18.2 | 48.1 | 58.3 |
| ✔ | ✔ | ✗ | 20.9 | 49.4 | 58.6 |
| ✔ | ✗ | ✔ | 22.3 | 49.3 | 59.0 |
| ✔ | ✔ | ✔ | **26.1** | **50.0** | **60.1** |

- σ_τ(·) gating: +3.6% trên GSM8K
- + L_Indirect: cải thiện MMLU-STEM đáng kể (+15.5% trên baseline)
- + L_EM: MATH từ 18.1 → 22.3
- Full model: cải thiện tổng cộng 44.2% so với baseline GKD

### Analysis of Opener (c%) [Fig. 3]

- Optimal: c = 10%; performance giảm khi c > 10%.
- Khi c = 100% (distillation toàn bộ): performance tệ hơn cả baseline → **xác nhận tầm quan trọng của token selection**.

### On-policy vs Off-policy [Table 6]

- On-policy TSD-KD (avg 45.7) > Off-policy TSD-KD (avg 43.4): **+2.3 điểm**, đặc biệt GSM8K (+4.1) và GSM-Plus (+3.4).

### PEFT Experiment [Table 9]

- TSD-KD vẫn đạt cao nhất (avg 43.5) khi dùng LoRA — **top 5/6 benchmarks**.

---

## Author-Stated Limitations

[AUTHORS DID NOT STATE LIMITATIONS]

---

## Key Terms

| Term | Definition (as used in this paper) |
|---|---|
| **TSD-KD** | Token-Selective Dual Knowledge Distillation — framework student-centric kết hợp indirect + direct distillation trên tập token được chọn lọc |
| **Opener** | Phần đầu của response (consecutive token sequence) nơi cumulative entropy đạt c% tổng entropy; chứa các branching points quan trọng của lập luận |
| **Indirect distillation** | Distillation dạng gián tiếp: teacher không ép student khớp phân phối đầy đủ mà chỉ cung cấp preference ranking trên candidate token của student |
| **Direct distillation** | Distillation trực tiếp bằng JSD, nhưng có gating theo uncertainty gap (H_t(p_S) - H_t(p_T)) |
| **On-policy KD** | KD trong đó student sinh output từ chính nó (distribution riêng), thay vì dùng output tĩnh từ teacher (off-policy) |
| **Uncertainty gap** | Chênh lệch entropy giữa student và teacher tại cùng một token position: H_t(p_S) - H_t(p_T) |
| **Plackett-Luce (PL) model** | Mô hình xác suất cho thứ tự ưu tiên (ranking); được dùng để formulate indirect distillation loss |
| **Entropy regularization (L_EM)** | Mục tiêu tối thiểu hóa entropy tại top-10% token không chắc chắn nhất của student |
| **Mode-seeking (Reverse KL)** | Hành vi của D_{KL}(Q||P): student tập trung vào một mode của teacher, bỏ qua các mode khác |
| **Mode-covering (Forward KL)** | Hành vi của D_{KL}(P||Q): student trải đều xác suất để cover tất cả mode của teacher |
| **Sigmoidal gating σ_τ** | Hàm `σ_τ(u) = (1 + exp(-u/τ))^{-1}` dùng để soft-select token dựa trên entropy gap |
| **GKD** | Generalized Knowledge Distillation — baseline on-policy dùng JSD(β) |
| **Sub-response y_{→t}** | Partial response gồm t token đầu tiên: y_{→t} := (y_1, ..., y_t) |

---

## Notable Quotes

> "A student with limited capacity can be overwhelmed by such extensive supervision causing a distribution mismatch, especially in complex reasoning tasks." — Abstract

> "The key idea is to encourage the student to explain reasoning in its own words to promote reasoning capability." — Section 3

> "Interestingly, performance worsens at c = 100%, which is equivalent to applying distillation to all tokens. This result strongly supports our key hypothesis: a strategic selection of tokens is essential." — Section 4.2

> "Our study demonstrates the effectiveness of student-centered approaches in KD." — Section 6 (Conclusion)

---

## References Worth Following

- **GKD** (Agarwal et al., 2024) — Baseline on-policy KD dùng JSD(β); TSD-KD xây trên nền này
- **MiniLLM** (Gu et al., 2023) — On-policy KD với reverse KL; strong baseline được so sánh trực tiếp
- **Speculative KD** (Xu et al., 2025) — On-policy KD bridging teacher-student gap; ICLR 2025
- **UltraInteract** (Yuan et al., 2025) — Dataset training dùng trong paper; preference tree cho reasoning
- **Wang et al. (2025)** — "Beyond 80/20 rule: High-entropy minority tokens drive effective RL for LLM reasoning" — cơ sở lý thuyết cho việc chọn top-10% entropy tokens
- **Agarwal et al. (2025)** — "The unreasonable effectiveness of entropy minimization in LLM reasoning" — cơ sở cho L_EM component
- **Plackett (1975)** — PL model cho preference ranking; framework toán học của indirect distillation
- **KPOD** (Feng et al., 2024) — Off-policy KD với selective key tokens + curriculum learning; liên quan đến token selection
