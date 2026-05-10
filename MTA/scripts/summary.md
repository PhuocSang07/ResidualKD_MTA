# Paper Summary: MTA: Multi-Granular Trajectory Alignment for Large Language Model Distillation

## Metadata
- **Title**: MTA: Multi-Granular Trajectory Alignment for Large Language Model Distillation
- **Authors**: Anonymous ACL submission (tên tác giả không được tiết lộ)
- **Year**: [NOT FOUND IN PAPER]
- **Venue**: ACL (Anonymous submission)
- **arXiv ID**: [NOT FOUND IN PAPER]
- **DOI**: [NOT FOUND IN PAPER]
- **Source**: Local — `paper/2077_MTA_Multi_Granular_Trajec.pdf`
- **Code Available**: Not provided

---

## Problem Statement

Hầu hết các phương pháp Knowledge Distillation (KD) hiện tại căn chỉnh biểu diễn của teacher và student tại các lớp cố định hoặc ở mức token, bỏ qua cách biểu diễn **tiến hóa xuyên suốt chiều sâu mạng**. Do đó, student chỉ được hướng dẫn yếu về cấu trúc quan hệ nội tại của teacher, hạn chế khả năng truyền tri thức. Ngoài ra, các phương pháp hiện hành áp dụng cùng một mục tiêu căn chỉnh đồng nhất cho tất cả các lớp, không phản ánh tính phân cấp trong ngôn ngữ tự nhiên (các lớp thấp xử lý lexical, các lớp cao xử lý ngữ nghĩa tổng hợp).

---

## Key Contributions

1. **Nhận diện hạn chế của căn chỉnh trung gian đồng nhất**: Đề xuất **MTA** (Multi-Granular Trajectory Alignment), framework mới căn chỉnh quỹ đạo biểu diễn của LLM theo cấu trúc phân cấp vốn có của chúng.

2. **Layer-adaptive distillation objective**: Giới thiệu mục tiêu căn chỉnh thích ứng theo lớp — các lớp thấp dùng *word-level spans* để giữ thông tin từ vựng, các lớp cao dùng *phrase-level spans* (Noun Phrases, Verb Phrases) để nắm bắt ngữ nghĩa tổng hợp.

3. **MTA là module tổng quát có thể tích hợp**: Chứng minh MTA hoạt động như một plug-in modular, có thể tích hợp vào các framework distillation SOTA (FDD, DistiLLM, DistiLLM-2) với cùng tokenizer, và liên tục cải thiện hiệu suất trên nhiều kiến trúc.

---

## Methodology

### Tổng quan

MTA đề xuất hai mục tiêu bổ sung nhau:

1. **Dynamic Structural Alignment Loss (L_DSA)** — căn chỉnh hình học quan hệ giữa các span ngữ nghĩa trong cùng một lớp.
2. **Hidden Representation Alignment (L_Hid)** — căn chỉnh trực tiếp các hidden state đã chọn giữa teacher và student.

**Hàm mục tiêu tổng thể** (Eq. 14):
$$\mathcal{L}_{Total} = \mathcal{L}_{Base} + λ_{DSA} * \mathcal{L}_{DSA} + λ_{Hid} * \mathcal{L}_{Hid}$$

---

### 3.1 Hierarchical Representational Trajectory

Dựa trên các nghiên cứu về khả năng diễn giải Transformer:
- **Lớp thấp**: xử lý thông tin bề mặt/từ vựng (lexical memory).
- **Lớp cao**: thực hiện suy luận trừu tượng và suy diễn tổng hợp.

MTA coi chuỗi biểu diễn theo chiều sâu là một **hierarchical representational trajectory** — không gian biểu diễn thay đổi ngữ nghĩa granularity theo chiều sâu.

---

### 3.2 Layer-Adaptive Multi-Granular Spans

**Token Weight** (Eq. 5–7): Vì causal attention trong LLM có bias về phía token trước, tác giả dùng cơ chế self-attention không có self-loop để tính trọng số token theo tính trung tâm (attention centrality) theo cả hai chiều:
$$Ĥ_{t,l} = \frac{H_{t,l}} {σ(H_{t,l})}$$
$$
S_{s→t,l} = \frac{(Ĥ_{s,l} Ĥ^T_{t,l})} {\sqrt{d}} + M_{s,t} 
$$
$$w_{t,l} = \frac{1}{N}\sum_{s=1}^N α_{s→t,l} $$

**Span Weight** (Eq. 9): Trọng số của span được tổng hợp từ trọng số token thành viên:
$$w^{sp}_{i,l} = \frac{w̃^{sp}_{i,l}}{\sum_j^{N_l^{sp}} w̃^{sp}_{j,l}}$$

**Span Representation** (Eq. 8): Biểu diễn của span k tại lớp l là trung bình có trọng số của các hidden state token thành viên:
$$U_{k,l} = \frac{\sum_{t∈S_k} w_{t,l} H_{t,l}}  {\sum_{t∈S_k} w_{t,l}}$$

**Định nghĩa span**:
- **Lớp thấp (Lower)**: dùng *Word Spans* — nhóm token thành các đơn vị từ hoàn chỉnh.
- **Lớp cao (Higher)**: trích xuất *Noun Phrases (NPs)* và *Verb Phrases (VPs)* dùng syntactic parser (spaCy).

---

### 3.3 Dynamic Structural Alignment Loss (L_DSA)

Mục tiêu trung tâm của MTA. Tính toán pairwise cosine distance giữa tất cả các cặp span trong một lớp, cho cả teacher lẫn student, sau đó tối thiểu hóa sự khác biệt giữa hai ma trận khoảng cách này (Eq. 10–11):
$$\mathcal{L}_{DSA} = \frac{1}{|\mathcal{L}_{key}|} \sum_{l∈\mathcal{L}_{key}} \mathcal{L}^{(l)}_{DSA}$$
$$\mathcal{L}^{(l)}_{DSA} = \sum_i \sum_{j≠i} w^{sp}_{ij,l} \left(d(U^S_{i,l}, U^S_{j,l}) - d(U^T_{i,l}, U^T_{j,l})\right)^2$$
trong đó `d(·,·)` là cosine distance, và cặp span được weighting theo salience $w^{sp}_{ij,l} = w^{sp}_{i,l} · w^{sp}_{j,l}$.

---

### 3.4 Hidden Representation Alignment (L_Hid)

Vì dimension của student ($d_S$) thường nhỏ hơn teacher ($d_T$), tác giả dùng một learnable linear projector $W_l ∈ ℝ^{d_S × d_T}$ để ánh xạ student vào không gian teacher (Eq. 12):
$$
H̃^S_{t,l} = H^S_{t,l} W_l
$$

Sau đó tối thiểu hóa weighted cosine distance (Eq. 13):
$$\mathcal{L}_{Hid} = \sum_{l∈\mathcal{L}_{key}} \sum_{t∈M_l} w^T_{t,l} \left(1 - \frac{<H̃^S_{t,l}, H^T_{t,l}>} {{‖H̃^S_{t,l}‖}_2 {‖H^T_{t,l}‖}_2}\right)$$

---

## Experimental Setup

- **Datasets**:
  - Training: **DATABRICKS-DOLLY-15K**
  - Evaluation: Dolly test split, **Vicuna** (Chiang et al., 2023), **SELFINST** (Wang et al., 2023b), **S-NI** (Wang et al., 2022)

- **Student Models**:
  - GPT-2 120M (teacher: GPT-2 1.5B)
  - Qwen1.5-0.5B (teacher: Qwen1.5-1.8B)
  - OPT-1.3B (teacher: OPT-6.7B)

- **Baselines**:
  - SFT (Supervised Fine-Tuning)
  - FDD (Gong et al., 2025)
  - DistiLLM (Ko et al., 2024)
  - DistiLLM-2 (Ko et al., 2025)

- **Metrics**: ROUGE-L (Lin, 2004); GPT-4o-mini score (1–100) dùng LLM-as-a-judge

- **Compute**: Single NVIDIA A100 GPU (40 GB); GPT-2 và Qwen1.5 dùng full-parameter fine-tuning; OPT dùng LoRA (rank=256, alpha=8, dropout=0.1)

---

## Results

### Main Findings

1. **MTA tăng hiệu suất nhất quán cho tất cả baselines**: Tích hợp MTA vào FDD, DistiLLM, và DistiLLM-2 đều cải thiện ROUGE-L trên tất cả 4 benchmark. Ví dụ, DistiLLM + MTA đạt avg. 21.45 so với 20.21 của DistiLLM gốc (GPT-2 1.5B → 120M).

2. **Cải thiện GPT-4o-mini score**: FDD + MTA: 19.94 → 21.03; DistiLLM + MTA: 17.46 → 21.45; DistitLLM-2 + MTA: 19.48 → 19.48 (FDD), theo hướng dương trên cả hai kiến trúc GPT-2 và OPT.

3. **L_DSA đóng góp lớn hơn L_Hid**: L_DSA thường mang lại cải thiện lớn hơn (đặc biệt trên S-NI), cho thấy căn chỉnh cấu trúc quan hệ bổ sung nhiều hơn so với point-wise feature matching.

4. **Full MTA (cả L_DSA + L_Hid) tốt nhất**: Synergy của hai loss tốt hơn từng loss riêng lẻ, ví dụ DistiLLM + Full: avg 21.45 so với chỉ +L_Hid: 20.55 và +L_DSA: 20.92.

### Ablation Studies

**Tác động của hierarchical granularity** (Table 3):
- *Word-only*: tốt ở surface tasks nhưng kém trên reasoning-intensive (Vicuna). DistitLLM-2 word-only: avg 19.10.
- *Phrase-only*: thường tốt hơn word-only (21.17 vs 20.80 cho DistiLLM), nhưng vẫn kém MTA full-level (21.45).
- *Full-level (Adaptive)*: tốt nhất — kết hợp lexical grounding ở lớp thấp và compositional semantics ở lớp cao.

**Số lớp distill M** (Figure 6):
- Performance tăng từ M=0 (baseline) đến M=3 (đỉnh 21.45 ROUGE-L).
- M=4 hoặc 5 dẫn đến diminishing returns, gợi ý *inter-layer redundancy*. M=3 được chọn làm optimal.

**Word vs. Phrase allocation** (Table 4):
- Cấu hình tốt nhất: **1 Word : 2 Phrase** (MTA Ours) → avg 21.45.
- All Phrase: 21.17; All Word: 20.80; Hybrid A (2:1): 20.90.

**Span weights** (Table 10):
- DistiLLM + Ours *w/ weight*: 21.45 vs *w/o weight*: 20.66 — trọng số span quan trọng cho hiệu suất.

**Chi phí tính toán** (Table 9):
- DistiLLM: 0.26s/step → DistiLLM + Ours: 0.66s/step (avg_alloc: 6.53 → 6.54 GB; peak: 16.91 → 17.94 GB).
- FDD: 0.49s/step → FDD + Ours: 0.88s/step (peak: 23.04 → 24.05 GB).

---

## Author-Stated Limitations

> "While MTA improves distillation quality, it introduces additional computational cost due to the use of external sparse structures for layer-wise alignment. Although this overhead remains manageable in our current experiments, an important direction for future work is to design more lightweight yet still layer-adaptive alignment mechanisms that reduce computational cost while preserving performance. In addition, our experiments are conducted under fixed computational budgets and benchmark settings."

---

## Key Terms

| Term | Definition (as used in this paper) |
|------|-----------------------------------|
| **Knowledge Distillation (KD)** | Quá trình train model nhỏ (student) để bắt chước model lớn hơn (teacher) bằng cách tối thiểu hóa divergence giữa các phân phối điều kiện của chúng |
| **MTA** | Multi-Granular Trajectory Alignment — framework căn chỉnh quỹ đạo biểu diễn của teacher và student theo cấu trúc phân cấp ngữ nghĩa |
| **Hierarchical Representational Trajectory** | Chuỗi có thứ tự các không gian biểu diễn trong LLM mà granularity ngữ nghĩa thay đổi có hệ thống theo chiều sâu mạng |
| **Dynamic Structural Alignment (L_DSA)** | Loss căn chỉnh hình học quan hệ pairwise giữa các span trong cùng một lớp của teacher và student |
| **Hidden Representation Alignment (L_Hid)** | Loss căn chỉnh feature-level trực tiếp giữa hidden state đã chọn của teacher và student qua weighted cosine distance |
| **Word Span** | Nhóm token tạo thành một từ hoàn chỉnh — dùng ở các lớp thấp để grounding từ vựng |
| **Phrase Span** | Noun Phrases (NP) hoặc Verb Phrases (VP) được trích xuất bởi syntactic parser (spaCy) — dùng ở các lớp cao để nắm bắt compositional semantics |
| **Token Weight (w_{t,l})** | Trọng số quan trọng của token tại lớp l, tính bằng mean attention nhận được từ tất cả token khác (không có self-loop) |
| **Feature Dynamics Distillation (FDD)** | Phương pháp distillation xem chiều sâu Transformer như discrete time steps của dynamical system liên tục, căn chỉnh cả trajectory lẫn derivative (Gong et al., 2025) |
| **DistiLLM** | Framework distillation dùng Skew KLD và adaptive off-policy generation (Ko et al., 2024) |
| **DistiLLM-2** | Framework distillation mở rộng DistiLLM với contrastive approach (CALD) và curriculum-based adaptive learning (Ko et al., 2025) |
| **L_key** | Tập con các lớp trung gian được chọn để distillation, dùng strided top-down approach |
| **ROUGE-L** | Metric đánh giá chất lượng text generation dựa trên Longest Common Subsequence |

---

## Notable Quotes

> "Most existing Knowledge Distillation (KD) approaches for LLMs often focus on minimizing the divergence between the output probability distributions of the teacher and the student... these methods overlook the structural knowledge embedded within the intermediate representations." — Section 1

> "We therefore view a model's internal representations as forming a hierarchical representational trajectory: an ordered sequence of representation spaces whose semantic granularity systematically changes with depth. Under this perspective, applying a single, uniform alignment rule across all layers is suboptimal." — Section 3.1

> "The Full-level (Layer-Adaptive) strategy consistently achieves the highest scores across all baselines (e.g., 20.50 for FDD and 21.45 for DistiLLM)." — Section 5.2

---

## References Worth Following

- **Gong et al., 2025** — *Beyond logits: Aligning feature dynamics for effective knowledge distillation* (FDD) — framework tiền nhiệm được MTA mở rộng; published ACL 2025.
- **Ko et al., 2024** — *Distillm: Towards streamlined distillation for large language models* (DistiLLM) — baseline chính; ICML 2024.
- **Ko et al., 2025** — *Distillm-2: A contrastive approach boosts the distillation of LLMs* (DistiLLM-2) — baseline nâng cao; ICML 2025 Spotlight.
- **Tenney et al., 2019** — *BERT rediscovers the classical NLP pipeline* — bằng chứng thực nghiệm cho tổ chức phân cấp của Transformer layers.
- **Wang et al., 2025** / **Yang et al., 2025** — Phân tích vai trò của các lớp LLM trong retrieval, reasoning và subtask scheduling.
- **Hinton et al., 2015** — *Distilling the knowledge in a neural network* — bài báo gốc về Knowledge Distillation.
