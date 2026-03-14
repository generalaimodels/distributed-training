# LLM Pretraining Playbook

> **A comprehensive, research-grade technical documentation suite for large language model pretraining** — covering the full lifecycle from strategic justification through architecture design, data curation, scaling laws, optimizer configuration, distributed training infrastructure, ablation methodology, and post-training alignment.

All findings are grounded in empirical ablation studies. **SmolLM3-3B** (Hugging Face) serves as the running case study throughout, illustrating each decision with concrete hyperparameter configurations, benchmark results, and engineering trade-offs.

---

## Recommended Reading Order

The documents below are ordered to mirror the natural decision sequence of an LLM pretraining project — from *why* to *what* to *how*.

| # | Document | Synopsis |
|---|---|---|
| 1 | [Strategic Justification](./01_strategic_justification.md) | When (and when **not**) to pretrain. Decision-theoretic framework, anti-patterns, hierarchical evaluation protocol. |
| 2 | [Design Specification](./02_design_specification.md) | Translating strategic objectives → architecture, data mixture, compute, and organizational specifications. |
| 3 | [Data Curation](./03_data_curation.md) | Data source taxonomy, quality stratification, mathematical mixture optimization, multi-stage curriculum training. |
| 4 | [Data Mixture Case Study](./04_data_mixture_case_study.md) | SmolLM3 data mixture design — English web, multilingual, code, math — with ablation-driven validation. |
| 5 | [Scaling Laws](./05_scaling_laws.md) | Kaplan formulation, Chinchilla revision, overtraining paradigm, inference-aware compute-optimal training. |
| 6 | [Optimizer & Hyperparameters](./06_optimizer_and_hyperparameters.md) | AdamW, Muon, learning rate schedules (cosine, WSD), batch size configuration, critical batch size theory. |
| 7 | [Training Infrastructure](./07_training_infrastructure.md) | GPU architecture, NVLink/PCIe/EFA communication, NVMe/WekaFS/Lustre storage, fault resilience, throughput optimization. |
| 8 | [Ablation Methodology](./08_ablation_methodology.md) | Small-scale experimentation → large-model decisions. Training framework selection, sequential integration, evaluation rigor. |
| 9 | [Post-Training](./09_post_training.md) | SFT, continued pretraining, preference optimization (DPO/APO), RLVR with GRPO, hybrid reasoning models. |

---

## Architecture Deep-Dives

The [`architecture_design/`](./architecture_design/) module provides self-contained technical reports on each architectural component of the transformer.

| # | Document | Synopsis |
|---|---|---|
| A1 | [Tokenizer Design](./architecture_design/01_tokenizer_design.md) | BPE algorithm, vocabulary sizing, fertility metrics, cross-linguistic benchmarking, domain-specific considerations. |
| A2 | [Embeddings & Positional Encoding](./architecture_design/02_embeddings_and_positional_encoding.md) | Weight tying ablation, RoPE/NoPE/RNoPE, ABF/YaRN context extension, sliding window & chunked attention, attention sinks. |
| A3 | [Attention Mechanisms](./architecture_design/03_attention_mechanisms.md) | MHA, MQA, GQA, MLA — KV cache analysis, iso-parameter ablation, intra-document masking. |
| A4 | [Sparsity & MoE](./architecture_design/04_sparsity_and_moe.md) | MoE routing, efficiency leverage, expert granularity, shared experts, load balancing, hybrid SSM/linear attention. |
| A5 | [Training Stability](./architecture_design/05_training_stability.md) | Z-loss, QK-norm, weight decay exclusion, initialization strategies, SwiGLU, depth-width trade-offs. |

---

## Reference Materials

Source presentation slides and PDFs are preserved in [`assets/notes/`](./assets/notes/):

| File | Description |
|---|---|
| `LLM_Pretraining_Playbook.pdf` / `.pptx` | End-to-end pretraining playbook slides |
| `LLM_Scaling_Mathematics.pdf` / `.pptx` | Scaling law derivations and mathematical analysis |
| `LLM_Tokenization_Architecture.pdf` / `.pptx` | Tokenizer architecture and evaluation |
| `Modern_LLM_Architecture_Blueprints.pdf` / `.pptx` | Architecture design blueprints and comparisons |
| `Scientific_LLM_Development.pdf` / `.pptx` | Scientific methodology for LLM development |

---

## Key Models & Architectures Referenced

| Model | Role in Documentation |
|---|---|
| **SmolLM3-3B** | Primary case study — all ablations, hyperparameter configs, and design decisions demonstrated |
| Llama 3 / 3.1 / 3.2 | Baseline architecture and tokenizer reference |
| Qwen 2.5 / 3 | Multilingual benchmarking, hybrid reasoning template |
| DeepSeek-V2 / V3 | MLA attention, MoE routing, loss-free load balancing |
| Kimi K2 | High-sparsity MoE reference |

---

## Prerequisites

This documentation assumes familiarity with:

- Transformer architecture fundamentals (attention, FFN, residual connections)
- Gradient-based optimization (SGD, Adam family)
- Basic information theory (cross-entropy, perplexity)
- Linear algebra (matrix operations, singular value decomposition)
- GPU computing concepts (FLOPS, memory bandwidth, mixed precision)

---

*Each document is self-contained with its own table of contents, mathematical formulations, code examples, ablation tables, and references.*
