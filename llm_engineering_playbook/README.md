# Large Language Models (LLMs) Cheat Book

Welcome to the **LLMs Cheat Book**, a rigorous and technically deep collection of notes covering everything from the fundamental building blocks of modern NLP to the absolute state-of-the-art in Generative AI. 

Whether you are a researcher, AI engineer, or student, this guide provides structured, mathematically accurate, and visually comprehensive explanations of the core concepts in the field.

---

## 📖 Table of Contents & Reading Order

For the best learning experience, it is recommended to read the chapters in the following order. **Our UI navigation builder automatically structures the left-rail according to these chapter definitions.**

### Part 1: Foundations

### Chapter 1: Word Embeddings
*   *Covers:* Distributional hypothesis, vector semantics, count-based vs. prediction-based models (Word2Vec, SGNS).

### Chapter 2: Language Models
*   *Covers:* Autoregressive factorization, n-grams to neural LMs, early RNNs, scaling laws, and emergent capabilities.

### Part 2: Architecture & Pretraining Overview

### Chapter 3: Transformers
*   *Covers:* The complete Transformer architecture, scaled dot-product attention, multi-head self-attention, positional encodings.

### Chapter 4: Language Model Pretraining
*   *Covers:* Self-supervised objectives, ELMo, encoder-based models, and evaluation benchmarks.

### Part 3: The Pretraining Playbook
An in-depth guide covering the full lifecycle of LLM pretraining using SmolLM3-3B as a case study.

### Chapter 5: Strategic Justification
*   *Covers:* Decision-theoretic framework, anti-patterns, hierarchical evaluation.

### Chapter 6: Design Specification
*   *Covers:* Translating strategic objectives to architecture and compute specifictions.

### Chapter 7: Data Curation
*   *Covers:* Source taxonomy, mathematical mixture optimization, and curriculum training.

### Chapter 8: Data Mixture Case Study
*   *Covers:* English web, multilingual, code, math — with ablation-driven validation.

### Chapter 9: Scaling Laws
*   *Covers:* Kaplan/Chinchilla formulation, overtraining paradigm, inference-aware training.

### Chapter 10: Optimizer and Hyperparameters
*   *Covers:* AdamW, Muon, learning rate schedules, critical batch size theory.

### Chapter 11: Training Infrastructure
*   *Covers:* GPU architecture, NVLink, storage, fault resilience, throughput optimization.

### Chapter 12: Ablation Methodology
*   *Covers:* Small-scale experimentation → large-model decisions, sequential integration.

### Part 4: Adaptation & Alignment

### Chapter 13: Post Training
*   *Covers:* Continued pretraining, RLVR, preference optimization.

### Chapter 14: Fine-Tuning and Alignment
*   *Covers:* Supervised Fine-Tuning (SFT), Instruction Tuning, RLHF, and DPO.

### Chapter 15: Parameter-Efficient Fine-Tuning
*   *Covers:* Knowledge Distillation, Quantization (GPTQ, AWQ, QLoRA), LoRA, DoRA.

### Part 5: Advanced Paradigms

### Chapter 16: Prompt Engineering
*   *Covers:* In-context learning, Chain-of-Thought (CoT), Tree of Thoughts (ToT).

### Chapter 17: Augmented Language Models
*   *Covers:* RAG architecture, Vector Indices, Tool Calling, and Agentic frameworks.

### Chapter 18: Multimodal
*   *Covers:* Cross-lingual transfer, tokenization equity, contrastive alignment.

### Chapter 19: Recent Advances
*   *Covers:* Novel architectures (State Space Models/Mamba, Mixture of Experts), sequence routing.

---

### 🎨 Visual Assets
This directory also contains an `assets/` folder with generated, scientifically accurate 8K diagrams and architectural visualisations embedded within the markdown files.

---

**Happy Learning!**
