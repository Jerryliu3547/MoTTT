# Implementation Plan: Test-Time LoRA Scratchpad with Query-Aware Expert Mixture (MoTTT)

## 1. Executive Summary & Objective

This implementation plan formalizes an architecture that enables small language models (specifically **Qwen2.5-0.5B**) to perform robust mathematical reasoning across extended, distractor-heavy contexts (4,000 to 8,000+ tokens) without catastrophic forgetting or context degradation.

The architecture decouples **factual ingestion / episodic memory** from **domain reasoning**:
1. **Dynamic Memory Scratchpad (Test-Time LoRA):** A dedicated regular LoRA adapter trained from scratch at test time via Next-Token Prediction (NLL) on chunked, indexed long contexts (256-token effective receptive field, $C = 32$ chunks for 8,192 tokens).
2. **Pre-trained LoRA Reasoning Experts:** An offline-trained ensemble of $E \in [4, 6]$ regular LoRA experts specialized in multi-step mathematical and symbolic reasoning (e.g., GSM8K, MATH).
3. **Query-Aware MLP Router with Asymmetric Load Balancing:** A dynamic gating module conditioned on both the base model's current token representation and the global pooled embedding of the user's standalone question. It employs **Asymmetric Load Balancing** during offline meta-training to prevent expert collapse among the reasoning experts without imposing an artificial statistical quota on the test-time scratchpad.

---

## 2. Mathematical Formalization & Notation

| Symbol | Definition | Default Value / Range |
| :--- | :--- | :--- |
| $\mathcal{M}$ | Base Model | `Qwen/Qwen2.5-0.5B` |
| $d$ | Hidden state dimension of $\mathcal{M}$ | $896$ |
| $L_{\text{layers}}$ | Total transformer layers | $24$ |
| $E$ | Number of static reasoning LoRA experts | $E \in [4, 6]$ (Default: $4$) |
| $\mathcal{C}$ | Context / Scratchpad index set | $\{0\}$ |
| $\mathcal{R}$ | Reasoning expert index set | $\{1, \dots, E\}$ |
| $C$ | Number of chunks in chunked long-context memory | $C = L_{\text{ctx}} / 256$ ($C = 32$ for $8,192$ tokens) |
| $K$ | Chunk token length | $256$ tokens |
| $M$ | Total active tokens evaluated in an outer batch | $M = \sum \text{batch\_size} \times \text{seq\_len}$ |
| $r$ | LoRA rank | $16$ |
| $\alpha$ | LoRA scaling parameter | $16$ |
| $\Delta W_{\text{pad}}$ | Weight update matrix from dynamic scratchpad LoRA | $\mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ |
| $\Delta W_{\text{exp}}^{(j)}$ | Weight update matrix of expert $j \in \mathcal{R}$ | $\mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ |
| $g_t$ | Full routing gate vector at decode step $t$ | $\Delta^{E}$ (simplex over $\mathcal{C} \cup \mathcal{R}$) |
| $\lambda_{\text{bal}}$ | Weight for asymmetric load balancing loss | $0.01$ |

### 2.1 Standard LoRA Scaling Analysis
Standard LoRA computes the adapted forward pass as:
$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x$$

Where the constant scaling multiplier is defined as:
$$\gamma = \frac{\alpha}{r}$$

* Setting $r = 16$ and $\alpha = 16$ yields an exact unit multiplier:
  $$\gamma = \frac{16}{16} = 1.0$$
* This unit multiplier ensures stable gradient propagation during the fast test-time inner loop adaptation.

---

## 3. Asymmetric Load Balancing Formulation

Standard MoE auxiliary losses (e.g., Switch Transformer / GShard load balancing) force equal average routing across all available pathways. In MoTTT, applying a uniform quota across both the reasoning experts and the dynamic scratchpad causes fundamental failures:
* **The Problem with Symmetric Balancing:** The Scratchpad adapter contains sample-specific episodic memory. It should *only* be queried when the model needs to retrieve factual premises from the distractor context. Penalizing the router for ignoring the scratchpad during purely symbolic derivation forces spurious factual hallucination. Conversely, penalizing the router for prioritizing the scratchpad during retrieval starves the reasoning experts.
* **The Asymmetric Solution:** Split the expert pool into the Context set $\mathcal{C} = \{0\}$ and the Reasoning set $\mathcal{R} = \{1, \dots, E\}$. Calculate the auxiliary balancing loss strictly over $\mathcal{R}$.

### 3.1 Mathematical Loss Definition
Let $z_k^{(i)}$ denote the pre-softmax router logit assigned to expert $i$ for token $k \in \{1, \dots, M\}$.

1. **Restricted Reasoning Probabilities:**
   For each token $k$, isolate the logits belonging to $\mathcal{R}$ and re-normalize across $\mathcal{R}$:
   $$P_{\text{reason}}^{(i)}(k) = \frac{\exp(z_k^{(i)} / \tau)}{\sum_{j \in \mathcal{R}} \exp(z_k^{(j)} / \tau)} \quad \forall i \in \mathcal{R}$$

2. **Average Reasoning Expert Allocation:**
   Compute the mean empirical routing allocation $q_{\text{reason}}^{(i)}$ across the batch of $M$ tokens:
   $$q_{\text{reason}}^{(i)} = \frac{1}{M} \sum_{k=1}^M P_{\text{reason}}^{(i)}(k) = \frac{1}{M} \sum_{k=1}^M \frac{\exp(z_k^{(i)} / \tau)}{\sum_{j \in \mathcal{R}} \exp(z_k^{(j)} / \tau)}$$

3. **Asymmetric Balancing Objective:**
   Maximize entropy across $\mathcal{R}$ via negative sum log loss:
   $$\mathcal{L}_{\text{balance\_reason}} = -\sum_{i \in \mathcal{R}} \log\left(q_{\text{reason}}^{(i)} + \epsilon\right)$$
   *(Minimized when $q_{\text{reason}}^{(i)} = \frac{1}{\vert{}\mathcal{R}\vert{}} = \frac{1}{E}$ for all $i \in \mathcal{R}$.)*

4. **Total Meta-Training Loss:**
   $$\mathcal{L}_{\text{outer}} = \mathcal{L}_{\text{task}} + \lambda_{\text{bal}} \cdot \mathcal{L}_{\text{balance\_reason}}$$

### 3.2 Dual Operational Guarantees
* **Protects Reasoning LoRAs:** Router capacity cannot collapse into a single dominant reasoning path. Different reasoning LoRAs are incentivized to specialize in distinct sub-skills (e.g., algebraic computation, multi-step planning, symbolic deduction).
* **Frees the Scratchpad:** Router utilization of the dynamic scratchpad $\mathcal{C}$ is unconstrained by $\mathcal{L}_{\text{balance\_reason}}$. It is driven strictly by $\nabla_{\theta} \mathcal{L}_{\text{task}}$, engaging only when factual premise recovery improves task prediction.

---

## 4. Dataset Engineering: Distractor-Injected Long-Context GSM8K / MATH

### 4.1 Separation of Premise and Query
In standard GSM8K, problems intertwine context premises with the prompt query:
* **Original Problem:** *"Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"*
* **Decomposed Premise (Stored in Long Context):**
  $$\mathcal{P} = \text{"Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May."}$$
* **Isolated Query (Supplied at Generation Time Only):**
  $$\mathcal{Q} = \text{"How many clips did Natalia sell altogether in April and May?"}$$

### 4.2 Distractor Synthesis & Needle Embedding
1. **Distractor Reservoir:** Sample narrative sequences from English Wikipedia or Project Gutenberg books, pre-filtered of mathematical and numerical tokens that could induce spurious reasoning paths.
2. **Context Length Specification:** Construct total text sequences spanning $L_{\text{ctx}} \in [4096, 8192]$ tokens.
3. **Needle Placement:** Insert premise $\mathcal{P}$ at depth ratio $\delta \in [0.1, 0.9]$ to evaluate robustness against *Lost-in-the-Middle* degradation.