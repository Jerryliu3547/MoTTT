# LongBench-v2 Evaluation for MoTTT

This experiment evaluates **MoTTT** on **LongBench-v2**, a standardized benchmark designed to evaluate language models across long, distractor-heavy contexts spanning 8,000 to 100,000+ tokens.

---

## 1. Benchmark Overview

LongBench-v2 tests realistic long-context understanding where key evidence is buried within large documents:
- **Context Length**: $L_{\text{ctx}} \in [8k, 100k]$ tokens
- **Tasks**:
  - Single-document QA (in-depth comprehension)
  - Multi-document cross-evidence reasoning
  - Long summarization & factual aggregation
  - Codebase analysis and dependency navigation

---

## 2. MoTTT's Scaling to LongBench-v2

MoTTT decouples factual context absorption from reasoning:
1. **Dynamic Memory Scratchpad ($C$ chunks)**:
   - For $L_{\text{ctx}} = 8,192$ tokens: $C = 8192 / 256 = 32$ chunks.
   - For $L_{\text{ctx}} = 32,768$ tokens: $C = 32768 / 256 = 128$ chunks.
   - The test-time LoRA scratchpad performs fast NLL adaptation across all $C$ chunks with unit scaling $\gamma = 1.0$.
2. **Query-Aware Routing**:
   - The router dynamically queries the adapted scratchpad only when evidence retrieval from the document is necessary, while delegating multi-step deductive steps to the pre-trained reasoning experts.
3. **Asymmetric Load Balancing**:
   - Prevents reasoning expert collapse without penalizing the router for utilizing the scratchpad during heavy factual retrieval.

---

## 3. Workflow & Usage

### Data Preparation
```bash
.venv/bin/python experiments/longbenchv2/data_preparation.py --split test --output_dir experiments/longbenchv2/data
```

### Model Evaluation
```bash
.venv/bin/python experiments/longbenchv2/model_test.py \
    --test_data experiments/longbenchv2/data/longbenchv2_chunked.jsonl \
    --checkpoint_dir experiments/gsm8k/checkpoints \
    --output_dir experiments/longbenchv2/results
```
