# MoTTT: Test-Time LoRA Scratchpad with Query-Aware Expert Mixture

Implementation of **MoTTT** as specified in [ideas.md](ideas.md).

MoTTT decouples factual ingestion / episodic memory from multi-step domain reasoning:
1. **Dynamic Memory Scratchpad (Test-Time LoRA $\mathcal{C}=\{0\}$)**: Adapts dynamically via NLL on chunked long contexts ($K=256$, unit scaling $\gamma = \alpha / r = 1.0$).
2. **Pre-trained LoRA Reasoning Experts ($\mathcal{R}=\{1 \dots E\}$)**: Specialized in multi-step reasoning.
3. **Query-Aware MLP Router with Asymmetric Load Balancing**: Protects reasoning experts from collapse while freeing the scratchpad from artificial quotas.

---

## 1. Environment Setup

The local environment is built with **Python 3.12** and managed via `uv`.

### Local CPU Environment (Current)
To keep disk usage minimal and run without requiring local GPU resources:
```bash
# Activate virtual environment
source .venv/bin/activate

# Or run commands directly via .venv
.venv/bin/python scripts/run_smoke_test.py
.venv/bin/pytest tests/ -v
```

### Remote GPU Cluster Deployment
When deploying to a remote machine equipped with NVIDIA GPUs (CUDA 12.4+):
```bash
uv venv --python python3.12 .venv
source .venv/bin/activate
uv pip install -r requirements-gpu.txt
uv pip install -e .
```

---

## 2. Dataset Engineering: Distractor-Injected Long-Context GSM8K

MoTTT provides an automated data pipeline using Hugging Face's `openai/gsm8k` dataset that:
1. Decomposes problems into a factual premise $\mathcal{P}$ and test-time query $\mathcal{Q}$.
2. Injects premise needles into non-numerical distractor text across depth ratios $\delta \in [0.1, 0.9]$ ($L_{\text{ctx}} \in [4096, 8192]$).
3. Exports benchmark copies in standardized formats (JSONL & Hugging Face Dataset) for evaluating other baseline models on the exact same data.

### Generating the Benchmark Dataset
```bash
# Generate distractor dataset (e.g. for test split)
.venv/bin/python scripts/build_distractor_dataset.py \
    --split test \
    --depth_ratios 0.1,0.3,0.5,0.7,0.9 \
    --target_context_tokens 4096 \
    --output_dir data/gsm8k_distractor
```

### Evaluating Other Baseline Models on the Benchmark
To benchmark any external Hugging Face model (e.g. `Qwen/Qwen2.5-0.5B`, LLaMA, Mistral) on the modified dataset:
```bash
# Run baseline evaluation on GPU
.venv/bin/python scripts/evaluate_baseline.py \
    --dataset_path data/gsm8k_distractor/gsm8k_distractor_all.jsonl \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --output_results results/qwen_baseline_results.json

# Or test in mock mode on CPU:
.venv/bin/python scripts/evaluate_baseline.py \
    --dataset_path data/gsm8k_distractor/gsm8k_distractor_all.jsonl \
    --mock
```

---

## 3. Running Local Verification (Mock / CPU Mode)

All local tests run against lightweight synthetic tensors and mock modules without downloading heavy weights or external datasets:

```bash
# Run unit tests (12 tests covering router, loss, scratchpad, GSM8K pipeline, and export)
.venv/bin/pytest tests/ -v

# Run smoke test script
.venv/bin/python scripts/run_smoke_test.py
```

---

## 4. Directory Layout

```
MoTTT/
├── ideas.md                    # Core mathematical and architecture specification
├── pyproject.toml              # Packaging and dependency configuration
├── requirements-cpu.txt        # CPU-optimized requirements (local)
├── requirements-gpu.txt        # CUDA requirements (remote GPU cluster)
├── scripts/
│   ├── run_smoke_test.py       # End-to-end CPU smoke verification
│   ├── build_distractor_dataset.py # GSM8K distractor benchmark builder
│   └── evaluate_baseline.py    # Multi-model baseline evaluation runner
├── src/
│   └── mottt/
│       ├── configs/
│       │   ├── default.yaml    # Target Qwen2.5-0.5B config for GPU
│       │   └── smoke_test.yaml # Mock test config for CPU
│       ├── data/
│       │   ├── gsm8k_loader.py         # Hugging Face GSM8K loader & answer extractor
│       │   ├── distractor_generator.py # Premise/Query separation & needle synthesis
│       │   └── dataset_exporter.py     # JSONL & HF Dataset exporter for baseline testing
│       └── models/
│           ├── router.py       # QueryAwareRouter & AsymmetricBalancingLoss
│           ├── scratchpad.py   # TestTimeScratchpadLoRA with unit scaling
│           └── mottt_model.py  # Unified MoTTT architecture wrapper
└── tests/
    ├── test_router.py
    ├── test_scratchpad_inner_loop.py
    ├── test_distractor_pipeline.py
    ├── test_mottt_model.py
    └── test_gsm8k_pipeline.py
```
