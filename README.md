# MoTTT: Test-Time LoRA Scratchpad with Query-Aware Expert Mixture

Implementation of **MoTTT** as specified in [ideas.md](ideas.md).

MoTTT decouples factual ingestion / episodic memory from multi-step domain reasoning:
1. **Dynamic Memory Scratchpad (Test-Time LoRA $\mathcal{C}=\{0\}$)**: Adapts dynamically via NLL on chunked long contexts ($K=256$, unit scaling $\gamma = \alpha / r = 1.0$).
2. **Pre-trained LoRA Reasoning Experts ($\mathcal{R}=\{1 \dots E\}$)**: Specialized in multi-step reasoning.
3. **Query-Aware MLP Router with Asymmetric Load Balancing**: Protects reasoning experts from collapse while freeing the scratchpad from artificial quotas.

---

## 1. Environment Setup (Conda)

The environment is built with **Python 3.12** and managed via **Conda**.

### Option A: Local CPU Environment (Current)
To keep disk usage minimal and run locally without requiring GPU resources:
```bash
# 1. Create and activate the conda environment
conda create -n mottt python=3.12 -y
conda activate mottt

# 2. Install dependencies (CPU-optimized PyTorch build)
pip install -r requirements-cpu.txt
pip install -e .

# (Or create directly via environment.yml)
# conda env create -f environment.yml
# conda activate mottt
```

### Option B: Remote GPU Cluster Deployment (CUDA 12.4+)
When deploying to a remote machine equipped with NVIDIA GPUs:
```bash
# 1. Create and activate the conda environment
conda create -n mottt python=3.12 -y
conda activate mottt

# 2. Install GPU-accelerated PyTorch & dependencies
pip install -r requirements-gpu.txt
pip install -e .

# (Or create directly via environment-gpu.yml)
# conda env create -f environment-gpu.yml
# conda activate mottt
```

---

## 2. Dataset Engineering: Distractor-Injected Long-Context GSM8K

MoTTT provides an automated data pipeline using Hugging Face's `openai/gsm8k` dataset that:
1. Decomposes problems into a factual premise $\mathcal{P}$ and test-time query $\mathcal{Q}$.
2. Injects premise needles into non-numerical distractor text across depth ratios $\delta \in [0.1, 0.9]$ ($L_{\text{ctx}} \in [4096, 8192]$).
3. Exports benchmark copies in standardized formats (JSONL & Hugging Face Dataset) for evaluating other baseline models on the exact same data.

### Generating the Benchmark Dataset
```bash
conda activate mottt

# Generate distractor dataset (e.g. for test split with fixed grid)
python scripts/build_distractor_dataset.py \
    --split test \
    --depth_ratios 0.1,0.3,0.5,0.7,0.9 \
    --target_context_tokens 4096 \
    --output_dir data/gsm8k_distractor

# Or generate training dataset with random depth per example (prevents 5x dataset inflation):
python scripts/build_distractor_dataset.py \
    --split train \
    --random_depth \
    --random_mode choice \
    --depth_ratios 0.1,0.3,0.5,0.7,0.9 \
    --target_context_tokens 4096 \
    --output_dir data/gsm8k_distractor_train
```

### Evaluating Other Baseline Models on the Benchmark
To benchmark any external Hugging Face model (e.g. `Qwen/Qwen2.5-0.5B`, LLaMA, Mistral) on the modified dataset:
```bash
conda activate mottt

# Run baseline evaluation on GPU
python scripts/evaluate_baseline.py \
    --dataset_path data/gsm8k_distractor/gsm8k_distractor_all.jsonl \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --output_results results/qwen_baseline_results.json

# Or test in mock mode on CPU:
python scripts/evaluate_baseline.py \
    --dataset_path data/gsm8k_distractor/gsm8k_distractor_all.jsonl \
    --mock
```

---

## 3. Dedicated Experiment Pipelines (`experiments/`)

### A. GSM8K Experiment Suite (`experiments/gsm8k/`)
```bash
conda activate mottt

# 1. Prepare distractor long-context datasets
python experiments/gsm8k/data_preparation.py --depth_ratios 0.1,0.3,0.5,0.7,0.9 --output_dir experiments/gsm8k/data

# 2. Train MoTTT with Qwen2.5-0.5B and Query-Aware Router
python experiments/gsm8k/model_train.py \
    --data_path experiments/gsm8k/data/train_distractor.jsonl \
    --base_model_name Qwen/Qwen2.5-0.5B \
    --output_dir experiments/gsm8k/checkpoints \
    --epochs 3

# 3. Test model: inner-loop scratchpad adaptation & depth evaluation
python experiments/gsm8k/model_test.py \
    --test_data experiments/gsm8k/data/test_distractor.jsonl \
    --checkpoint_dir experiments/gsm8k/checkpoints \
    --output_dir experiments/gsm8k/results

# 4. Interactive Standalone Jupyter Notebook (All-in-One: Prep -> Train -> Test -> Viz)
jupyter lab experiments/gsm8k/notebook/gsm8k_experiment.ipynb
```

### B. LongBench-v2 Experiment Suite (`experiments/longbenchv2/`)
See [experiments/longbenchv2/README.md](experiments/longbenchv2/README.md) for extreme long-context evaluation ($8k \to 100k$ tokens).

---

## 4. Running Local Verification (Mock / CPU Mode)

All local tests run against lightweight synthetic tensors and mock modules without downloading heavy weights or external datasets:

```bash
conda activate mottt

# Run unit tests (13 tests covering router, loss, scratchpad, GSM8K, and experiment cycles)
pytest tests/ -v

# Run smoke test script
python scripts/run_smoke_test.py
```

---

## 5. Directory Layout

```
MoTTT/
├── ideas.md                    # Core mathematical and architecture specification
├── environment.yml             # Conda environment definition (CPU / local)
├── environment-gpu.yml         # Conda environment definition (CUDA GPU)
├── pyproject.toml              # Packaging and dependency configuration
├── requirements-cpu.txt        # CPU-optimized requirements (local)
├── requirements-gpu.txt        # CUDA requirements (remote GPU cluster)
├── experiments/
│   ├── gsm8k/
│   │   ├── data_preparation.py # GSM8K distractor preparation
│   │   ├── model_train.py      # Qwen2.5-0.5B + MoTTT training loop
│   │   ├── model_test.py       # Test-time scratchpad adaptation & evaluation
│   │   └── notebook/           # Standalone Jupyter notebook (prep -> train -> test -> viz)
│   └── longbenchv2/
│       ├── README.md           # LongBench-v2 specifications
│       ├── data_preparation.py # LongBench-v2 task formatting
│       └── model_test.py       # LongBench-v2 evaluation runner
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
    ├── test_gsm8k_pipeline.py
    └── test_experiments_gsm8k.py
```
