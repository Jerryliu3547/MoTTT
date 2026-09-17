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

## 2. Running Local Verification (Mock / CPU Mode)

All local tests run against lightweight synthetic tensors and mock modules without downloading heavy weights or external datasets:

```bash
# Run unit tests
.venv/bin/pytest tests/ -v

# Run smoke test script
.venv/bin/python scripts/run_smoke_test.py
```

---

## 3. Directory Layout

```
MoTTT/
├── ideas.md                    # Core mathematical and architecture specification
├── pyproject.toml              # Packaging and dependency configuration
├── requirements-cpu.txt        # CPU-optimized requirements (local)
├── requirements-gpu.txt        # CUDA requirements (remote GPU cluster)
├── scripts/
│   └── run_smoke_test.py       # End-to-end CPU smoke verification
├── src/
│   └── mottt/
│       ├── configs/
│       │   ├── default.yaml    # Target Qwen2.5-0.5B config for GPU
│       │   └── smoke_test.yaml # Mock test config for CPU
│       ├── data/
│       │   └── distractor_generator.py  # Premise/Query separation & needle embedding
│       └── models/
│           ├── router.py       # QueryAwareRouter & AsymmetricBalancingLoss
│           ├── scratchpad.py   # TestTimeScratchpadLoRA with unit scaling
│           └── mottt_model.py  # Unified MoTTT architecture wrapper
└── tests/
    ├── test_router.py
    ├── test_scratchpad_inner_loop.py
    ├── test_distractor_pipeline.py
    └── test_mottt_model.py
```
