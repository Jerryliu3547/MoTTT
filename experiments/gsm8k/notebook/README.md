# GSM8K Experiment Notebooks: MoTTT vs. Full Fine-Tuning

This directory contains standalone Jupyter notebooks for running experiments on the **GSM8K Distractor Benchmark**:

## Available Notebooks

### 1. `gsm8k_experiment.ipynb` (MoTTT Architecture)
The standalone notebook for the **MoTTT** (Mixture of Test-Time Trained LoRAs) framework:
1. **Distractor Needle Synthesis**: Buries factual premise needles ($\mathcal{P}$) in non-numerical distractor text across needle depth ratios $\delta \in [0.1, 0.9]$.
2. **Model Assembly**: Integrates the Query-Aware Router MLP, reasoning LoRA experts ($\mathcal{R}=\{1 \dots E\}$), and dynamic scratchpad ($\mathcal{C}=\{0\}$).
3. **Outer-Loop Training**: Optimizes router and reasoning experts with Asymmetric Load Balancing loss $\mathcal{L}_{\text{bal}}$.
4. **Test-Time Adaptation & Evaluation**: Streaming inner-loop gradient updates on context chunks ($K=256$, $\gamma = 1.0$) and multi-depth evaluation.
5. **Resilience & Gate Analysis**: Visualizes Lost-in-the-Middle resilience and module gate allocation percentages.

### 2. `gsm8k_full_finetune.ipynb` (Full Fine-Tuning SFT Baseline)
The standalone notebook implementing the **Full Fine-Tuning (SFT)** baseline and evaluation:
1. **Environment & Cluster Detection**: Automatic detection of Google Colab, Georgia Tech Phoenix cluster (SLURM), and local environments.
2. **Loss Masking Collate**: Prompt tokens masked with `-100` so loss is computed strictly on mathematical reasoning and solution tokens.
3. **Full Parameter Unfreezing**: Unfreezes 100% of parameters in `Qwen/Qwen2.5-0.5B` (~494M trainable parameters) with gradient checkpointing.
4. **SFT Training Loop**: AdamW optimization with linear warmup and cosine decay scheduler.
5. **Multi-Depth Evaluation**: Tests factual retrieval and reasoning across needle depth ratios $\delta \in [0.1, 0.3, 0.5, 0.7, 0.9]$.
6. **Lost-in-the-Middle Analysis & MoTTT Comparison**: Plots accuracy degradation in the middle ($\delta = 0.5$) and compares side-by-side against MoTTT.
7. **Qualitative Traces**: Interactive DataFrame and step-by-step reasoning trace inspector.

---

## How to Run

### Local Execution
```bash
conda activate mottt
jupyter lab experiments/gsm8k/notebook/
```
Select the **Python 3 (ipykernel)** kernel. If no CUDA GPU is detected, both notebooks automatically default to mock/CPU mode for rapid verification.

### Google Colab
1. Upload or clone the repository to Google Drive or Colab workspace.
2. Open either notebook (`gsm8k_experiment.ipynb` or `gsm8k_full_finetune.ipynb`).
3. Under **Runtime → Change runtime type**, select **GPU** (T4, L4, or A100).
4. Run all cells in order (`Runtime → Run all`).

### Georgia Tech Phoenix Cluster (Slurm)
Request an interactive GPU node or launch a Jupyter notebook server on a GPU compute node:
```bash
# Request an interactive GPU node (e.g. A100 or V100)
salloc -p gpu-a100 -N 1 --gres=gpu:1 -t 04:00:00

# Activate conda environment and launch Jupyter Lab
conda activate mottt
jupyter lab --no-browser --port=8888
```
Then port-forward `8888` over SSH to your local machine:
```bash
ssh -L 8888:localhost:8888 <username>@login-phoenix.pace.gatech.edu
```
Open `http://localhost:8888` in your browser to run the notebooks with full GPU acceleration.

