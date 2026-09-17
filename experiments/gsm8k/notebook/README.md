# MoTTT GSM8K Standalone Experiment Notebook

This directory contains the standalone Jupyter notebook for executing the complete **MoTTT** (Mixture of Test-Time Trained LoRAs) GSM8K experiment.

## Files

- **`gsm8k_experiment.ipynb`**: The self-contained notebook covering the entire pipeline:
  1. **Environment Setup & Configuration**: Path resolution, device selection (CPU or GPU), hyperparameter specification.
  2. **Distractor Needle Synthesis**: Decomposing problems into factual premise $\mathcal{P}$ and query $\mathcal{Q}$, burying premise needles in non-numerical distractor text across needle depth ratios $\delta \in [0.1, 0.9]$.
  3. **Model Initialization**: Assembling the `MoTTTModel` with the Query-Aware Router MLP, reasoning LoRA experts ($\mathcal{R}=\{1 \dots E\}$), and dynamic scratchpad ($\mathcal{C}=\{0\}$).
  4. **Outer-Loop Training**: Optimizing router and reasoning experts with Asymmetric Load Balancing loss $\mathcal{L}_{\text{bal}}$.
  5. **Training Loss Curves**: Visualizing training convergence (task loss + balancing loss) using `matplotlib`.
  6. **Test-Time Evaluation & Adaptation**: Performing inner-loop gradient steps ($K=256$, $\gamma = 1.0$) on streaming context chunks, executing query-aware routing inference, and extracting answers.
  7. **Multi-Depth Resilience & Router Gate Analysis**: Plotting "Lost-in-the-Middle" resilience (Accuracy vs. $\delta$) and module gate allocation percentages.
  8. **Qualitative Sample Inspection**: Interactive preview of predicted vs. gold answers and gate weights.

## How to Run

### 1. Activate Environment
Ensure you are using the project's virtual environment:
```bash
# If using conda:
conda activate mottt

# If using local .venv:
source .venv/bin/activate
```

### 2. Launch Jupyter
```bash
# From workspace root or notebook folder:
jupyter lab experiments/gsm8k/notebook/gsm8k_experiment.ipynb
# or
jupyter notebook experiments/gsm8k/notebook/gsm8k_experiment.ipynb
```
Select the **Python 3 (ipykernel)** kernel associated with your environment.

### 3. Execution Modes
In **Cell 2 (Configuration)**:
- **Mock / CPU Mode (`USE_MOCK = True`)**:
  - Automatically enabled if CUDA is unavailable.
  - Runs with synthetic embeddings in seconds for immediate verification and interactive experimentation.
- **Full GPU Mode (`USE_MOCK = False`)**:
  - Automatically enabled when CUDA is present (or manually toggle `USE_MOCK = False`).
  - Downloads/uses `Qwen/Qwen2.5-0.5B` and the Hugging Face `openai/gsm8k` dataset for full-scale GPU benchmark reproduction.
