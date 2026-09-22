# GSM8K Distractor Experiments: MoTTT vs. Full Fine-Tuning

This directory contains the training, evaluation, and comparison pipelines for the **GSM8K Distractor Benchmark** using **Qwen2.5-0.5B**. 

The benchmark injects needle premises at various depth ratios ($d \in [0.1, 0.9]$) into long distractor backgrounds to measure:
1. **Factual needle retrieval** under distraction.
2. **Multi-step mathematical reasoning**.
3. **Lost-in-the-Middle resilience** (comparing middle depth $d=0.5$ against edges $d=0.1, 0.9$).

---

## Directory Structure

```text
experiments/gsm8k/
├── data/                               # Distractor JSONL datasets
│   ├── train_distractor.jsonl          # Training set (or data/gsm8k_distractor_train/...)
│   └── test_distractor.jsonl           # Test set (or data/gsm8k_distractor_test/...)
├── checkpoints/                        # Saved MoTTT checkpoints
│   ├── mottt_router.pt                 # Query-Aware MLP Router weights
│   ├── reasoning_experts.pt            # Static Reasoning LoRA Expert weights
│   └── training_config.json            # MoTTT hyperparameter config & history
├── checkpoints_full_finetune/          # Saved Full Fine-Tuning (SFT) baseline weights
│   └── sft_training_config.json        # SFT config & training loss history
├── results/                            # JSON evaluation reports & comparisons
│   ├── gsm8k_eval_report.json          # MoTTT test evaluation report
│   ├── baseline_sft_eval_report.json   # Full Fine-Tuning baseline report
│   └── comparison_summary.json         # Side-by-side comparison report
├── data_preparation.py                 # Distractor dataset generator
├── model_train.py                      # MoTTT training pipeline (Router + Reasoning LoRAs)
├── model_test.py                       # MoTTT testing pipeline (Test-time Scratchpad + Routing)
├── train_full_finetune.py              # Full parameter fine-tuning baseline (SFT)
├── test_full_finetune.py               # Full fine-tuned baseline evaluation
├── compare_results.py                  # Side-by-side performance comparison script
└── notebook/                           # Interactive visualization notebook
```

---

## Quick Start: Running the Experiments

Ensure your conda environment is activated:
```bash
conda activate mottt
```

---

### Pipeline 1: MoTTT (Mixture of Test-Time Training)

MoTTT decouples factual episodic ingestion (Dynamic Scratchpad LoRA $E_0$) from multi-step math reasoning (Static Reasoning LoRA Experts $E_1 \dots E_E$) via a Query-Aware Router.

#### 1. Train MoTTT
```bash
python experiments/gsm8k/model_train.py \
    --data_path data/gsm8k_distractor_train/gsm8k_distractor_all.jsonl \
    --base_model_name Qwen/Qwen2.5-0.5B \
    --output_dir experiments/gsm8k/checkpoints \
    --num_experts 4 \
    --rank 16 \
    --alpha 16.0 \
    --lambda_bal 0.01 \
    --batch_size 4 \
    --epochs 1
```

*Key Options:*
* `--num_experts`: Number of reasoning experts $E$ (e.g. `4`, `6`, or `8`). Total routed experts will be $1 + E$.
* `--rank` and `--alpha`: LoRA rank and alpha (scaling $\gamma = \alpha / r$).
* `--lambda_bal`: Weight for the asymmetric load balancing loss.

#### 2. Test MoTTT (Test-Time Adaptation & Evaluation)
```bash
python experiments/gsm8k/model_test.py \
    --test_data data/gsm8k_distractor_test/gsm8k_distractor_all.jsonl \
    --checkpoint_dir experiments/gsm8k/checkpoints \
    --output_dir experiments/gsm8k/results \
    --inner_steps 1 \
    --inner_lr 1e-3 \
    --max_new_tokens 512 \
    --show_outputs
```

*Saves:* `experiments/gsm8k/results/gsm8k_eval_report.json`

---

### Pipeline 2: Full Fine-Tuning Baseline (SFT)

A standard Supervised Fine-Tuning (SFT) baseline where all parameters of `Qwen/Qwen2.5-0.5B` are unfrozen and trained directly on context + question inputs with target math solution labels.

#### 1. Train Full Fine-Tuning Baseline
```bash
python experiments/gsm8k/train_full_finetune.py \
    --data_path data/gsm8k_distractor_train/gsm8k_distractor_all.jsonl \
    --base_model_name Qwen/Qwen2.5-0.5B \
    --output_dir experiments/gsm8k/checkpoints_full_finetune \
    --batch_size 4 \
    --grad_accum_steps 2 \
    --lr 2e-5 \
    --epochs 1
```

*Key Options:*
* `--grad_accum_steps`: Accumulate gradients to fit in GPU VRAM (effective batch size = `batch_size * grad_accum_steps`).
* `--gradient_checkpointing`: Pass flag to reduce VRAM usage if encountering Out-Of-Memory.
* `--lr`: Learning rate for full model tuning (default: `2e-5`).

#### 2. Test Full Fine-Tuning Baseline
```bash
python experiments/gsm8k/test_full_finetune.py \
    --test_data data/gsm8k_distractor_test/gsm8k_distractor_all.jsonl \
    --checkpoint_dir experiments/gsm8k/checkpoints_full_finetune \
    --output_dir experiments/gsm8k/results \
    --output_file baseline_sft_eval_report.json \
    --max_new_tokens 512 \
    --show_outputs
```

*Saves:* `experiments/gsm8k/results/baseline_sft_eval_report.json`

---

### Pipeline 3: Compare MoTTT vs. Full Fine-Tuning Baseline

Compare the results side-by-side to assess overall accuracy and Lost-in-the-Middle resilience:

```bash
python experiments/gsm8k/compare_results.py \
    --mottt_report experiments/gsm8k/results/gsm8k_eval_report.json \
    --sft_report experiments/gsm8k/results/baseline_sft_eval_report.json \
    --output_summary experiments/gsm8k/results/comparison_summary.json
```

**Expected Terminal Output:**
```text
==============================================================================
MoTTT vs. Full Fine-Tuning (SFT) Baseline Performance Comparison
==============================================================================

--- 1. OVERALL ACCURACY ---
MoTTT Overall Accuracy:            68.50%
Full Fine-Tuning (SFT) Accuracy:   54.20%
Difference (MoTTT - SFT):         +14.30%

--- 2. NEEDLE DEPTH BREAKDOWN (Lost-in-the-Middle Analysis) ---
Needle Depth   | MoTTT Acc    | Full SFT Acc   | Delta (MoTTT - SFT) 
----------------------------------------------------------------------
0.10           | 72.00%       | 66.00%         | +6.00%              
0.30           | 68.00%       | 54.00%         | +14.00%             
0.50 (Middle)  | 66.00%       | 42.00%         | +24.00%             
0.70           | 67.00%       | 51.00%         | +16.00%             
0.90           | 70.00%       | 64.00%         | +6.00%              
----------------------------------------------------------------------
```

---

## Offline / Mock Mode (For Local Testing Without GPU)

To dry-run all scripts locally without downloading weights or using a GPU:

```bash
# MoTTT Mock Cycle
python experiments/gsm8k/model_train.py --mock --epochs 1 --batch_size 2
python experiments/gsm8k/model_test.py --mock

# Full Fine-Tuning Mock Cycle
python experiments/gsm8k/train_full_finetune.py --mock --epochs 1 --batch_size 2
python experiments/gsm8k/test_full_finetune.py --mock

# Comparison
python experiments/gsm8k/compare_results.py
```
