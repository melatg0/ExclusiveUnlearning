#!/bin/bash
#SBATCH --job-name=eu_score_cache
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/score_cache_%j.log
#SBATCH --error=slurm_logs/score_cache_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Scores cached EU eval responses with ShieldLM only (no base model loaded).
# Writes out/eval/<run>_asr.json next to each *_responses_cache.json.

cd /data/lingo/melatg/ExclusiveUnlearning
source .venv/bin/activate          # or: conda activate <env>
export HF_HOME=/data/scratch/melatg/huggingface

mkdir -p slurm_logs

python -m eu_eval.score_cache \
  --cache_glob 'out/eval/*_harm1_responses_cache.json' \
  --safeunlearning_dir data/safeunlearning \
  --harm_set 1 \
  --shieldlm_model thu-coai/ShieldLM-14B-qwen \
  --shieldlm_batch_size 8 \
  --detector_threshold 5.0
