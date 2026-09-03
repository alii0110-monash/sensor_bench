#!/bin/bash
#SBATCH --job-name=motion_ln
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=08:00:00

set -e
cd /seu_share2/home/wangshuai02/220255046/sensorbench

# 假设: motion 版融合劣化源于 ViT depth token 统计与其他模态不匹配
# 干预: encoder 输出加 LayerNorm 对齐 → 1 seed 快速验证
# 对照: motion seed0 (best val 0.820) / temporal 基线 (robustness 0.6904)

export LD_LIBRARY_PATH=/seu_share/apps/anaconda3-2024.10-1/envs/sglang-0.5.10-cuda12.8/lib
export PYTHONUNBUFFERED=1
PY=/seu_share/apps/anaconda3-2024.10-1/envs/sglang-0.5.10-cuda12.8/bin/python

echo "[slurm] job=$SLURM_JOB_ID host=$(hostname) start=$(date)"
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader || true

$PY -u scripts/train.py \
  --dataset datasets/mmfi/v4 \
  --model token_fusion \
  --motion-depth --motion-depth-layernorm --temporal --mode eager \
  --seed 0 --epochs 30 --batch-size 32 \
  --out-dir checkpoints_motion_ln

$PY -u scripts/run_eval.py \
  --dataset datasets/mmfi/v4 \
  --protocol protocol_v5.json \
  --ckpt-dir checkpoints_motion_ln \
  --model token_fusion \
  --seeds 0 \
  --mode eager \
  --out leaderboard_motion_ln_seed0.json

echo "[slurm] done $(date)"