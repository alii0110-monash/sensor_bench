#!/bin/bash
#SBATCH --job-name=ap_extreme
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

# 极端缺失过采样: extreme_missing_p=0.3 (30% 批次强制 k∈{1,2} 模态)
# 假设: only-depth 残留 (0.059) 与 miss2-mmwave-rgb 极差场景源于训练分布罕见
# 对照: motion+AP seed0 (robustness 0.7068, only-depth 0.059, miss2-mmwave-rgb 0.174)

export LD_LIBRARY_PATH=/seu_share/apps/anaconda3-2024.10-1/envs/sglang-0.5.10-cuda12.8/lib
export PYTHONUNBUFFERED=1
PY=/seu_share/apps/anaconda3-2024.10-1/envs/sglang-0.5.10-cuda12.8/bin/python

echo "[slurm] job=$SLURM_JOB_ID host=$(hostname) start=$(date)"

$PY -u scripts/train.py \
  --dataset datasets/mmfi/v4 \
  --model token_fusion \
  --motion-depth --adaptive-pool --extreme-missing-p 0.3 --temporal --mode eager \
  --seed 0 --epochs 30 --batch-size 32 \
  --out-dir checkpoints_motion_ap_ext

$PY -u scripts/run_eval.py \
  --dataset datasets/mmfi/v4 \
  --protocol protocol_v5.json \
  --ckpt-dir checkpoints_motion_ap_ext \
  --model token_fusion \
  --seeds 0 \
  --mode eager \
  --out leaderboard_motion_ap_ext_seed0.json

echo "[slurm] done $(date)"