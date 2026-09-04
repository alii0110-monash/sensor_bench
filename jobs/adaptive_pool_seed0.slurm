#!/bin/bash
#SBATCH --job-name=adaptive_pool
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

# 修 only-* 悖论: adaptive_pool 只平均在场 token（1 模态→/16, 2→/32...）
# 组合: motion_depth（单模态探针 0.474，悖论最尖锐：进融合后 only-depth 0.058）
# 验收: only-depth 从 0.058 大幅上探（向 0.47 靠拢）；full/miss-* 不劣化
# 对照: motion 3-seed (robustness 0.6399, only-depth 0.058, full 0.9159)

export LD_LIBRARY_PATH=/seu_share/apps/anaconda3-2024.10-1/envs/sglang-0.5.10-cuda12.8/lib
export PYTHONUNBUFFERED=1
PY=/seu_share/apps/anaconda3-2024.10-1/envs/sglang-0.5.10-cuda12.8/bin/python

echo "[slurm] job=$SLURM_JOB_ID host=$(hostname) start=$(date)"

$PY -u scripts/train.py \
  --dataset datasets/mmfi/v4 \
  --model token_fusion \
  --motion-depth --adaptive-pool --temporal --mode eager \
  --seed 0 --epochs 30 --batch-size 32 \
  --out-dir checkpoints_motion_ap

$PY -u scripts/run_eval.py \
  --dataset datasets/mmfi/v4 \
  --protocol protocol_v5.json \
  --ckpt-dir checkpoints_motion_ap \
  --model token_fusion \
  --seeds 0 \
  --mode eager \
  --out leaderboard_motion_ap_seed0.json

echo "[slurm] done $(date)"