#!/bin/bash
#SBATCH --job-name=ap_seeds12
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=12:00:00

set -e
cd /seu_share2/home/wangshuai02/220255046/sensorbench

# 3-seed 确认: motion+AP (seed0 已有 0.7068 robustness)，补 seeds 1,2 + 3-seed eval

export LD_LIBRARY_PATH=/seu_share/apps/anaconda3-2024.10-1/envs/sglang-0.5.10-cuda12.8/lib
export PYTHONUNBUFFERED=1
PY=/seu_share/apps/anaconda3-2024.10-1/envs/sglang-0.5.10-cuda12.8/bin/python

echo "[slurm] job=$SLURM_JOB_ID host=$(hostname) start=$(date)"

for SEED in 1 2; do
  echo "[train] seed $SEED start $(date)"
  $PY -u scripts/train.py \
    --dataset datasets/mmfi/v4 \
    --model token_fusion \
    --motion-depth --adaptive-pool --temporal --mode eager \
    --seed $SEED --epochs 30 --batch-size 32 \
    --out-dir checkpoints_motion_ap
  echo "[train] seed $SEED done $(date)"
done

$PY -u scripts/run_eval.py \
  --dataset datasets/mmfi/v4 \
  --protocol protocol_v5.json \
  --ckpt-dir checkpoints_motion_ap \
  --model token_fusion \
  --seeds 0,1,2 \
  --mode eager \
  --out leaderboard_motion_ap_3seed.json

echo "[slurm] done $(date)"