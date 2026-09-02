#!/bin/bash
#SBATCH --job-name=layer_cka_v4
#SBATCH --partition=normal_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --comment=layer_cka_v4

set -e
cd /seu_share2/home/wangshuai02/220255046/sensorbench

# normal_test CPU 队列（30 min 自动 kill）— val=1870 样本 × 3 seeds，CPU 估时 5-10 分钟。
# 选用 sglang-0.5.10-cuda12.8 env：torch 2.9.1 与训练一致。
PY=/seu_share/apps/anaconda3-2024.10-1/envs/sglang-0.5.10-cuda12.8/bin/python

echo "[slurm] job=$SLURM_JOB_ID host=$(hostname) start=$(date)"
echo "[slurm] python: $($PY -c 'import sys,torch,numpy,matplotlib; print(sys.executable, torch.__version__, numpy.__version__, matplotlib.__version__)')"

export PYTHONUNBUFFERED=1
$PY -u scripts/run_layer_cka.py \
    --checkpoint_dir checkpoints_v4_temporal \
    --dataset_root datasets/mmfi/v4 \
    --output_dir results \
    --seeds 0 1 2 \
    --batch_size 64 \
    --device cpu

echo "[slurm] done $(date)"