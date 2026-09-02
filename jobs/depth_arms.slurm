#!/bin/bash
#SBATCH --job-name=depth_arms
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=04:00:00

set -e
cd /seu_share2/home/wangshuai02/220255046/sensorbench

export LD_LIBRARY_PATH=/seu_share/apps/anaconda3-2024.10-1/envs/sglang-0.5.10-cuda12.8/lib
export PYTHONUNBUFFERED=1
PY=/seu_share/apps/anaconda3-2024.10-1/envs/sglang-0.5.10-cuda12.8/bin/python

echo "[slurm] job=$SLURM_JOB_ID host=$(hostname) start=$(date)"
nvidia-smi --query-gpu=name,memory.used --format=csv 2>/dev/null || true
$PY -u scripts/depth_arms_experiment.py
echo "[slurm] done $(date)"