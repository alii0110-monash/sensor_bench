#!/bin/bash
#SBATCH --job-name=cka_ctrl
#SBATCH --partition=normal_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o %j.out
#SBATCH -e %j.err

set -e
cd /seu_share2/home/wangshuai02/220255046/sensorbench

PY=/seu_share/apps/anaconda3-2024.10-1/envs/sglang-0.5.10-cuda12.8/bin/python
export PYTHONUNBUFFERED=1

echo "[slurm] job=$SLURM_JOB_ID host=$(hostname) start=$(date)"
$PY -u scripts/layer_cka_controls.py
echo "[slurm] done $(date)"