#!/bin/bash
#SBATCH --job-name=sftmvp_v3
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH -o sftmvp_v3_%j.out
#SBATCH -e sftmvp_v3_%j.err
PYTHON=/seu_share2/home/wangshuai02/220255046/.conda/envs/minimind-o/bin/python
WT=/seu_share2/home/wangshuai02/220255046/sensorbench/.worktrees/sftmvp
cd $WT
$PYTHON -u scripts/train_sft.py --qa-multi --epochs 4 --batch-size 32 --seed 0 \
  --out checkpoints_sftmvp_v3 --log results/sftmvp/train_log_v3.json
echo "== default-question eval (frozen protocol)"
$PYTHON -u scripts/eval_sft.py --ckpt checkpoints_sftmvp_v3 \
  --out results/sftmvp/eval_v3.json --batch-size 16 --collect-gens 100
echo "== yesno probe"
$PYTHON -u scripts/eval_yesno.py --ckpt checkpoints_sftmvp_v3 --n 200
echo "V3 DONE"
