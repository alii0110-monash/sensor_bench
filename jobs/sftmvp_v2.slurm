#!/bin/bash
#SBATCH --job-name=sftmvp_v2
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH -o sftmvp_v2_%j.out
#SBATCH -e sftmvp_v2_%j.err

PYTHON=/seu_share2/home/wangshuai02/220255046/.conda/envs/minimind-o/bin/python
WT=/seu_share2/home/wangshuai02/220255046/sensorbench/.worktrees/sftmvp
cd $WT
$PYTHON -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "== step1: re-eval v1 ckpt with stem matcher (fair re-baseline)"
$PYTHON -u scripts/eval_sft.py --ckpt checkpoints_sftmvp \
  --out results/sftmvp/eval_v1_rebase.json --batch-size 16 --collect-gens 0
echo "== step2: train v2 (variants + 8 epochs)"
$PYTHON -u scripts/train_sft.py --variants-jsonl results/captions_route_c2_train.jsonl \
  --epochs 8 --batch-size 32 --seed 0 \
  --out checkpoints_sftmvp_v2 --log results/sftmvp/train_log_v2.json
echo "== step3: eval v2"
$PYTHON -u scripts/eval_sft.py --ckpt checkpoints_sftmvp_v2 \
  --out results/sftmvp/eval_v2.json --batch-size 16 --collect-gens 200
echo "SFTMVP-V2 DONE"
