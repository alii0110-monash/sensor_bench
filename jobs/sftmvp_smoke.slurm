#!/bin/bash
#SBATCH --job-name=sftmvp_smoke
#SBATCH --partition=normal_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH -o sftmvp_smoke_%j.out
#SBATCH -e sftmvp_smoke_%j.err

PYTHON=/seu_share2/home/wangshuai02/220255046/.conda/envs/minimind-o/bin/python
WT=/seu_share2/home/wangshuai02/220255046/sensorbench/.worktrees/sftmvp
cd $WT
$PYTHON -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')"
$PYTHON -u scripts/train_sft.py --max-train 100 --epochs 2 --batch-size 16 \
  --out checkpoints_sftmvp_smoke --log results/sftmvp/smoke_train_log.json
$PYTHON -u scripts/eval_sft.py --ckpt checkpoints_sftmvp_smoke \
  --out results/sftmvp/eval_smoke.json --limit 50 --batch-size 16 --collect-gens 50
echo "SMOKE DONE"
