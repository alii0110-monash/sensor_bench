#!/bin/bash
#SBATCH --job-name=sftmvp_train
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH -o sftmvp_train_%j.out
#SBATCH -e sftmvp_train_%j.err

PYTHON=/seu_share2/home/wangshuai02/220255046/.conda/envs/minimind-o/bin/python
WT=/seu_share2/home/wangshuai02/220255046/sensorbench/.worktrees/sftmvp
cd $WT
$PYTHON -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')"
$PYTHON -u scripts/train_sft.py --epochs 4 --batch-size 32 --seed 0 \
  --out checkpoints_sftmvp --log results/sftmvp/train_log.json
$PYTHON -u scripts/eval_sft.py --ckpt checkpoints_sftmvp \
  --out results/sftmvp/eval_mvp.json --batch-size 16 --collect-gens 200
echo "SFTMVP DONE"
