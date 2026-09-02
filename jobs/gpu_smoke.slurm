#!/bin/bash
#SBATCH --job-name=gpu_smoke
#SBATCH --partition=normal_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o %j.out
#SBATCH -e %j.err

echo "[smoke] host=$(hostname) start=$(date)"
echo "[smoke] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

PY=/seu_share/apps/anaconda3-2024.10-1/envs/sglang-0.5.10-cuda12.8/bin/python

echo "--- test 1: as-is (expect cudnn abort) ---"
$PY -c "import torch; x=torch.randn(2,1,64,64).cuda(); c=torch.nn.Conv2d(1,8,3).cuda(); print('conv ok:', c(x).shape, torch.cuda.get_device_name(0))" 2>&1 | tail -2

echo "--- test 2: LD_LIBRARY_PATH override to env lib ---"
export LD_LIBRARY_PATH=/seu_share/apps/anaconda3-2024.10-1/envs/sglang-0.5.10-cuda12.8/lib
echo "[smoke] new LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
$PY -c "import torch; x=torch.randn(2,1,64,64).cuda(); c=torch.nn.Conv2d(1,8,3).cuda(); print('conv ok:', c(x).shape, 'dev:', torch.cuda.get_device_name(0)); import torch.nn.functional as F; y=F.conv2d(x,c); print('cudnn conv ok:', y.shape)" 2>&1 | tail -3

echo "--- test 3: check transformers availability ---"
$PY -c "import transformers; print('transformers', transformers.__version__)" 2>&1 | tail -1

echo "[smoke] done $(date)"