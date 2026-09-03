#!/bin/bash
#SBATCH --job-name=motion_main
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=18:00:00

set -e
cd /seu_share2/home/wangshuai02/220255046/sensorbench

# 对照基线: leaderboard_temporal_full.json (token_fusion v4 raw temporal, 0.6904/0.9451)
# 变更: --motion-depth (DepthEncoder -> ViTMotionEncoder 2ch [d_t, Δ_t])

export LD_LIBRARY_PATH=/seu_share/apps/anaconda3-2024.10-1/envs/sglang-0.5.10-cuda12.8/lib
export PYTHONUNBUFFERED=1
PY=/seu_share/apps/anaconda3-2024.10-1/envs/sglang-0.5.10-cuda12.8/bin/python

echo "[slurm] job=$SLURM_JOB_ID host=$(hostname) start=$(date)"
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader || true

# ---- smoke: motion 路径正确性（stack/predict_batch/save-load）----
$PY -u - <<'EOF'
import sys; sys.path.insert(0, '.')
import numpy as np, torch
from framework.models.token_fusion import TokenFusionModel
m = TokenFusionModel(num_classes=27, temporal=True, motion_depth=True)
T = 5
class FS:
    def __init__(self, i):
        self.label = i % 27; self.id = f's{i}'
        shapes = {'wifi': (T,3,114,10), 'depth': (T,1,224,224), 'lidar': (T,1536,3),
                  'mmwave': (T,64,5), 'rgb': (T,17,2)}
        self.modalities = {k: type('M',(),{'data': np.random.randn(*shapes[k]).astype(np.float32)})() for k in shapes}
fs = [FS(i) for i in range(4)]
cfg = type('C',(),{'device':'cuda','time_mask_p':0.0,'batch_size':4})()
mods = m._stack_mods(fs, {k: True for k in fs[0].modalities}, cfg)
assert mods['depth'].shape == (4, T, 2, 224, 224), mods['depth'].shape
m = m.cuda()
print('[smoke] predict_batch:', tuple(m.predict_batch(fs, list(fs[0].modalities.keys())).shape))
m.save('/tmp/motion_smoke.pt')
m2 = TokenFusionModel.load('/tmp/motion_smoke.pt')
assert m2.motion_depth and m2.temporal
print('[smoke] OK')
EOF

# ---- train 3 seeds (sequential, same GPU) ----
for SEED in 0 1 2; do
  echo "[train] seed $SEED start $(date)"
  $PY -u scripts/train.py \
    --dataset datasets/mmfi/v4 \
    --model token_fusion \
    --motion-depth --temporal --mode eager \
    --seed $SEED --epochs 30 --batch-size 32 \
    --out-dir checkpoints_motion_v4
  echo "[train] seed $SEED done $(date)"
done

# ---- leaderboard eval (protocol_v5, 21 profiles x 3 seeds) ----
$PY -u scripts/run_eval.py \
  --dataset datasets/mmfi/v4 \
  --protocol protocol_v5.json \
  --ckpt-dir checkpoints_motion_v4 \
  --model token_fusion \
  --seeds 0,1,2 \
  --mode eager \
  --out leaderboard_motion_v4.json

echo "[slurm] done $(date)"