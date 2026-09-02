#!/bin/bash
# Sequential v2 training + eval driver (RAM-safe: one job at a time)
cd /home/li/projects/sensorbench
PY=/home/li/projects/holollm/.venv/bin/python
for m in token_fusion late_fusion; do
  for s in 0 1 2; do
    echo "=== training $m seed $s (v2) ==="
    $PY -u scripts/train.py --dataset datasets/mmfi/v2 --model $m --seed $s --epochs 30 --out-dir checkpoints_v2 > logs/train_v2_${m}_${s}.log 2>&1
  done
done
echo "=== all v2 training done, running eval ==="
$PY scripts/run_eval.py --dataset datasets/mmfi/v2 --protocol protocol.json \
  --ckpt-dir checkpoints_v2 --seeds 0,1,2 --out leaderboard_v2.json > logs/eval_v2.log 2>&1
echo "=== v2 eval done ==="
