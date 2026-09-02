#!/bin/bash
# Sequential v4 training + eval driver (RAM-safe: one job at a time)
cd /home/li/projects/sensorbench
PY=/home/li/projects/holollm/.venv/bin/python
for m in token_fusion late_fusion; do
  for s in 0 1 2; do
    echo "=== training $m seed $s (v4) ==="
    $PY -u scripts/train.py --dataset datasets/mmfi/v4 --model $m --seed $s --epochs 30 --out-dir checkpoints_v4 > logs/train_v4_${m}_${s}.log 2>&1
  done
done
echo "=== all v4 training done, running eval ==="
$PY -u scripts/run_eval.py --dataset datasets/mmfi/v4 --protocol protocol_v3.json \
  --ckpt-dir checkpoints_v4 --seeds 0,1,2 --out leaderboard_v4.json > logs/eval_v4.log 2>&1
echo "=== v4 eval done ==="
