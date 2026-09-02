#!/bin/bash
# Sequential v3 training + eval driver (RAM-safe: one job at a time)
cd /home/li/projects/sensorbench
PY=/home/li/projects/holollm/.venv/bin/python
for m in token_fusion late_fusion; do
  for s in 0 1 2; do
    echo "=== training $m seed $s (v3) ==="
    $PY -u scripts/train.py --dataset datasets/mmfi/v3 --model $m --seed $s --epochs 30 --out-dir checkpoints_v3 > logs/train_v3_${m}_${s}.log 2>&1
  done
done
echo "=== all v3 training done, running eval ==="
$PY -u scripts/run_eval.py --dataset datasets/mmfi/v3 --protocol protocol_v3.json \
  --ckpt-dir checkpoints_v3 --seeds 0,1,2 --out leaderboard_v3.json > logs/eval_v3.log 2>&1
echo "=== v3 eval done ==="
