#!/bin/bash
# Sequential training driver (each job loads the full 18GB dataset; RAM-safe one at a time)
cd /home/li/projects/sensorbench
PY=/home/li/projects/holollm/.venv/bin/python
for m in token_fusion late_fusion; do
  for s in 0 1 2; do
    echo "=== training $m seed $s ==="
    $PY scripts/train.py --dataset datasets/mmfi/v1 --model $m --seed $s --epochs 30 --out-dir checkpoints > logs/train_${m}_${s}.log 2>&1
  done
done
echo "=== all training done, running eval ==="
$PY scripts/run_eval.py --dataset datasets/mmfi/v1 --protocol protocol.json \
  --ckpt-dir checkpoints --seeds 0,1,2 --out leaderboard_v1.json > logs/eval_v1.log 2>&1
echo "=== eval done ==="
