#!/usr/bin/env bash
# Chạy sau E13: E14 (cùng normaliser) rồi huấn luyện 4 checkpoint demo.
set -e
cd "$(dirname "$0")/.."
until grep -q "^xong" logs/E13.log; do sleep 30; done
OMP_NUM_THREADS=1 python3 -W ignore experiments.py E14 --workers 4 --seeds 0 1 2 3 4 > logs/E14.log 2>&1
for ds in XJTU TJU MIT HUST; do
  OMP_NUM_THREADS=1 python3 -W ignore demo.py train --dataset $ds > logs/demo_$ds.log 2>&1 &
done
wait
echo "E14+demo xong" >> logs/E14.log
