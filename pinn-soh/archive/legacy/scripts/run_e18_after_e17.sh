#!/usr/bin/env bash
# Chạy E18 (ma trận chuyển miền đầy đủ) sau khi E17 xong.
cd "$(dirname "$0")/.."
until grep -q "^xong" logs/E17.log; do sleep 60; done
OMP_NUM_THREADS=1 python3 -W ignore experiments.py E18 --workers 4 --seeds 0 1 2 3 4 > logs/E18.log 2>&1
