#!/bin/bash
cd /home/claude/pinn-soh; export PYTHONWARNINGS=ignore
python3 experiments.py E6 --datasets XJTU TJU --threads 1 > logs/E6_a.log 2>&1
echo E6A_DONE >> logs/ALL_DONE
