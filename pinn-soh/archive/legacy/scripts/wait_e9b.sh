#!/bin/bash
cd /home/claude/pinn-soh; export PYTHONWARNINGS=ignore
while pgrep -f "experiments.py E10" > /dev/null; do sleep 15; done
python3 experiments.py E9 --datasets MIT HUST --threads 1 > logs/E9_b.log 2>&1
echo E9B_DONE >> logs/ALL_DONE
