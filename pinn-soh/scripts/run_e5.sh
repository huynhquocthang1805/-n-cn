#!/bin/bash
cd /home/claude/pinn-soh; export PYTHONWARNINGS=ignore
while [ "$(pgrep -f 'experiments.py E3' | wc -l)" -ge 2 ]; do sleep 15; done
python3 experiments.py E5 --threads 1 > logs/E5.log 2>&1
echo E5_DONE >> logs/ALL_DONE
