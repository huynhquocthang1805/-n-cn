#!/bin/bash
# wait for E1 processes to finish, then run E4, E3 (proc A) and E2 (proc B)
cd /home/claude/pinn-soh
export PYTHONWARNINGS=ignore
while pgrep -f "experiments.py E1" > /dev/null; do sleep 20; done
( python3 experiments.py E4 --threads 1 > logs/E4.log 2>&1; python3 experiments.py E3 --threads 1 > logs/E3.log 2>&1 ) &
( python3 experiments.py E2 --threads 1 > logs/E2.log 2>&1 ) &
wait
echo ALL_DONE > logs/ALL_DONE
