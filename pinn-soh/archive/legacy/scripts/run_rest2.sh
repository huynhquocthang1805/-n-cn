#!/bin/bash
cd /home/claude/pinn-soh; export PYTHONWARNINGS=ignore
( python3 experiments.py E4 --threads 1 > logs/E4.log 2>&1; python3 experiments.py E3 --threads 1 > logs/E3.log 2>&1; echo A_DONE >> logs/ALL_DONE ) &
( while pgrep -f "experiments.py E1" > /dev/null; do sleep 15; done; python3 experiments.py E2 --threads 1 > logs/E2.log 2>&1; echo B_DONE >> logs/ALL_DONE ) &
wait
