#!/usr/bin/env bash
cd /u701/j692wu/swift-opt/mtp/p3
while [ ! -f logs/chain1.done ]; do sleep 20; done
/u701/j692wu/.conda/envs/inference/bin/python -I sim_check.py > logs/sim_check_bf16.log 2>&1
echo SIMCHECK_DONE > logs/simcheck.done
