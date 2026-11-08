#!/usr/bin/env bash
cd /u701/j692wu/swift-opt/mtp/p3
PY=/u701/j692wu/.conda/envs/inference/bin/python
rm -f logs/chain1.done
$PY -I verify_ttt.py bf16 > logs/verify_ttt_bf16.log 2>&1
$PY -I simulate.py bf16 > logs/simulate_bf16.log 2>&1
$PY -I simulate.py fp32router > logs/simulate_fp32router.log 2>&1
$PY -I sim_check.py > logs/sim_check_bf16.log 2>&1
echo CHAIN_DONE > logs/chain1.done
