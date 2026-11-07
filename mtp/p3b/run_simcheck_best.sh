#!/bin/bash
# waits for the original-weights recheck (PID given) to exit, then runs the best-weights recheck
B=/u701/j692wu/swift-opt/mtp/p3b
P=/u701/j692wu/.conda/envs/inference/bin/python
while kill -0 "$1" 2>/dev/null; do sleep 20; done
cd "$B"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $P -I "$B/sim_check_ckpt.py" "$B/ckpt/best.safetensors" "$B/verify/simcheck_best.json" "$B/eval/preds_s0996.npz" > "$B/logs/simcheck_best.out" 2>&1
echo done > "$B/logs/simcheck_best.done"
