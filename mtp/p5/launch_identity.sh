#!/usr/bin/env bash
cd /u701/j692wu/swift-opt
export GPU_FREE_MIB=2000 PRESET=final-code SWIFT_FP8=1 SWIFT_DRAFT_HEAD_FP8=1 SWIFT_FP8_TARGETS=qkvz,gdn_out,qsa_qkv,qsa_o,lm_head,mtp SWIFT_FP8_PLAN=/u701/j692wu/swift-opt/release/plan/fp8_plan_row.json SWIFT_CHECKPOINT_PATH=/u701/j692wu/swift-opt/mtp/view-identity
export EXTRA_CMD='/usr/bin/python3 /u701/j692wu/swift-profile/score/score_logprobs.py /u701/j692wu/swift-opt/runs/p5-identity/scores.json'
exec bash run_config.sh p5-identity 8
