#!/usr/bin/env bash
# Laptop LoRA path: tiny synthetic LeRobot dataset + FastFlow, optional smolvla_base.
set -euo pipefail
cd "$(dirname "$0")/.."
python src/train_smolvla_lora.py --record --steps "${STEPS:-400}" --smolvla-steps "${SMOLVLA_STEPS:-50}"
