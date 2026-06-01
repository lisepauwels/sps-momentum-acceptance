#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANUAL_SWEEP_SCRIPT="${SCRIPT_DIR}/run_manual_tune_sweep.py"

COMMON_ARGS=(
  --sextupoles-mode on
  --xi-x 0.5
  --xi-y 0.5
  --schedule-points 41
  --num-turns 6000
  --num-particles 100
  --progress-every 100
)

python "${MANUAL_SWEEP_SCRIPT}" \
  "${COMMON_ARGS[@]}" \
  --dq-per-turn-x=-2.25e-5 \
  --dq-per-turn-y=0 \
  --error-variant none \
  --batch-name sext_on_qx_xi05

python "${MANUAL_SWEEP_SCRIPT}" \
  "${COMMON_ARGS[@]}" \
  --dq-per-turn-x=-2.25e-5 \
  --dq-per-turn-y=0 \
  --error-variant all \
  --batch-name sext_on_qx_xi05_errors_all

python "${MANUAL_SWEEP_SCRIPT}" \
  "${COMMON_ARGS[@]}" \
  --dq-per-turn-x=0 \
  --dq-per-turn-y=-3.1e-5 \
  --error-variant all \
  --batch-name sext_on_qy_xi05_errors_all
