#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/Users/lisepauwels/miniforge3/envs/xcoll/bin/python}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-tune-scan}"
MODE="${1:-all}"

run_py() {
  MPLCONFIGDIR="$MPLCONFIGDIR" "$PYTHON_BIN" "$@"
}

cd "$SCRIPT_DIR"

case "$MODE" in
  maps)
    run_py TuneScan.py
    ;;
  tune-plots)
    run_py PlotTuneMaps.py
    ;;
  md-plots)
    run_py PlotMDIntensity.py
    ;;
  plots)
    run_py PlotTuneMaps.py
    run_py PlotMDIntensity.py
    ;;
  single)
    shift || true
    run_py generate_single_machine_tune_map.py "$@"
    ;;
  all)
    run_py TuneScan.py
    run_py PlotTuneMaps.py
    run_py PlotMDIntensity.py
    ;;
  *)
    echo "Usage: $0 [maps|tune-plots|md-plots|plots|single|all] [single-point args]"
    echo
    echo "Examples:"
    echo "  $0 maps"
    echo "  $0 md-plots"
    echo "  $0 single --case WithoutErrors --qx 20.13 --qy 20.18 --xi-x 0.505 --xi-y 0.3"
    exit 1
    ;;
esac
