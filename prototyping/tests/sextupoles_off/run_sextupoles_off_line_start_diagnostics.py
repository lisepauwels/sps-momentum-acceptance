from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the sextupoles-off diagnostics workflow at the beginning of the line. "
            "This uses RF-style one-turn summaries without cycling, so all means/std/tune "
            "plots are evaluated at the original line start."
        )
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--line-path", default=None)
    parser.add_argument("--qx", type=float, default=20.13)
    parser.add_argument("--qy", type=float, default=20.18)
    parser.add_argument("--xi-x", type=float, default=0.5)
    parser.add_argument("--xi-y", type=float, default=0.5)
    parser.add_argument("--error-variant", default="none")
    parser.add_argument("--planes", nargs="+", choices=("DPpos", "DPneg"), default=["DPpos", "DPneg"])
    parser.add_argument("--num-particles", type=int, default=100)
    parser.add_argument("--num-turns", type=int, default=6000)
    parser.add_argument("--total-sweep-hz", type=float, default=3000.0)
    parser.add_argument("--nemitt-x", type=float, default=2e-6)
    parser.add_argument("--nemitt-y", type=float, default=2e-6)
    parser.add_argument("--sigma-z", type=float, default=0.224)
    parser.add_argument("--naff-harmonics", type=int, default=5)
    parser.add_argument("--fft-window", type=int, default=256)
    parser.add_argument("--fft-step", type=int, default=64)
    parser.add_argument("--omp-threads", default="0")
    parser.add_argument("--output-base", default="sextupoles_off_monitor_outputs")
    parser.add_argument(
        "--batch-name",
        default=f"sextupoles_off_line_start_diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--reuse-tune-map",
        action="store_true",
        help="Reuse the shared sextupoles-off tune map if it exists.",
    )
    parser.add_argument(
        "--force-rebuild-tune-map",
        action="store_true",
        help="Rebuild the shared sextupoles-off tune map before running.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = Path(__file__).with_name("run_sextupoles_off_monitor_scan.py")

    cmd = [
        args.python,
        str(runner),
        "--observation-mode",
        "rf_style",
        "--no-cycle",
        "--planes",
        *args.planes,
        "--num-particles",
        str(args.num_particles),
        "--num-turns",
        str(args.num_turns),
        "--total-sweep-hz",
        str(args.total_sweep_hz),
        "--nemitt-x",
        str(args.nemitt_x),
        "--nemitt-y",
        str(args.nemitt_y),
        "--sigma-z",
        str(args.sigma_z),
        "--qx",
        str(args.qx),
        "--qy",
        str(args.qy),
        "--xi-x",
        str(args.xi_x),
        "--xi-y",
        str(args.xi_y),
        "--error-variant",
        str(args.error_variant),
        "--naff-harmonics",
        str(args.naff_harmonics),
        "--fft-window",
        str(args.fft_window),
        "--fft-step",
        str(args.fft_step),
        "--omp-threads",
        str(args.omp_threads),
        "--output-base",
        str(args.output_base),
        "--batch-name",
        str(args.batch_name),
    ]
    if args.line_path is not None:
        cmd.extend(["--line-path", args.line_path])
    if args.reuse_tune_map:
        cmd.append("--reuse-tune-map")
    if args.force_rebuild_tune_map:
        cmd.append("--force-rebuild-tune-map")

    print("[line_start_diag] Launching:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
