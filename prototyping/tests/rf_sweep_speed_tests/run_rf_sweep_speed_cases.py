from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_SPEEDS = {
    "slow": 0.5,
    "nominal": 1.0,
    "fast": 2.4551111909641804e-05,
}

DEFAULT_PLANES = ("DPpos", "DPneg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch several RF-sweep speed cases and compare outputs."
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num-particles", type=int, default=100)
    parser.add_argument("--num-turns", type=int, default=6000)
    parser.add_argument("--snapshot-every", type=int, default=500)
    parser.add_argument("--fft-window", type=int, default=128)
    parser.add_argument("--fft-step", type=int, default=32)
    parser.add_argument("--line-path", default=None)
    parser.add_argument("--qx", type=float, default=20.13)
    parser.add_argument("--qy", type=float, default=20.18)
    parser.add_argument("--xi-x", type=float, default=0.5)
    parser.add_argument("--xi-y", type=float, default=0.5)
    parser.add_argument(
        "--error-variant",
        choices=[
            "all",
            "dipole_b3",
            "dipole_b3_quadrupole_b4",
            "dipole_b3b5",
            "dipole_b5",
            "none",
            "quadrupole_b4",
            "quadrupole_b4b6",
            "quadrupole_b6",
        ],
        default="none",
    )
    parser.add_argument("--tune-map-case", choices=["WithErrors", "WithoutErrors", "Simplified"], default=None)
    parser.add_argument("--output-base", default="rf_sweep_speed_outputs")
    parser.add_argument("--batch-name", default="three_speed_scan")
    return parser.parse_args()


def run_case(args: argparse.Namespace, plane: str, speed_label: str, sweep_per_turn_hz: float) -> str:
    case_name = f"{args.batch_name}_{plane}_{speed_label}"
    print(
        "[run_rf_sweep_speed_cases] Starting case: "
        f"{case_name} (plane={plane}, sweep_per_turn={sweep_per_turn_hz} Hz/turn)"
    )
    cmd = [
        args.python,
        str(Path(__file__).with_name("rf_sweep_speed_scan.py")),
        "--output-base",
        str(Path(args.output_base) / args.batch_name),
        "--case-name",
        case_name,
        "--plane",
        plane,
        "--sweep-per-turn-hz",
        str(sweep_per_turn_hz),
        "--num-turns",
        str(args.num_turns),
        "--num-particles",
        str(args.num_particles),
        "--snapshot-every",
        str(args.snapshot_every),
        "--fft-window",
        str(args.fft_window),
        "--fft-step",
        str(args.fft_step),
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
    ]
    if args.tune_map_case is not None:
        cmd.extend(["--tune-map-case", args.tune_map_case])
    if args.line_path is not None:
        cmd.extend(["--line-path", args.line_path])
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parent.parent)
    print(f"[run_rf_sweep_speed_cases] Finished case: {case_name}")
    return case_name


def load_case_table(case_dir: Path, filename: str) -> pd.DataFrame:
    return pd.read_parquet(case_dir / filename)


def plot_loss_comparison(batch_dir: Path, case_dirs: dict[str, Path]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    plane_axes = dict(zip(DEFAULT_PLANES, axes))

    for plane in DEFAULT_PLANES:
        ax = plane_axes[plane]
        for speed_label in DEFAULT_SPEEDS:
            case_key = f"{plane}_{speed_label}"
            loss = load_case_table(case_dirs[case_key], "intensity_loss.parquet")
            ax.plot(loss["delta"] * 1e3, loss["surviving_fraction"], label=speed_label)
        ax.set_title(plane)
        ax.set_xlabel(r"$\delta$ [$10^{-3}$]")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Normalised intensity")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(batch_dir / "comparison_intensity_loss.png", dpi=200)
    plt.close(fig)


def plot_centroid_comparison(batch_dir: Path, case_dirs: dict[str, Path], coord: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    plane_axes = dict(zip(DEFAULT_PLANES, axes))

    for plane in DEFAULT_PLANES:
        ax = plane_axes[plane]
        for speed_label in DEFAULT_SPEEDS:
            case_key = f"{plane}_{speed_label}"
            summary = load_case_table(case_dirs[case_key], "turn_summary.parquet")
            ax.plot(summary["turn"], summary[f"{coord}_mean"], label=speed_label)
        ax.set_title(f"{plane} {coord}_mean")
        ax.set_xlabel("Turn")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(coord)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(batch_dir / f"comparison_{coord}_mean.png", dpi=200)
    plt.close(fig)


def write_batch_index(batch_dir: Path, case_dirs: dict[str, Path]) -> None:
    payload = {
        "batch_dir": str(batch_dir),
        "cases": {key: str(path) for key, path in case_dirs.items()},
        "sweep_per_turn_hz": DEFAULT_SPEEDS,
    }
    with (batch_dir / "batch_index.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main() -> None:
    args = parse_args()
    repo_tests_dir = Path(__file__).resolve().parent
    batch_dir = repo_tests_dir / args.output_base / args.batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run_rf_sweep_speed_cases] Batch directory: {batch_dir}")
    print(
        "[run_rf_sweep_speed_cases] Config: "
        f"particles={args.num_particles}, turns={args.num_turns}, "
        f"speeds={DEFAULT_SPEEDS}, planes={DEFAULT_PLANES}, "
        f"error_variant={args.error_variant}, tune_map_case={args.tune_map_case}"
    )

    case_dirs: dict[str, Path] = {}
    for plane in DEFAULT_PLANES:
        for speed_label, sweep_per_turn_hz in DEFAULT_SPEEDS.items():
            case_name = run_case(args, plane=plane, speed_label=speed_label, sweep_per_turn_hz=sweep_per_turn_hz)
            case_dirs[f"{plane}_{speed_label}"] = batch_dir / case_name

    print("[run_rf_sweep_speed_cases] All cases finished, building comparison plots")
    write_batch_index(batch_dir, case_dirs)
    plot_loss_comparison(batch_dir, case_dirs)
    plot_centroid_comparison(batch_dir, case_dirs, coord="x")
    plot_centroid_comparison(batch_dir, case_dirs, coord="y")
    print("[run_rf_sweep_speed_cases] Done")


if __name__ == "__main__":
    main()
