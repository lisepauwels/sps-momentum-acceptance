"""
plot_intensity_loss_simulations.py
===================================
Regenerate intensity-loss-vs-delta plots from IntensityScan2 simulation data.

Saves PDFs to the studies/intensity_scan2/ directory alongside this script.

Usage
-----
    python plot_intensity_loss_simulations.py                   # without-errors only
    python plot_intensity_loss_simulations.py --line-types errors       # errors only
    python plot_intensity_loss_simulations.py --line-types linear errors # both
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "helper_functions" / "intensity_helpers"))

from intensity_loss import plot_intensity_drop
from midpoints_analysis import df_to_delta

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "helper_functions"))
from load_paths import get_path as _get_path
DEFAULT_STUDY_ROOT = (
    _get_path("sps_simulations_data_root",
               default=str(Path.home() / "phd" / "data" / "sps-simulations"))
    / "momentum-acceptance" / "intensity_scan2"
)
SWEEP_PER_TURN = 1.0
NUM_PARTICLES = 2000 * 500
OUTPUT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate IntensityScan2 intensity-loss plots.")
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument(
        "--line-types", nargs="+", default=["linear"], choices=["linear", "errors"],
        help="Model variants to plot. Default: linear (no errors).",
    )
    parser.add_argument("--fontsize", type=int, default=16)
    return parser.parse_args()


def load_normalised_intensity(study_results_dir: Path) -> dict:
    data: dict[str, dict[float, dict]] = {"linear": {}, "errors": {}}
    for path in sorted(study_results_dir.glob("combined_*.json.gzip")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            raw = json.load(fh)
        _, line_type, chroma_raw = path.name.replace(".json.gzip", "").split("_")
        chroma = float(chroma_raw)
        data[line_type].setdefault(chroma, {})
        for plane, pdata in raw.items():
            turns, counts = np.unique(pdata["at_turn"], return_counts=True)
            data[line_type][chroma][plane] = {"turns": turns, "counts": counts}

    normalised: dict[str, dict[float, dict]] = {}
    for line_type, by_chroma in data.items():
        normalised[line_type] = {}
        for chroma, by_plane in sorted(by_chroma.items()):
            normalised[line_type][chroma] = {}
            for plane, pdata in by_plane.items():
                turns = np.concatenate(([0], pdata["turns"]))
                counts = np.concatenate(([0], pdata["counts"]))
                deltas = df_to_delta(turns * SWEEP_PER_TURN)
                if plane == "DPneg":
                    deltas = -deltas
                normalised[line_type][chroma][plane] = {
                    "deltas": deltas,
                    "values": 1.0 - np.cumsum(counts) / NUM_PARTICLES,
                }
    return normalised


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")

    args = parse_args()
    study_results_dir = args.study_root / "study_results"
    if not study_results_dir.exists():
        raise FileNotFoundError(f"study_results not found: {study_results_dir}")

    print(f"Loading data from {study_results_dir} ...")
    normalised_intensity = load_normalised_intensity(study_results_dir)

    for line_type in args.line_types:
        n_chromas = len(normalised_intensity.get(line_type, {}))
        if n_chromas == 0:
            print(f"[WARNING] No data for line_type='{line_type}', skipping.")
            continue
        print(f"Plotting {line_type} ({n_chromas} chromas) ...")
        fig, ax = plot_intensity_drop(
            normalised_intensity,
            line_types=[line_type],
            fontsize=args.fontsize,
            legend=False,
        )
        label = "wo_errors" if line_type == "linear" else "errors"
        out_path = OUTPUT_DIR / f"intensity_loss_{label}.pdf"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
