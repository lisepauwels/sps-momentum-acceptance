"""
plot_comparison_errors.py
==========================
Intensity-loss comparison: without errors (solid) vs with errors (dashed),
for selected chromaticity values, each in a fixed colour.

Output: studies/intensity_scan2/Figures/intensity_loss_comparison_errors.pdf
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "helper_functions" / "intensity_helpers"))

from midpoints_analysis import df_to_delta

STUDY_RESULTS = Path(
    "/Users/lisepauwels/sps_simulations/Studies/MomentumAcceptance/IntensityScan2/study_results"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "Figures"

CHROMAS = [0.5, 0.7, 1.0]
COLORS  = {0.5: "tab:blue", 0.7: "tab:green", 1.0: "tab:red"}
FONTSIZE = 16

SWEEP_PER_TURN = 1.0
NUM_PARTICLES  = 2000 * 500


def load_one(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        raw = json.load(fh)
    result = {}
    for plane, pdata in raw.items():
        turns, counts = np.unique(pdata["at_turn"], return_counts=True)
        turns  = np.concatenate(([0], turns))
        counts = np.concatenate(([0], counts))
        deltas = df_to_delta(turns * SWEEP_PER_TURN)
        if plane == "DPneg":
            deltas = -deltas
        result[plane] = {
            "deltas": deltas,
            "values": 1.0 - np.cumsum(counts) / NUM_PARTICLES,
        }
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    for chroma in CHROMAS:
        color = COLORS[chroma]
        for line_type, ls in [("linear", "-"), ("errors", "--")]:
            fname = f"combined_{line_type}_{chroma:.1f}.json.gzip"
            path  = STUDY_RESULTS / fname
            if not path.exists():
                print(f"[WARNING] Not found: {path}")
                continue
            data = load_one(path)
            for plane in ["DPpos", "DPneg"]:
                ax.plot(
                    data[plane]["deltas"],
                    data[plane]["values"],
                    color=color,
                    linestyle=ls,
                    linewidth=1.5,
                )

    # Legend: one entry per chroma (colour) + one per model (linestyle)
    chroma_handles = [
        Line2D([0], [0], color=COLORS[c], lw=2, label=rf"$\xi = {c}$")
        for c in CHROMAS
    ]
    model_handles = [
        Line2D([0], [0], color="black", lw=2, ls="-",  label="No errors"),
        Line2D([0], [0], color="black", lw=2, ls="--", label="With errors"),
    ]
    ax.legend(
        handles=chroma_handles + model_handles,
        fontsize=FONTSIZE - 2,
        frameon=True,
    )

    ax.set_xlabel(r"$\delta$", fontsize=FONTSIZE)
    ax.set_ylabel("Normalised Intensity", fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE - 2)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "intensity_loss_comparison_errors.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
