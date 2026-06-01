"""
generate_figures.py
===================
Regenerate all IntensityScan2 figures.

Simulation data is read from the sps-simulations GitLab repo.
Set 'sps_simulations_data_root' in config/paths.yaml to point to your local
clone of that repo (default: ~/phd/code/sps-simulations).

MD midpoints and tune maps are committed in sps-momentum-acceptance and
resolved automatically.

Output: studies/intensity_scan2/Figures/

Usage
-----
    python generate_figures.py                   # all figures
    python generate_figures.py --skip chroma     # skip coupled chroma scan
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(REPO_ROOT / "helper_functions"))
sys.path.insert(0, str(REPO_ROOT / "helper_functions" / "intensity_helpers"))
sys.path.insert(0, str(REPO_ROOT / "helper_functions" / "tune_diagram_helpers"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_paths import get_path
import acceptance_centers as ac
import midpoints_analysis as mpa
from intensity_loss import plot_intensity_drop
from generate_intensityscan2_midpoint_qx_plots import (
    build_tune_midpoints_from_maps,
    plot_qx_midpoints_vs_xi,
    plot_qy_midpoints_vs_xi,
)
from plot_midpoints_centres_acceptance import generate as _generate_midpoints_set

# ── paths ─────────────────────────────────────────────────────────────────────

_SPS_SIM_ROOT = get_path(
    "sps_simulations_data_root",
    default=str(Path.home() / "phd" / "code" / "sps-simulations"),
)
STUDY_RESULTS = _SPS_SIM_ROOT / "momentum-acceptance" / "intensity_scan2" / "study_results"
MD_ROOT       = Path(__file__).resolve().parent / "data"   # midpoints_MD.json committed here
OUTPUT_DIR    = Path(__file__).resolve().parent / "Figures"
MAP_LINEAR    = REPO_ROOT / "sps-chromaticity-maps" / "without_errors"
MAP_ERRORS    = REPO_ROOT / "sps-chromaticity-maps" / "with_errors"

SWEEP_PER_TURN = 1.0
NUM_PARTICLES  = 2000 * 500
FONTSIZE       = 16

# Chromaticity values shown in the errors-comparison figure
CHROMAS_COMPARISON = [0.5, 0.7, 1.0]
COLORS_COMPARISON  = {0.5: "tab:blue", 0.7: "tab:green", 1.0: "tab:red"}

# ── data loading ──────────────────────────────────────────────────────────────

def load_study_results(study_results_dir: Path) -> dict:
    """
    Load combined_*.json.gzip into normalised intensity curves.

    Returns {line_type: {chroma: {plane: {"deltas": array, "values": array}}}}
    where "values" is normalised intensity (1 at start, 0 when all particles lost)
    and "deltas" is the momentum offset swept at each turn.
    """
    if not study_results_dir.exists():
        raise FileNotFoundError(
            f"Study results not found: {study_results_dir}\n"
            "Check 'sps_simulations_data_root' in config/paths.yaml."
        )

    raw: dict = {"linear": {}, "errors": {}}
    for path in sorted(study_results_dir.glob("combined_*.json.gzip")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            content = json.load(fh)
        _, line_type, chroma_str = path.stem.replace(".json", "").split("_")
        chroma = float(chroma_str)
        raw[line_type].setdefault(chroma, {})
        for plane, pdata in content.items():
            turns, counts = np.unique(pdata["at_turn"], return_counts=True)
            raw[line_type][chroma][plane] = {"turns": turns, "counts": counts}

    out: dict = {"linear": {}, "errors": {}}
    for line_type, by_chroma in raw.items():
        for chroma, by_plane in sorted(by_chroma.items()):
            out[line_type][chroma] = {}
            for plane, d in by_plane.items():
                turns  = np.concatenate(([0], d["turns"]))
                counts = np.concatenate(([0], d["counts"]))
                deltas = mpa.df_to_delta(turns * SWEEP_PER_TURN)
                if plane == "DPneg":
                    deltas = -deltas
                out[line_type][chroma][plane] = {
                    "deltas": deltas,
                    "values": 1.0 - np.cumsum(counts) / NUM_PARTICLES,
                }
    return out


def load_md_midpoints(md_root: Path) -> dict | None:
    path = md_root / "midpoints_MD.json"
    if not path.exists():
        print(f"[WARNING] MD midpoints not found: {path}")
        return None
    with path.open("r", encoding="utf-8") as fh:
        return mpa.restructure_md_midpoints(json.load(fh))


# ── figures ───────────────────────────────────────────────────────────────────

def fig_intensity_loss(normalised_intensity: dict, output_dir: Path) -> None:
    """Intensity vs delta for the no-errors model."""
    fig, _ = plot_intensity_drop(
        normalised_intensity,
        line_types=["linear"],
        fontsize=FONTSIZE,
        legend=False,
    )
    path = output_dir / "intensity_loss_wo_errors.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


def fig_intensity_comparison(output_dir: Path) -> None:
    """
    No-errors (solid) vs with-errors (dashed) intensity curves for selected
    chromaticity values. Each chroma gets one colour; line style encodes model.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for chroma in CHROMAS_COMPARISON:
        for line_type, ls in [("linear", "-"), ("errors", "--")]:
            gzip_path = STUDY_RESULTS / f"combined_{line_type}_{chroma:.1f}.json.gzip"
            if not gzip_path.exists():
                print(f"  [skip] {gzip_path.name} not found")
                continue
            with gzip.open(gzip_path, "rt", encoding="utf-8") as fh:
                raw = json.load(fh)
            for plane, pdata in raw.items():
                turns, counts = np.unique(pdata["at_turn"], return_counts=True)
                turns  = np.concatenate(([0], turns))
                counts = np.concatenate(([0], counts))
                deltas = mpa.df_to_delta(turns * SWEEP_PER_TURN)
                if plane == "DPneg":
                    deltas = -deltas
                ax.plot(
                    deltas,
                    1.0 - np.cumsum(counts) / NUM_PARTICLES,
                    color=COLORS_COMPARISON[chroma],
                    linestyle=ls,
                    linewidth=1.5,
                )

    chroma_handles = [
        Line2D([0], [0], color=COLORS_COMPARISON[c], lw=2, label=rf"$\xi = {c}$")
        for c in CHROMAS_COMPARISON
    ]
    model_handles = [
        Line2D([0], [0], color="black", lw=2, ls="-",  label="No errors"),
        Line2D([0], [0], color="black", lw=2, ls="--", label="With errors"),
    ]
    ax.legend(handles=chroma_handles + model_handles, fontsize=FONTSIZE - 2, frameon=True)
    ax.set_xlabel(r"$\delta$", fontsize=FONTSIZE)
    ax.set_ylabel("Normalised Intensity", fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE - 2)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = output_dir / "intensity_loss_comparison_errors.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


def fig_midpoints_centres_acceptance(
    normalised_intensity: dict,
    md_midpoints: dict | None,
    output_dir: Path,
) -> None:
    """
    Two sets of five figures each (without-errors only, then both models):
      midpoints, centres, acceptance, Qx at delta_50, Qy at delta_50.

    Delegates to plot_midpoints_centres_acceptance.generate(), which also
    handles the map-based Qx/Qy estimation and saves to Figures/.
    """
    for line_types, suffix in [
        (["linear"],           "wo_errors"),
        (["linear", "errors"], "both"),
    ]:
        _generate_midpoints_set(normalised_intensity, md_midpoints, line_types, suffix)


def fig_coupled_chroma_scan() -> None:
    """
    Tune diagram overlaying selected coupled-xi trajectories from WithErrors
    ChromaScanX maps. Output goes to the map directory (not Figures/).
    """
    try:
        from plot_selected_coupled_chromascan import main as _run
        _run()
    except FileNotFoundError as exc:
        print(f"  [skip] {exc}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate all IntensityScan2 figures.")
    parser.add_argument(
        "--skip", nargs="*", default=[],
        choices=["intensity", "comparison", "midpoints", "chroma"],
        help="Figure groups to skip.",
    )
    args = parser.parse_args()
    skip = set(args.skip or [])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Study results: {STUDY_RESULTS}")
    print(f"Output:        {OUTPUT_DIR}\n")

    data = load_study_results(STUDY_RESULTS)
    md   = load_md_midpoints(MD_ROOT)

    if "intensity" not in skip:
        print("[1/4] Intensity loss (no-errors)...")
        fig_intensity_loss(data, OUTPUT_DIR)

    if "comparison" not in skip:
        print("[2/4] Error-model comparison...")
        fig_intensity_comparison(OUTPUT_DIR)

    if "midpoints" not in skip:
        print("[3/4] Midpoints / centres / acceptance / Qx50 / Qy50...")
        fig_midpoints_centres_acceptance(data, md, OUTPUT_DIR)

    if "chroma" not in skip:
        print("[4/4] Selected coupled chroma scan diagram...")
        fig_coupled_chroma_scan()

    print("\nDone.")


if __name__ == "__main__":
    main()
