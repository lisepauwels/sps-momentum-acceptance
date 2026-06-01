"""
plot_midpoints_centres_acceptance.py
======================================
Regenerate midpoint, centre, and acceptance plots for IntensityScan2,
with MD data overlaid.

Produces two sets of PDFs in studies/intensity_scan2/Figures/:
  - *_wo_errors.pdf   (linear model only + MD)
  - *_both.pdf        (linear + errors + MD)

Usage
-----
    python plot_midpoints_centres_acceptance.py
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "helper_functions" / "intensity_helpers"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import acceptance_centers as ac
import midpoints_analysis as mpa
from generate_intensityscan2_midpoint_qx_plots import (
    build_tune_midpoints_from_maps,
    convert_md_delta_midpoints_to_tune_from_maps,
    plot_qx_midpoints_vs_xi,
    plot_qy_midpoints_vs_xi,
)

from load_paths import get_path as _get_path
STUDY_ROOT = (
    _get_path("sps_simulations_data_root",
               default=str(Path.home() / "phd" / "data" / "sps-simulations"))
    / "momentum-acceptance" / "intensity_scan2"
)
MD_ROOT    = Path(__file__).resolve().parent / "data"
MAP_ROOT_LINEAR = REPO_ROOT / "sps-chromaticity-maps" / "without_errors"
MAP_ROOT_ERRORS = REPO_ROOT / "sps-chromaticity-maps" / "with_errors"
OUTPUT_DIR      = Path(__file__).resolve().parent / "Figures"

SWEEP_PER_TURN = 1.0
NUM_PARTICLES  = 2000 * 500
FONTSIZE       = 16

COLOURS = {
    "linear":   "royalblue",
    "errors":   "crimson",
    "MD":       "blueviolet",
    "MD_means": "darkviolet",
    "MD_stds":  "violet",
}


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
                turns  = np.concatenate(([0], pdata["turns"]))
                counts = np.concatenate(([0], pdata["counts"]))
                deltas = mpa.df_to_delta(turns * SWEEP_PER_TURN)
                if plane == "DPneg":
                    deltas = -deltas
                normalised[line_type][chroma][plane] = {
                    "deltas": deltas,
                    "values": 1.0 - np.cumsum(counts) / NUM_PARTICLES,
                }
    return normalised


def load_md_midpoints(md_root: Path) -> dict | None:
    path = md_root / "midpoints_MD.json"
    if not path.exists():
        print(f"[WARNING] MD midpoints not found: {path}")
        return None
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return mpa.restructure_md_midpoints(raw)


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def generate(normalised_intensity, md_midpoints, line_types: list[str], suffix: str) -> None:
    intensity_midpoints = mpa.get_midpoints(normalised_intensity)
    colours = {k: v for k, v in COLOURS.items() if k in line_types or k.startswith("MD")}

    fig, _ = mpa.plot_midpoints(
        intensity_midpoints,
        midpoints_md=md_midpoints,
        line_types=line_types,
        colours=colours,
        fontsize=FONTSIZE,
    )
    save(fig, OUTPUT_DIR / f"midpoints_{suffix}.pdf")

    fig, _ = ac.plot_centers(
        intensity_midpoints,
        line_types=line_types,
        colours=colours,
        md_midpoints=md_midpoints,
        fontsize=FONTSIZE,
    )
    save(fig, OUTPUT_DIR / f"centres_{suffix}.pdf")

    fig, ax = ac.plot_acceptance(
        intensity_midpoints,
        line_types=line_types,
        colours=colours,
        md_midpoints=md_midpoints,
        fontsize=FONTSIZE,
    )
    ax.set_ylim(bottom=0)
    save(fig, OUTPUT_DIR / f"acceptance_{suffix}.pdf")

    # Qx and Qy at delta_50 from tune maps
    map_root_errors = MAP_ROOT_ERRORS if "errors" in line_types else None
    qx_midpoints, _ = build_tune_midpoints_from_maps(
        intensity_midpoints, MAP_ROOT_LINEAR,
        line_types=line_types, map_axis="x", fixed_map_xi=0.0,
        coupled_map_xi=True, component="qx",
        map_root_errors=map_root_errors,
    )
    qy_midpoints, _ = build_tune_midpoints_from_maps(
        intensity_midpoints, MAP_ROOT_LINEAR,
        line_types=line_types, map_axis="x", fixed_map_xi=0.0,
        coupled_map_xi=True, component="qy",
        map_root_errors=map_root_errors,
    )
    plot_qx_midpoints_vs_xi(
        qx_midpoints, OUTPUT_DIR / f"qx50_vs_xi_{suffix}.pdf",
        line_types=line_types, qx_md_midpoints=None, fontsize=FONTSIZE,
    )
    print(f"  Saved qx50_vs_xi_{suffix}.pdf")

    plot_qy_midpoints_vs_xi(
        qy_midpoints, OUTPUT_DIR / f"qy50_vs_xi_{suffix}.pdf",
        line_types=line_types, qy_md_midpoints=None, fontsize=FONTSIZE,
    )
    print(f"  Saved qy50_vs_xi_{suffix}.pdf")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading simulation data...")
    normalised_intensity = load_normalised_intensity(STUDY_ROOT / "study_results")
    md_midpoints = load_md_midpoints(MD_ROOT)

    print("\nGenerating without-errors + MD...")
    generate(normalised_intensity, md_midpoints, ["linear"], "wo_errors")

    print("\nGenerating both models + MD...")
    generate(normalised_intensity, md_midpoints, ["linear", "errors"], "both")

    print(f"\nDone. Plots in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
