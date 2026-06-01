"""
generate_figures.py
===================
Regenerate all IntensityScan2 figures.

Simulation data is read from the sps-simulations GitLab repo.
Set 'sps_simulations_data_root' in config/paths.yaml to point to your local
clone of that repo (default: ~/phd/data/sps-simulations).

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

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
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
from tune_diagram import TuneDiagram, TuneMap
from workflow_common import map_case_root_for_scan_type

# ── paths ─────────────────────────────────────────────────────────────────────

_SPS_SIM_ROOT = get_path(
    "sps_simulations_data_root",
    default=str(Path.home() / "phd" / "data" / "sps-simulations"),
)
STUDY_RESULTS = _SPS_SIM_ROOT / "momentum-acceptance" / "intensity_scan2" / "study_results"
MD_ROOT       = Path(__file__).resolve().parent / "data"
OUTPUT_DIR    = Path(__file__).resolve().parent / "Figures"
MAP_LINEAR    = REPO_ROOT / "sps-chromaticity-maps" / "without_errors"
MAP_ERRORS    = REPO_ROOT / "sps-chromaticity-maps" / "with_errors"

SWEEP_PER_TURN = 1.0
NUM_PARTICLES  = 2000 * 500
FONTSIZE       = 16
QX0            = 20.13
QY0            = 20.18

CHROMAS_COMPARISON = [0.5, 0.7, 1.0]
COLORS_COMPARISON  = {0.5: "tab:blue", 0.7: "tab:green", 1.0: "tab:red"}

COLOURS = {
    "linear":   "royalblue",
    "errors":   "crimson",
    "MD":       "blueviolet",
    "MD_means": "darkviolet",
    "MD_stds":  "violet",
}
_MARKERS     = {"DPpos": "o", "DPneg": "s"}
_LINE_LABELS = {"linear": "No errors", "errors": "Errors"}

# ── data loading ──────────────────────────────────────────────────────────────

def load_study_results(study_results_dir: Path) -> dict:
    """Load combined_*.json.gzip into normalised intensity curves.

    Returns {line_type: {chroma: {plane: {"deltas": array, "values": array}}}}
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


# ── tune map helpers ───────────────────────────────────────────────────────────

def _find_tune_map(
    map_root: Path,
    chroma: float,
    *,
    map_axis: str,
    fixed_map_xi: float,
    coupled_map_xi: bool,
) -> Path | None:
    if coupled_map_xi:
        patterns = [
            f"tune_map_Qx{QX0:.3f}_Qy{QY0:.3f}_xix{chroma:.3f}_xiy{chroma:.3f}.npz",
            f"tune_map_Qx{QX0:.3f}_Qy{QY0:.3f}_xix{chroma:.2f}_xiy{chroma:.2f}.npz",
            f"tune_map_Qx{QX0:.3f}_Qy{QY0:.3f}_xix{chroma:.1f}_xiy{chroma:.1f}.npz",
        ]
    elif map_axis == "x":
        patterns = [
            f"tune_map_Qx{QX0:.3f}_Qy{QY0:.3f}_xix{chroma:.3f}_xiy{fixed_map_xi:.3f}.npz",
            f"tune_map_Qx{QX0:.3f}_Qy{QY0:.3f}_xix{chroma:.2f}_xiy{fixed_map_xi:.3f}.npz",
            f"tune_map_Qx{QX0:.3f}_Qy{QY0:.3f}_xix{chroma:.1f}_xiy{fixed_map_xi:.3f}.npz",
            f"tune_map_Qx{QX0:.3f}_Qy{QY0:.3f}_xix{chroma:.3f}_xiy*.npz",
            f"tune_map_Qx{QX0:.3f}_Qy{QY0:.3f}_xix{chroma:.2f}_xiy*.npz",
            f"tune_map_Qx{QX0:.3f}_Qy{QY0:.3f}_xix{chroma:.1f}_xiy*.npz",
        ]
    else:
        patterns = [
            f"tune_map_Qx{QX0:.3f}_Qy{QY0:.3f}_xix{fixed_map_xi:.3f}_xiy{chroma:.3f}.npz",
            f"tune_map_Qx{QX0:.3f}_Qy{QY0:.3f}_xix{fixed_map_xi:.3f}_xiy{chroma:.2f}.npz",
            f"tune_map_Qx{QX0:.3f}_Qy{QY0:.3f}_xix{fixed_map_xi:.3f}_xiy{chroma:.1f}.npz",
            f"tune_map_Qx{QX0:.3f}_Qy{QY0:.3f}_xix*_xiy{chroma:.3f}.npz",
            f"tune_map_Qx{QX0:.3f}_Qy{QY0:.3f}_xix*_xiy{chroma:.2f}.npz",
            f"tune_map_Qx{QX0:.3f}_Qy{QY0:.3f}_xix*_xiy{chroma:.1f}.npz",
        ]
    for pattern in patterns:
        matches = sorted(map_root.glob(pattern))
        if matches:
            return matches[0]
    return None


def build_tune_midpoints_from_maps(
    midpoints: dict,
    map_root: Path,
    *,
    line_types: list[str],
    map_axis: str,
    fixed_map_xi: float,
    coupled_map_xi: bool,
    component: str,
    map_root_errors: Path | None = None,
) -> tuple[dict, dict[str, dict[float, Path]]]:
    """Interpolate Qx or Qy at the δ₅₀ midpoint for each (line_type, chroma) from tune maps."""
    if component not in {"qx", "qy"}:
        raise ValueError(f"Unsupported tune component: {component}")

    converted: dict = {}
    used_maps: dict[str, dict[float, Path]] = {}
    for line_type in line_types:
        converted[line_type] = {}
        used_maps[line_type] = {}
        if line_type == "linear":
            active_root = map_root
        elif map_root_errors is not None:
            active_root = map_root_errors
        else:
            active_root = map_root.with_name("with_errors") if map_root.name == "without_errors" else map_root

        for chroma, by_plane in sorted(midpoints[line_type].items()):
            map_path = _find_tune_map(
                active_root, chroma,
                map_axis=map_axis, fixed_map_xi=fixed_map_xi, coupled_map_xi=coupled_map_xi,
            )
            if map_path is None:
                print(f"[WARNING] No tune map for xi={chroma:.3f} in {active_root}; skipping {component} for {line_type}.")
                continue
            tm = TuneMap.load(str(map_path))
            used_maps[line_type][chroma] = map_path
            converted[line_type][chroma] = {}
            for plane, delta_val in by_plane.items():
                if delta_val is None:
                    converted[line_type][chroma][plane] = np.nan
                    continue
                try:
                    qx_val, qy_val = tm(float(delta_val))
                except ValueError:
                    print(f"[WARNING] delta {delta_val:.6f} outside map range for xi={chroma:.3f}, {plane}; skipping.")
                    tune_val = np.nan
                else:
                    tune_val = qx_val if component == "qx" else qy_val
                converted[line_type][chroma][plane] = float(tune_val)
    return converted, used_maps


def convert_md_delta_midpoints_to_tune_from_maps(
    md_midpoints: dict | None,
    map_root: Path,
    *,
    map_axis: str,
    fixed_map_xi: float,
    coupled_map_xi: bool,
    component: str,
) -> dict | None:
    """Convert MD δ midpoints to Qx or Qy values by interpolating from tune maps."""
    if md_midpoints is None:
        return None
    if component not in {"qx", "qy"}:
        raise ValueError(f"Unsupported tune component: {component}")
    converted: dict = {}
    for chroma, by_plane in md_midpoints.items():
        map_path = _find_tune_map(
            map_root, chroma,
            map_axis=map_axis, fixed_map_xi=fixed_map_xi, coupled_map_xi=coupled_map_xi,
        )
        if map_path is None:
            print(f"[WARNING] No tune map for MD xi={chroma:.3f}; skipping {component} overlay.")
            continue
        tm = TuneMap.load(str(map_path))
        converted[chroma] = {}
        for plane, values in by_plane.items():
            tune_vals: list[float] = []
            for delta_val in np.asarray(values, dtype=float):
                try:
                    qx_val, qy_val = tm(float(delta_val))
                except ValueError:
                    continue
                tune_vals.append(float(qx_val if component == "qx" else qy_val))
            converted[chroma][plane] = tune_vals
    return converted


def plot_qx_midpoints_vs_xi(
    qx_midpoints: dict,
    out_path: Path,
    *,
    line_types: list[str],
    qx_md_midpoints: dict | None,
    fontsize: int = 16,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for line_type in line_types:
        if line_type not in qx_midpoints or not qx_midpoints[line_type]:
            continue
        chromas = sorted(qx_midpoints[line_type].keys())
        for plane in ["DPpos", "DPneg"]:
            values = [qx_midpoints[line_type][chroma][plane] for chroma in chromas]
            ax.plot(chromas, values, color=COLOURS[line_type],
                    linestyle="-" if plane == "DPpos" else "--",
                    marker=_MARKERS[plane], markersize=5,
                    label=f"{_LINE_LABELS.get(line_type, line_type)} - {plane}")
    if qx_md_midpoints is not None:
        for plane in ["DPpos", "DPneg"]:
            xs, ys = [], []
            for chroma in sorted(qx_md_midpoints.keys()):
                vals = np.asarray(qx_md_midpoints[chroma][plane], dtype=float)
                xs.extend([chroma] * len(vals))
                ys.extend(vals.tolist())
            ax.scatter(xs, ys, color=COLOURS["MD"], marker=_MARKERS[plane],
                       s=12, label=f"MD - {plane}", zorder=5)
    ax.grid(alpha=0.3)
    ax.set_xlabel(r"Normalised Chromaticity $\xi$", fontsize=fontsize)
    ax.set_ylabel(r"$Q_x(\delta_{50\%})$", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 2)
    ax.set_title(r"$Q_x$ at 50% intensity loss vs chromaticity", fontsize=fontsize)
    ax.legend(ncols=2, fontsize=fontsize - 2, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_qy_midpoints_vs_xi(
    qy_midpoints: dict,
    out_path: Path,
    *,
    line_types: list[str],
    qy_md_midpoints: dict | None,
    fontsize: int = 16,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for line_type in line_types:
        if line_type not in qy_midpoints or not qy_midpoints[line_type]:
            continue
        chromas = sorted(qy_midpoints[line_type].keys())
        for plane in ["DPpos", "DPneg"]:
            values = [qy_midpoints[line_type][chroma][plane] for chroma in chromas]
            ax.plot(chromas, values, color=COLOURS[line_type],
                    linestyle="-" if plane == "DPpos" else "--",
                    marker=_MARKERS[plane], markersize=5,
                    label=f"{_LINE_LABELS.get(line_type, line_type)} - {plane}")
    if qy_md_midpoints is not None:
        for plane in ["DPpos", "DPneg"]:
            xs, ys = [], []
            for chroma in sorted(qy_md_midpoints.keys()):
                vals = np.asarray(qy_md_midpoints[chroma][plane], dtype=float)
                xs.extend([chroma] * len(vals))
                ys.extend(vals.tolist())
            ax.scatter(xs, ys, color=COLOURS["MD"], marker=_MARKERS[plane],
                       s=12, label=f"MD - {plane}", zorder=5)
    ax.grid(alpha=0.3)
    ax.set_xlabel(r"Normalised Chromaticity $\xi$", fontsize=fontsize)
    ax.set_ylabel(r"$Q_y(\delta_{50\%})$", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 2)
    ax.set_title(r"$Q_y$ at 50% intensity loss vs chromaticity", fontsize=fontsize)
    ax.legend(ncols=2, fontsize=fontsize - 2, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── figure functions ───────────────────────────────────────────────────────────

def fig_intensity_loss(normalised_intensity: dict, output_dir: Path) -> None:
    """Intensity vs delta for the no-errors model."""
    fig, _ = plot_intensity_drop(
        normalised_intensity, line_types=["linear"], fontsize=FONTSIZE, legend=False,
    )
    path = output_dir / "intensity_loss_wo_errors.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


def fig_intensity_comparison(output_dir: Path) -> None:
    """No-errors (solid) vs with-errors (dashed) for selected chromaticity values."""
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
                ax.plot(deltas, 1.0 - np.cumsum(counts) / NUM_PARTICLES,
                        color=COLORS_COMPARISON[chroma], linestyle=ls, linewidth=1.5)
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


def generate(
    normalised_intensity: dict,
    md_midpoints: dict | None,
    line_types: list[str],
    suffix: str,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Generate midpoints, centres, acceptance, Qx50, Qy50 figures for one model set."""
    intensity_midpoints = mpa.get_midpoints(normalised_intensity)
    colours = {k: v for k, v in COLOURS.items() if k in line_types or k.startswith("MD")}

    def _save(fig, name: str) -> None:
        path = output_dir / name
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  {name}")

    fig, _ = mpa.plot_midpoints(
        intensity_midpoints, midpoints_md=md_midpoints,
        line_types=line_types, colours=colours, fontsize=FONTSIZE,
    )
    _save(fig, f"midpoints_{suffix}.pdf")

    fig, _ = ac.plot_centers(
        intensity_midpoints, line_types=line_types,
        colours=colours, md_midpoints=md_midpoints, fontsize=FONTSIZE,
    )
    _save(fig, f"centres_{suffix}.pdf")

    fig, ax = ac.plot_acceptance(
        intensity_midpoints, line_types=line_types,
        colours=colours, md_midpoints=md_midpoints, fontsize=FONTSIZE,
    )
    ax.set_ylim(bottom=0)
    _save(fig, f"acceptance_{suffix}.pdf")

    map_root_errors = MAP_ERRORS if "errors" in line_types else None
    qx_midpoints, _ = build_tune_midpoints_from_maps(
        intensity_midpoints, MAP_LINEAR, line_types=line_types, map_axis="x",
        fixed_map_xi=0.0, coupled_map_xi=True, component="qx",
        map_root_errors=map_root_errors,
    )
    qy_midpoints, _ = build_tune_midpoints_from_maps(
        intensity_midpoints, MAP_LINEAR, line_types=line_types, map_axis="x",
        fixed_map_xi=0.0, coupled_map_xi=True, component="qy",
        map_root_errors=map_root_errors,
    )
    plot_qx_midpoints_vs_xi(
        qx_midpoints, output_dir / f"qx50_vs_xi_{suffix}.pdf",
        line_types=line_types, qx_md_midpoints=None, fontsize=FONTSIZE,
    )
    print(f"  qx50_vs_xi_{suffix}.pdf")
    plot_qy_midpoints_vs_xi(
        qy_midpoints, output_dir / f"qy50_vs_xi_{suffix}.pdf",
        line_types=line_types, qy_md_midpoints=None, fontsize=FONTSIZE,
    )
    print(f"  qy50_vs_xi_{suffix}.pdf")


def fig_midpoints_centres_acceptance(
    normalised_intensity: dict,
    md_midpoints: dict | None,
    output_dir: Path,
) -> None:
    for line_types, suffix in [
        (["linear"],           "wo_errors"),
        (["linear", "errors"], "both"),
    ]:
        generate(normalised_intensity, md_midpoints, line_types, suffix, output_dir)


def fig_coupled_chroma_scan() -> None:
    """Tune diagram overlaying selected coupled-xi trajectories from WithErrors ChromaScanX maps."""
    XI_VALUES = [-1.5, -0.5, 0.0, 0.5, 1.5]
    try:
        map_dir = map_case_root_for_scan_type("ChromaScanX", "WithErrors")
        out_path = map_dir / "tune_diagram_WithErrors_selected_coupled_xi.pdf"
        maps: list[tuple[float, TuneMap]] = []
        for xi in XI_VALUES:
            path = map_dir / f"tune_map_Qx{QX0:.3f}_Qy{QY0:.3f}_xix{xi:.3f}_xiy{xi:.3f}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Missing map for xi={xi:.3f}: {path}")
            maps.append((xi, TuneMap.load(str(path))))
    except FileNotFoundError as exc:
        print(f"  [skip] {exc}")
        return

    td = TuneDiagram(qx0=QX0, qy0=QY0, half_range=0.4, max_order=3, skew=True)
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    td.plot(ax=ax, show_working_point=False)
    ax.set_aspect("equal")
    ax.scatter(QX0, QY0, color="k", s=55, zorder=7)

    norm_positions = np.linspace(0.1, 0.9, len(maps))
    xi_handles: list[mlines.Line2D] = []
    for (xi, tm), cc in zip(maps, norm_positions):
        color = cm.get_cmap("plasma")(cc)
        d_arr, qx_arr, qy_arr = tm.sample(400)
        ax.plot(qx_arr, qy_arr, color=color, lw=2.0, alpha=0.95)
        neg_idx = int(np.argmin(d_arr))
        pos_idx = int(np.argmax(d_arr))
        ax.annotate("", xy=(qx_arr[neg_idx], qy_arr[neg_idx]),
                    xytext=(qx_arr[min(8, len(qx_arr) - 1)], qy_arr[min(8, len(qy_arr) - 1)]),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=1.8), zorder=7)
        ax.annotate("", xy=(qx_arr[pos_idx], qy_arr[pos_idx]),
                    xytext=(qx_arr[max(len(qx_arr) - 9, 0)], qy_arr[max(len(qy_arr) - 9, 0)]),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.8), zorder=7)
        xi_handles.append(mlines.Line2D([], [], color=color, lw=2.0, label=rf"$\xi={xi:.1f}$"))

    xi_handles.extend([
        mpatches.FancyArrowPatch((0, 0), (1, 0), arrowstyle="->", mutation_scale=12,
                                  color="red", label="Positive sweep"),
        mpatches.FancyArrowPatch((0, 0), (1, 0), arrowstyle="->", mutation_scale=12,
                                  color="blue", label="Negative sweep"),
    ])
    resonance_legend = ax.legend(handles=td.legend_handles(), loc="upper left", frameon=True)
    ax.add_artist(resonance_legend)
    ax.legend(handles=xi_handles, loc="lower right", frameon=True)
    ax.set_xlabel(r"$Q_x$")
    ax.set_ylabel(r"$Q_y$")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import matplotlib
    matplotlib.use("Agg")

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
