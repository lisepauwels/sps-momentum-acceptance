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
sys.path.insert(0, str(REPO_ROOT / "helper_functions" / "tune_diagram_helpers"))

import acceptance_centers as ac
import midpoints_analysis as mpa
from tune_diagram import TuneMap


sys.path.insert(0, str(REPO_ROOT / "helper_functions"))
from load_paths import get_path as _get_path
DEFAULT_STUDY_ROOT = (
    _get_path("sps_simulations_data_root",
               default=str(Path.home() / "phd" / "data" / "sps-simulations"))
    / "momentum-acceptance" / "intensity_scan2"
)
DEFAULT_MD_ROOT = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT_DIRNAME = "Figures_midpoint_qx"
DEFAULT_MAP_ROOT = REPO_ROOT / "sps-chromaticity-maps" / "without_errors"

QX0 = 20.13
QY0 = 20.18
SWEEP_PER_TURN = 1.0
NUM_PARTICLES = 2000 * 500

DEFAULT_COLOURS = {
    "linear": "royalblue",
    "errors": "crimson",
    "MD": "blueviolet",
    "MD_means": "darkviolet",
    "MD_stds": "violet",
}
DEFAULT_MARKERS = {
    "DPpos": "o",
    "DPneg": "s",
}
LINE_LABELS = {
    "linear": "No errors",
    "errors": "Errors",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate IntensityScan2 midpoint-style plots, including "
            "delta50-vs-xi, Qx50-vs-xi, Qx50-vs-delta50, plus center and acceptance."
        )
    )
    parser.add_argument(
        "--study-root",
        type=Path,
        default=DEFAULT_STUDY_ROOT,
        help="IntensityScan2 root directory containing study_results/.",
    )
    parser.add_argument(
        "--md-root",
        type=Path,
        default=DEFAULT_MD_ROOT,
        help="MD root directory containing midpoints_MD.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <study-root>/Figures_midpoint_qx.",
    )
    parser.add_argument(
        "--map-root",
        type=Path,
        default=DEFAULT_MAP_ROOT,
        help="Directory containing WithoutErrors tune maps used for Qx midpoint estimation.",
    )
    parser.add_argument(
        "--map-root-errors",
        type=Path,
        default=None,
        help=(
            "Directory containing WithErrors tune maps used for error-model midpoint tune estimation. "
            "Defaults to sibling 'WithErrors' next to --map-root when available."
        ),
    )
    parser.add_argument(
        "--md-map-root",
        type=Path,
        default=None,
        help=(
            "Directory of tune maps used to convert MD delta midpoints into tune values. "
            "Defaults to --map-root."
        ),
    )
    parser.add_argument(
        "--map-axis",
        choices=["x", "y"],
        default="x",
        help="Which chromaticity axis the maps scan. IntensityScan2 Qx plots should use 'x'.",
    )
    parser.add_argument(
        "--fixed-map-xi",
        type=float,
        default=0.0,
        help=(
            "Companion chromaticity held fixed in the map family. "
            "For map-axis=x this is fixed xi_y; for map-axis=y this is fixed xi_x."
        ),
    )
    parser.add_argument(
        "--coupled-map-xi",
        action="store_true",
        help="Look for map files where xi_x = xi_y = study chroma.",
    )
    parser.add_argument(
        "--qx-mode",
        choices=["maps", "approx"],
        default="approx",
        help=(
            "How to estimate Qx at the 50%% midpoint. "
            "'approx' uses Qx = Qx0 + xi * Qx0 * delta50. "
            "'maps' interpolates from WithoutErrors tune maps."
        ),
    )
    parser.add_argument(
        "--line-types",
        nargs="+",
        default=["linear"],
        choices=["linear", "errors"],
        help="Simulation model variants to include. Default is no-errors only.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.5,
        help="Intensity percentile used for midpoint extraction.",
    )
    parser.add_argument(
        "--no-md",
        action="store_true",
        help="Do not overlay MD midpoint data.",
    )
    return parser.parse_args()


def load_simulation_intensity(study_results_dir: Path) -> dict:
    data_simulations: dict[str, dict[float, dict[str, dict[str, np.ndarray]]]] = {
        "linear": {},
        "errors": {},
    }
    for path in sorted(study_results_dir.glob("combined_*.json.gzip")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            study_results = json.load(fh)

        _, line_type, chroma_raw = path.name.replace(".json.gzip", "").split("_")
        chroma = float(chroma_raw)
        if chroma not in data_simulations[line_type]:
            data_simulations[line_type][chroma] = {}

        for plane in study_results:
            if abs(study_results[plane]["sweep_per_turn"]) != SWEEP_PER_TURN:
                raise ValueError(
                    f"Unexpected sweep_per_turn in {path.name} for {plane}: "
                    f"{study_results[plane]['sweep_per_turn']}"
                )
            turns, counts = np.unique(study_results[plane]["at_turn"], return_counts=True)
            data_simulations[line_type][chroma][plane] = {
                "turns": turns,
                "counts": counts,
            }

    normalised_intensity: dict[str, dict[float, dict[str, dict[str, np.ndarray]]]] = {}
    for line_type, by_chroma in data_simulations.items():
        normalised_intensity[line_type] = {}
        for chroma, by_plane in sorted(by_chroma.items()):
            normalised_intensity[line_type][chroma] = {}
            for plane, data in by_plane.items():
                turns = np.concatenate(([0], data["turns"]))
                counts = np.concatenate(([0], data["counts"]))
                deltas = mpa.df_to_delta(turns * SWEEP_PER_TURN)
                if plane == "DPneg":
                    deltas = -deltas

                values = 1.0 - np.cumsum(counts) / NUM_PARTICLES
                normalised_intensity[line_type][chroma][plane] = {
                    "deltas": deltas,
                    "values": values,
                }
    return normalised_intensity


def load_md_midpoints(md_root: Path) -> dict | None:
    path = md_root / "midpoints_MD.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        midpoints_md = json.load(fh)
    return mpa.restructure_md_midpoints(midpoints_md)


def find_without_errors_map(
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
        for pattern in patterns:
            matches = sorted(map_root.glob(pattern))
            if matches:
                return matches[0]
        return None

    if map_axis == "x":
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


def build_qx_midpoints_approx(
    midpoints: dict,
    *,
    line_types: list[str],
) -> dict:
    converted: dict = {}
    for line_type in line_types:
        converted[line_type] = {}
        for chroma, by_plane in sorted(midpoints[line_type].items()):
            converted[line_type][chroma] = {}
            for plane, delta_val in by_plane.items():
                if delta_val is None:
                    converted[line_type][chroma][plane] = np.nan
                    continue
                converted[line_type][chroma][plane] = float(QX0 + chroma * QX0 * float(delta_val))
    return converted


def build_qx_midpoints_from_maps(
    midpoints: dict,
    map_root: Path,
    *,
    line_types: list[str],
    map_axis: str,
    fixed_map_xi: float,
    coupled_map_xi: bool,
) -> tuple[dict, dict[float, Path]]:
    converted: dict = {}
    used_maps: dict[float, Path] = {}
    for line_type in line_types:
        converted[line_type] = {}
        for chroma, by_plane in sorted(midpoints[line_type].items()):
            map_path = find_without_errors_map(
                map_root,
                chroma,
                map_axis=map_axis,
                fixed_map_xi=fixed_map_xi,
                coupled_map_xi=coupled_map_xi,
            )
            if map_path is None:
                print(f"[WARNING] No WithoutErrors tune map found for xi={chroma:.3f}; skipping Qx midpoint estimate.")
                continue
            tm = TuneMap.load(str(map_path))
            used_maps[chroma] = map_path
            converted[line_type][chroma] = {}
            for plane, delta_val in by_plane.items():
                if delta_val is None:
                    converted[line_type][chroma][plane] = np.nan
                    continue
                try:
                    qx_val, _ = tm(float(delta_val))
                except ValueError:
                    print(
                        f"[WARNING] Midpoint delta {delta_val:.6f} for xi={chroma:.3f}, {plane} "
                        f"is outside tune-map range [{tm.delta_min:.6f}, {tm.delta_max:.6f}]; skipping."
                    )
                    qx_val = np.nan
                converted[line_type][chroma][plane] = float(qx_val)
    return converted, used_maps


def resolve_map_root_for_line_type(
    map_root: Path,
    map_root_errors: Path | None,
    line_type: str,
) -> Path:
    if line_type == "linear":
        return map_root
    if map_root_errors is not None:
        return map_root_errors
    if map_root.name == "WithoutErrors":
        return map_root.with_name("WithErrors")
    return map_root


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
    if component not in {"qx", "qy"}:
        raise ValueError(f"Unsupported tune component: {component}")

    converted: dict = {}
    used_maps: dict[str, dict[float, Path]] = {}
    for line_type in line_types:
        converted[line_type] = {}
        used_maps[line_type] = {}
        active_map_root = resolve_map_root_for_line_type(map_root, map_root_errors, line_type)
        for chroma, by_plane in sorted(midpoints[line_type].items()):
            map_path = find_without_errors_map(
                active_map_root,
                chroma,
                map_axis=map_axis,
                fixed_map_xi=fixed_map_xi,
                coupled_map_xi=coupled_map_xi,
            )
            if map_path is None:
                print(
                    f"[WARNING] No tune map found for xi={chroma:.3f} in {active_map_root}; "
                    f"skipping {component} midpoint estimate for {line_type}."
                )
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
                    print(
                        f"[WARNING] Midpoint delta {delta_val:.6f} for xi={chroma:.3f}, {plane} "
                        f"is outside tune-map range [{tm.delta_min:.6f}, {tm.delta_max:.6f}] "
                        f"in {map_path.name}; skipping."
                    )
                    tune_val = np.nan
                else:
                    tune_val = qx_val if component == "qx" else qy_val
                converted[line_type][chroma][plane] = float(tune_val)
    return converted, used_maps


def convert_md_delta_midpoints_to_qx_from_maps(
    md_midpoints: dict | None,
    map_root: Path,
    *,
    map_axis: str,
    fixed_map_xi: float,
    coupled_map_xi: bool,
) -> dict | None:
    if md_midpoints is None:
        return None
    converted: dict = {}
    for chroma, by_plane in md_midpoints.items():
        map_path = find_without_errors_map(
            map_root,
            chroma,
            map_axis=map_axis,
            fixed_map_xi=fixed_map_xi,
            coupled_map_xi=coupled_map_xi,
        )
        if map_path is None:
            print(f"[WARNING] No WithoutErrors tune map found for MD xi={chroma:.3f}; skipping MD Qx overlay.")
            continue
        tm = TuneMap.load(str(map_path))
        converted[chroma] = {}
        for plane, values in by_plane.items():
            arr = np.asarray(values, dtype=float)
            qx_vals: list[float] = []
            for delta_val in arr:
                try:
                    qx_val, _ = tm(float(delta_val))
                except ValueError:
                    continue
                qx_vals.append(float(qx_val))
            converted[chroma][plane] = qx_vals
    return converted


def convert_md_delta_midpoints_to_tune_from_maps(
    md_midpoints: dict | None,
    map_root: Path,
    *,
    map_axis: str,
    fixed_map_xi: float,
    coupled_map_xi: bool,
    component: str,
) -> dict | None:
    if md_midpoints is None:
        return None
    if component not in {"qx", "qy"}:
        raise ValueError(f"Unsupported tune component: {component}")
    converted: dict = {}
    for chroma, by_plane in md_midpoints.items():
        map_path = find_without_errors_map(
            map_root,
            chroma,
            map_axis=map_axis,
            fixed_map_xi=fixed_map_xi,
            coupled_map_xi=coupled_map_xi,
        )
        if map_path is None:
            print(f"[WARNING] No tune map found for MD xi={chroma:.3f}; skipping MD {component} overlay.")
            continue
        tm = TuneMap.load(str(map_path))
        converted[chroma] = {}
        for plane, values in by_plane.items():
            arr = np.asarray(values, dtype=float)
            tune_vals: list[float] = []
            for delta_val in arr:
                try:
                    qx_val, qy_val = tm(float(delta_val))
                except ValueError:
                    continue
                tune_vals.append(float(qx_val if component == "qx" else qy_val))
            converted[chroma][plane] = tune_vals
    return converted


def convert_md_delta_midpoints_to_qx_approx(md_midpoints: dict | None) -> dict | None:
    if md_midpoints is None:
        return None
    converted: dict = {}
    for chroma, by_plane in md_midpoints.items():
        converted[chroma] = {}
        for plane, values in by_plane.items():
            arr = np.asarray(values, dtype=float)
            converted[chroma][plane] = (QX0 + chroma * QX0 * arr).tolist()
    return converted


def plot_delta_midpoints_vs_xi(
    midpoints: dict,
    out_path: Path,
    *,
    line_types: list[str],
    midpoints_md: dict | None,
) -> None:
    fig, ax = mpa.plot_midpoints(
        midpoints,
        midpoints_md=midpoints_md,
        line_types=line_types,
        planes=["DPpos", "DPneg"],
        colours=DEFAULT_COLOURS,
        ylim=None,
    )
    ax.set_xlabel(r"Normalised Chromaticity $\xi$")
    ax.set_ylabel(r"$|\delta_{50\%}|$")
    ax.set_title("50% intensity midpoint vs chromaticity")
    ax.set_ylim(bottom=0.0)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
            ax.plot(
                chromas,
                values,
                color=DEFAULT_COLOURS[line_type],
                linestyle="-" if plane == "DPpos" else "--",
                marker=DEFAULT_MARKERS[plane],
                markersize=5,
                label=f"{LINE_LABELS.get(line_type, line_type)} - {plane}",
            )

    if qx_md_midpoints is not None:
        chromas_md = sorted(qx_md_midpoints.keys())
        for plane in ["DPpos", "DPneg"]:
            xs: list[float] = []
            ys: list[float] = []
            for chroma in chromas_md:
                vals = np.asarray(qx_md_midpoints[chroma][plane], dtype=float)
                xs.extend([chroma] * len(vals))
                ys.extend(vals.tolist())
            ax.scatter(
                xs,
                ys,
                color=DEFAULT_COLOURS["MD"],
                marker=DEFAULT_MARKERS[plane],
                s=12,
                label=f"MD - {plane}",
                zorder=5,
            )

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
            ax.plot(
                chromas,
                values,
                color=DEFAULT_COLOURS[line_type],
                linestyle="-" if plane == "DPpos" else "--",
                marker=DEFAULT_MARKERS[plane],
                markersize=5,
                label=f"{LINE_LABELS.get(line_type, line_type)} - {plane}",
            )

    if qy_md_midpoints is not None:
        chromas_md = sorted(qy_md_midpoints.keys())
        for plane in ["DPpos", "DPneg"]:
            xs: list[float] = []
            ys: list[float] = []
            for chroma in chromas_md:
                vals = np.asarray(qy_md_midpoints[chroma][plane], dtype=float)
                xs.extend([chroma] * len(vals))
                ys.extend(vals.tolist())
            ax.scatter(
                xs,
                ys,
                color=DEFAULT_COLOURS["MD"],
                marker=DEFAULT_MARKERS[plane],
                s=12,
                label=f"MD - {plane}",
                zorder=5,
            )

    ax.grid(alpha=0.3)
    ax.set_xlabel(r"Normalised Chromaticity $\xi$", fontsize=fontsize)
    ax.set_ylabel(r"$Q_y(\delta_{50\%})$", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 2)
    ax.set_title(r"$Q_y$ at 50% intensity loss vs chromaticity", fontsize=fontsize)
    ax.legend(ncols=2, fontsize=fontsize - 2, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_qx_vs_delta50(
    midpoints: dict,
    qx_midpoints: dict,
    out_path: Path,
    *,
    line_types: list[str],
    md_midpoints: dict | None,
    qx_md_midpoints: dict | None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    for line_type in line_types:
        if line_type not in qx_midpoints or not qx_midpoints[line_type]:
            continue
        chromas = sorted(qx_midpoints[line_type].keys())
        for plane in ["DPpos", "DPneg"]:
            delta_vals = np.array([midpoints[line_type][c][plane] for c in chromas], dtype=float)
            qx_vals = np.array([qx_midpoints[line_type][c][plane] for c in chromas], dtype=float)
            ax.plot(
                delta_vals,
                qx_vals,
                color=DEFAULT_COLOURS[line_type],
                linestyle="-" if plane == "DPpos" else "--",
                marker=DEFAULT_MARKERS[plane],
                markersize=5,
                label=f"{LINE_LABELS.get(line_type, line_type)} - {plane}",
            )

    if md_midpoints is not None and qx_md_midpoints is not None:
        chromas_md = sorted(md_midpoints.keys())
        for plane in ["DPpos", "DPneg"]:
            xs: list[float] = []
            ys: list[float] = []
            for chroma in chromas_md:
                delta_vals = np.asarray(md_midpoints[chroma][plane], dtype=float)
                qx_vals = np.asarray(qx_md_midpoints[chroma][plane], dtype=float)
                xs.extend(delta_vals.tolist())
                ys.extend(qx_vals.tolist())
            ax.scatter(
                xs,
                ys,
                color=DEFAULT_COLOURS["MD"],
                marker=DEFAULT_MARKERS[plane],
                s=12,
                label=f"MD - {plane}",
                zorder=5,
            )

    ax.grid(alpha=0.3)
    ax.set_xlabel(r"$\delta_{50\%}$")
    ax.set_ylabel(r"$Q_x(\delta_{50\%})$")
    ax.set_title(r"$Q_x$ vs $\delta$ at 50% intensity loss")
    ax.legend(ncols=2, fontsize=11, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_centers_and_acceptance(
    delta_midpoints: dict,
    out_dir: Path,
    *,
    line_types: list[str],
    md_midpoints: dict | None,
) -> None:
    fig_c, ax_c = ac.plot_centers(
        intensity_midpoints=delta_midpoints,
        line_types=line_types,
        colours=DEFAULT_COLOURS,
        md_midpoints=md_midpoints,
    )
    ax_c.set_xlabel(r"Normalised Chromaticity $\xi$")
    fig_c.savefig(out_dir / "centers_vs_xi.png", dpi=300, bbox_inches="tight")
    plt.close(fig_c)

    fig_a, ax_a = ac.plot_acceptance(
        intensity_midpoints=delta_midpoints,
        line_types=line_types,
        colours=DEFAULT_COLOURS,
        md_midpoints=md_midpoints,
    )
    ax_a.set_xlabel(r"Normalised Chromaticity $\xi$")
    ax_a.set_ylim(bottom=0.0)
    fig_a.savefig(out_dir / "acceptance_vs_xi.png", dpi=300, bbox_inches="tight")
    plt.close(fig_a)


def main() -> None:
    args = parse_args()

    study_root = args.study_root.resolve()
    md_root = args.md_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (study_root / DEFAULT_OUTPUT_DIRNAME)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    normalised_intensity = load_simulation_intensity(study_root / "study_results")
    delta_midpoints = mpa.get_midpoints(
        normalised_intensity,
        percentile=args.percentile,
    )
    md_midpoints = None if args.no_md else load_md_midpoints(md_root)
    used_maps: dict[float, Path] = {}

    if args.qx_mode == "maps":
        map_root = args.map_root.resolve()
        map_root_errors = (
            args.map_root_errors.resolve()
            if args.map_root_errors is not None
            else None
        )
        md_map_root = (
            args.md_map_root.resolve()
            if args.md_map_root is not None
            else map_root
        )
        qx_midpoints, used_maps = build_tune_midpoints_from_maps(
            delta_midpoints,
            map_root,
            line_types=args.line_types,
            map_axis=args.map_axis,
            fixed_map_xi=args.fixed_map_xi,
            coupled_map_xi=args.coupled_map_xi,
            component="qx",
            map_root_errors=map_root_errors,
        )
        qy_midpoints, _ = build_tune_midpoints_from_maps(
            delta_midpoints,
            map_root,
            line_types=args.line_types,
            map_axis=args.map_axis,
            fixed_map_xi=args.fixed_map_xi,
            coupled_map_xi=args.coupled_map_xi,
            component="qy",
            map_root_errors=map_root_errors,
        )
        qx_md_midpoints = convert_md_delta_midpoints_to_qx_from_maps(
            md_midpoints,
            md_map_root,
            map_axis=args.map_axis,
            fixed_map_xi=args.fixed_map_xi,
            coupled_map_xi=args.coupled_map_xi,
        )
        qy_md_midpoints = convert_md_delta_midpoints_to_tune_from_maps(
            md_midpoints,
            md_map_root,
            map_axis=args.map_axis,
            fixed_map_xi=args.fixed_map_xi,
            coupled_map_xi=args.coupled_map_xi,
            component="qy",
        )
    else:
        qx_midpoints = build_qx_midpoints_approx(
            delta_midpoints,
            line_types=args.line_types,
        )
        qy_midpoints = {}
        for line_type in args.line_types:
            qy_midpoints[line_type] = {}
            for chroma, by_plane in sorted(delta_midpoints[line_type].items()):
                qy_midpoints[line_type][chroma] = {}
                for plane, delta_val in by_plane.items():
                    if delta_val is None:
                        qy_midpoints[line_type][chroma][plane] = np.nan
                        continue
                    qy_midpoints[line_type][chroma][plane] = float(QY0 + chroma * QY0 * float(delta_val))
        qx_md_midpoints = convert_md_delta_midpoints_to_qx_approx(md_midpoints)
        qy_md_midpoints = None

    plot_delta_midpoints_vs_xi(
        delta_midpoints,
        output_dir / "delta50_vs_xi.png",
        line_types=args.line_types,
        midpoints_md=md_midpoints,
    )
    plot_qx_midpoints_vs_xi(
        qx_midpoints,
        output_dir / "qx50_vs_xi.png",
        line_types=args.line_types,
        qx_md_midpoints=qx_md_midpoints,
    )
    plot_qy_midpoints_vs_xi(
        qy_midpoints,
        output_dir / "qy50_vs_xi.png",
        line_types=args.line_types,
        qy_md_midpoints=qy_md_midpoints,
    )
    plot_qx_vs_delta50(
        delta_midpoints,
        qx_midpoints,
        output_dir / "qx50_vs_delta50.png",
        line_types=args.line_types,
        md_midpoints=md_midpoints,
        qx_md_midpoints=qx_md_midpoints,
    )
    plot_centers_and_acceptance(
        delta_midpoints,
        output_dir,
        line_types=args.line_types,
        md_midpoints=md_midpoints,
    )

    if used_maps:
        with (output_dir / "used_without_errors_maps.json").open("w", encoding="utf-8") as fh:
            serializable = {
                line_type: {f"{k:.6f}": str(v) for k, v in sorted(by_chroma.items())}
                for line_type, by_chroma in used_maps.items()
            }
            json.dump(serializable, fh, indent=2)

    print(f"Saved plots under {output_dir}")


if __name__ == "__main__":
    main()
