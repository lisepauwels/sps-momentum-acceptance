"""
PlotScanSummaries.py
====================
Simulation-only summary plots for the tune-scan workflow.

For each scan type and optics case, this script produces:

    - intensity_vs_delta.pdf
    - midpoints.pdf
    - centers.pdf
    - acceptance.pdf

Unlike the older TuneDiagramVariations notebook, the varying parameter is the
scan parameter of the current workflow:

    - Qx for QxScan
    - Qy for QyScan
    - xi_x for ChromaScanX
    - xi_y for ChromaScanY
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "helper_functions" / "tune_diagram_helpers"))
sys.path.insert(0, str(REPO_ROOT / "helper_functions" / "intensity_helpers"))

from PlotMDIntensity import (
    EXCLUDED_FILES,
    MAP_CASES,
    REP_IDS,
    SCANS,
    _folder_name_chroma,
    _folder_name_tune,
    _is_valid_rep,
    _scan_keys_and_labels,
    _smooth,
    dr_to_delta,
    get_data_from_file,
    is_tune_scan,
    scan_key_label,
    scan_plot_suffix,
    scan_key_to_working_point,
    scan_param_value,
)
from midpoints_analysis import interpolate_percentile_val


OUTPUT_ROOT = Path("summary_plots")
PLANES = ["DPneg", "DPpos"]


def _common_delta(reps: list[dict], n: int = 250) -> np.ndarray | None:
    d_min = max(r["delta"].min() for r in reps)
    d_max = min(r["delta"].max() for r in reps)
    if d_min >= d_max:
        return None
    return np.linspace(d_min, d_max, n)


def _interp_stack(reps: list[dict], delta_c: np.ndarray, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = np.array([np.interp(delta_c, r["delta"], r[key]) for r in reps])
    return stack, stack.mean(axis=0), stack.std(axis=0)


def _exclude_key(scan_cfg: dict, key, side: str, rep_id: int):
    if is_tune_scan(scan_cfg):
        qx, qy = scan_key_to_working_point(scan_cfg, key)
        return (round(qx, 3), round(qy, 2), side, rep_id)
    xi_x, xi_y = key
    return (round(xi_x, 3), round(xi_y, 2), side, rep_id)


def _side_folder(scan_cfg: dict, key, side: str) -> Path | None:
    if is_tune_scan(scan_cfg):
        qx, qy = scan_key_to_working_point(scan_cfg, key)
        return _folder_name_tune(qx, qy, "NEG" if side == "DPneg" else "POS")
    xi_x, xi_y = key
    return _folder_name_chroma(xi_x, xi_y, "NEG" if side == "DPneg" else "POS")


def load_side_reps(scan_cfg: dict, key, side: str) -> list[dict] | None:
    folder = _side_folder(scan_cfg, key, side)
    label = scan_key_label(scan_cfg, key)
    if folder is None:
        print(f"  [WARNING] Folder not found for {label} {side}")
        return None

    reps = []
    for rep_id in REP_IDS:
        if _exclude_key(scan_cfg, key, "NEG" if side == "DPneg" else "POS", rep_id) in EXCLUDED_FILES:
            continue

        try:
            _, _, radial, intensity_arr = get_data_from_file(folder / f"id{rep_id}.json")
        except FileNotFoundError as exc:
            print(f"  [WARNING] {exc}")
            continue

        delta = 1000 * dr_to_delta(radial)
        order = np.argsort(delta)
        delta = delta[order]
        intensity = np.clip(intensity_arr[order], 0.0, 1.0)

        _, unique_idx = np.unique(delta, return_index=True)
        delta = delta[unique_idx]
        intensity = intensity[unique_idx]

        if not _is_valid_rep(intensity):
            continue

        reps.append(
            {
                "delta": delta,
                "intensity": intensity,
                "di_ddelta": _smooth(np.gradient(intensity, delta)),
            }
        )

    return reps if reps else None


def build_scan_data(scan_cfg: dict) -> tuple[dict, list, list, str]:
    keys, scan_vals, cbar_label = _scan_keys_and_labels(scan_cfg)
    all_data = {}
    for key in keys:
        per_plane = {}
        for plane in PLANES:
            per_plane[plane] = load_side_reps(scan_cfg, key, plane)
        all_data[key] = per_plane
    return all_data, keys, scan_vals, cbar_label


def summarise_midpoints(all_data: dict) -> dict:
    summary = {}
    for key, planes in all_data.items():
        summary[key] = {}
        for plane in PLANES:
            reps = planes.get(plane)
            if reps is None:
                summary[key][plane] = np.nan
                continue
            delta_c = _common_delta(reps)
            if delta_c is None:
                summary[key][plane] = np.nan
                continue
            _, int_mean, _ = _interp_stack(reps, delta_c, "intensity")
            mid = interpolate_percentile_val(delta_c, int_mean, percentile=0.5)
            summary[key][plane] = np.nan if mid is None else float(mid)
    return summary


def plot_intensity_vs_delta(scan_cfg: dict, all_data: dict, scan_vals: list, cbar_label: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = colormaps["plasma"]
    norm = mpl.colors.Normalize(vmin=min(scan_vals), vmax=max(scan_vals))

    for key, planes in all_data.items():
        scan_val = scan_param_value(scan_cfg, key)
        color = cmap(norm(scan_val))
        for plane in PLANES:
            reps = planes.get(plane)
            if reps is None:
                continue
            delta_c = _common_delta(reps)
            if delta_c is None:
                continue
            _, int_mean, _ = _interp_stack(reps, delta_c, "intensity")
            ls = "-" if plane == "DPpos" else "--"
            ax.plot(delta_c, int_mean, color=color, lw=1.6, ls=ls)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=12)

    handles = [
        Line2D([0], [0], color="k", lw=1.6, ls="-", label="DPpos"),
        Line2D([0], [0], color="k", lw=1.6, ls="--", label="DPneg"),
    ]
    ax.legend(handles=handles, loc="best")
    ax.set_xlabel(r"$\delta\ [10^{-3}]$")
    ax.set_ylabel("Normalised Intensity")
    ax.set_title(f"Intensity vs delta — {scan_cfg['label']}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_midpoints(scan_cfg: dict, midpoints: dict, scan_vals: list, cbar_label: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = colormaps["plasma"]
    norm = mpl.colors.Normalize(vmin=min(scan_vals), vmax=max(scan_vals))

    x = np.array([scan_param_value(scan_cfg, key) for key in midpoints], dtype=float)
    y_pos = np.array([midpoints[key]["DPpos"] for key in midpoints], dtype=float)
    y_neg = np.array([np.abs(midpoints[key]["DPneg"]) for key in midpoints], dtype=float)
    colors = [cmap(norm(xx)) for xx in x]

    ax.scatter(x, y_pos, c=colors, marker="o", s=40, label="DPpos")
    ax.scatter(x, y_neg, c=colors, marker="s", s=40, label="DPneg")
    ax.plot(x, y_pos, color="0.4", alpha=0.4, lw=0.8)
    ax.plot(x, y_neg, color="0.4", alpha=0.4, lw=0.8)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=12)

    ax.set_xlabel(cbar_label)
    ax.set_ylabel(r"$|\delta_{50\%}| \ [10^{-3}]$")
    ax.set_title(f"Midpoints — {scan_cfg['label']}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_centers(scan_cfg: dict, midpoints: dict, scan_vals: list, cbar_label: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = colormaps["plasma"]
    norm = mpl.colors.Normalize(vmin=min(scan_vals), vmax=max(scan_vals))

    x = np.array([scan_param_value(scan_cfg, key) for key in midpoints], dtype=float)
    centers = np.array(
        [0.5 * (midpoints[key]["DPpos"] + midpoints[key]["DPneg"]) for key in midpoints],
        dtype=float,
    )
    colors = [cmap(norm(xx)) for xx in x]

    ax.scatter(x, centers, c=colors, s=40)
    ax.plot(x, centers, color="0.4", alpha=0.4, lw=0.8)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=12)

    ax.set_xlabel(cbar_label)
    ax.set_ylabel(r"$C = (\delta_{+} + \delta_{-})/2$")
    ax.set_title(f"Centers — {scan_cfg['label']}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_acceptance(scan_cfg: dict, midpoints: dict, scan_vals: list, cbar_label: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = colormaps["plasma"]
    norm = mpl.colors.Normalize(vmin=min(scan_vals), vmax=max(scan_vals))

    x = np.array([scan_param_value(scan_cfg, key) for key in midpoints], dtype=float)
    acceptance = np.array(
        [midpoints[key]["DPpos"] - midpoints[key]["DPneg"] for key in midpoints],
        dtype=float,
    )
    colors = [cmap(norm(xx)) for xx in x]

    ax.scatter(x, acceptance, c=colors, s=40)
    ax.plot(x, acceptance, color="0.4", alpha=0.4, lw=0.8)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=12)

    ax.set_xlabel(cbar_label)
    ax.set_ylabel(r"$A = \delta_{+} - \delta_{-}$")
    ax.set_title(f"Acceptance — {scan_cfg['label']}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for scan_cfg in SCANS:
        all_data, keys, scan_vals, cbar_label = build_scan_data(scan_cfg)
        midpoints = summarise_midpoints(all_data)

        out_dir = OUTPUT_ROOT / scan_cfg["label"]
        out_dir.mkdir(parents=True, exist_ok=True)

        suffix = scan_plot_suffix(scan_cfg)
        def name(stem: str) -> str:
            if suffix:
                return f"{stem}_{suffix}.pdf"
            return f"{stem}.pdf"

        figs = [
            (plot_intensity_vs_delta(scan_cfg, all_data, scan_vals, cbar_label), name("intensity_vs_delta")),
            (plot_midpoints(scan_cfg, midpoints, scan_vals, cbar_label), name("midpoints")),
            (plot_centers(scan_cfg, midpoints, scan_vals, cbar_label), name("centers")),
            (plot_acceptance(scan_cfg, midpoints, scan_vals, cbar_label), name("acceptance")),
        ]
        for fig, name in figs:
            fig.savefig(out_dir / name, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_dir / name}")


if __name__ == "__main__":
    main()
