"""
plot_tune_maps.py
=================
Reloads pre-computed TuneMaps and reproduces the sweep-trajectory
tune-diagram plots.

This version is generic and supports all scan types produced by TuneScan.py:
    - QxScan
    - QyScan
    - ChromaScanX
    - ChromaScanY

It only loads and plots TuneMap files. It does not use measurements.

Current storage convention:
    - tune scans:          SweepTrajectoryMaps/QxScan/... and QyScan/...
    - chromaticity maps:   sps-chromaticity-maps/{with_errors,without_errors,simplified}/
    - output PDFs:         sps-chromaticity-maps/plots/<scan_bucket>/<case>/

Usage
-----
    python PlotTuneMaps.py

To select which scans to plot, edit ACTIVE_SCANS below.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "helper_functions"))
sys.path.insert(0, str(REPO_ROOT / "helper_functions" / "intensity_helpers"))
sys.path.insert(0, str(THIS_DIR))

from load_paths import get_path
from tune_diagram import TuneDiagram, TuneMap
from workflow_common import (
    CHROMA_SCAN_X_MACHINE,
    CHROMA_SCAN_Y_MACHINE,
    MAP_CASES,
    MAX_ORDER,
    QX_SCAN,
    QX_SCAN_MACHINE,
    QY_SCAN,
    QY_SCAN_MACHINE,
    finalize_colorbar_heights,
    is_chroma_scan_type,
    iter_scan_entries,
    make_tune_diagram_figure,
    map_case_root_for_scan_type,
    plot_case_root_for_scan_type,
    scan_plot_suffix,
    scan_param_value,
    tune_diagram_spec,
    tune_map_filename,
)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration  (must match TuneScan.py)
# ──────────────────────────────────────────────────────────────────────────────

# Select which scans to plot.
ACTIVE_SCANS = [
    QX_SCAN,
    QY_SCAN,
    QX_SCAN_MACHINE,
    QY_SCAN_MACHINE,
    CHROMA_SCAN_Y_MACHINE,
    CHROMA_SCAN_X_MACHINE,
]

N_SAMPLE   = 300


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tune_map_filename(
    qx: float,
    qy: float,
    xi_x: float | None = None,
    xi_y: float | None = None,
) -> str:
    return tune_map_filename(qx, qy, xi_x=xi_x, xi_y=xi_y)


def _case_dir(scan_type: str, case_name: str) -> str:
    return str(map_case_root_for_scan_type(scan_type, case_name))


def _plot_dir(scan_type: str, case_name: str) -> str:
    return str(plot_case_root_for_scan_type(scan_type, case_name))


def _legacy_tune_map_filename(qx: float, qy: float) -> str:
    return tune_map_filename(qx, qy, xi_x=None, xi_y=None)


def _scan_output_pdf_for_cfg(scan_cfg: dict, case_name: str) -> str:
    stem = f"tune_diagram_{case_name}"
    if is_chroma_scan_type(scan_cfg["type"]):
        stem = f"tune_diagram_{scan_cfg['label']}_{case_name}"
    suffix = scan_plot_suffix(scan_cfg)
    if suffix:
        name = f"{stem}_{suffix}.pdf"
    else:
        name = f"{stem}.pdf"
    return os.path.join(_plot_dir(scan_cfg["type"], case_name), name)


def _iter_scan_entries(scan_cfg: dict):
    yield from iter_scan_entries(scan_cfg)


# ──────────────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────────────

def load_maps(scan_cfg: dict, case_name: str) -> dict:
    """
    Load all TuneMaps for one (scan type, case) combination.

    Missing files are skipped with a warning so partially completed scans can
    still be plotted.
    """
    scan_type = scan_cfg["type"]
    directory = _case_dir(scan_type, case_name)
    maps = {}

    expected = 0
    for key, qx, qy, xi_x_label, xi_y_label in _iter_scan_entries(scan_cfg):
        expected += 1
        path = os.path.join(
            directory,
            _tune_map_filename(qx, qy, xi_x=xi_x_label, xi_y=xi_y_label),
        )
        if not os.path.exists(path) and scan_type in ("QxScan", "QyScan") and xi_x_label is not None:
            legacy_path = os.path.join(directory, _legacy_tune_map_filename(qx, qy))
            if os.path.exists(legacy_path):
                path = legacy_path
        if not os.path.exists(path):
            print(f"  [WARNING] Missing: {path}")
            continue
        maps[key] = TuneMap.load(path)

    print(f"[{scan_type} | {case_name}] Loaded {len(maps)}/{expected} maps.")
    return maps


# ──────────────────────────────────────────────────────────────────────────────
# Plotter
# ──────────────────────────────────────────────────────────────────────────────

def plot_tune_diagram(
    maps: dict,
    scan_cfg: dict,
    case_name: str,
    n_sample: int = N_SAMPLE,
) -> plt.Figure:
    """
    One tune diagram for a single (scan, case) combination.

    The viridis colormap follows the scanned parameter (Qx, Qy, xi_x, or xi_y).
    The coolwarm colormap shows δ along each trajectory.
    """
    if not maps:
        raise ValueError(
            f"No maps loaded for scan '{scan_cfg['type']}' and case '{case_name}'."
        )

    scan_type = scan_cfg["type"]

    spec = tune_diagram_spec(scan_cfg)
    scan_vals = spec["scan_vals"]
    cbar_label = spec["cbar_label"]
    td = TuneDiagram(
        qx0=spec["qx0"],
        qy0=spec["qy0"],
        half_range=spec["half_range"],
        max_order=MAX_ORDER,
        skew=True,
    )

    def scan_val_of(key):
        return scan_param_value(scan_cfg, key)

    fig, ax, (cax_d, cax_wp) = make_tune_diagram_figure(n_cbars=2)

    td.plot(ax=ax, show_working_point=False)
    ax.set_aspect("equal")

    wp_cmap = cm.viridis
    wp_norm = Normalize(vmin=min(scan_vals), vmax=max(scan_vals))
    d_cmap = "coolwarm"
    d_min = min(tm.delta_min for tm in maps.values())
    d_max = max(tm.delta_max for tm in maps.values())
    d_norm = Normalize(vmin=d_min, vmax=d_max)

    ordered_keys = [key for key, *_ in _iter_scan_entries(scan_cfg) if key in maps]
    last_lc = None

    for key in ordered_keys:
        tm = maps[key]
        color = wp_cmap(wp_norm(scan_val_of(key)))

        d_arr, qx_arr, qy_arr = tm.sample(n_sample)
        points = np.array([qx_arr, qy_arr]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        d_mid = 0.5 * (d_arr[:-1] + d_arr[1:])

        lc = LineCollection(
            segments,
            cmap=d_cmap,
            norm=d_norm,
            lw=2.0,
            zorder=3,
            alpha=0.85,
        )
        lc.set_array(d_mid)
        ax.add_collection(lc)
        last_lc = lc

        qx_wp, qy_wp = tm(0.0)
        ax.scatter(
            qx_wp,
            qy_wp,
            color=color,
            zorder=5,
            s=45,
            edgecolors="k",
            linewidths=0.5,
        )

    cbar_d = fig.colorbar(last_lc, cax=cax_d)
    d_ticks = np.linspace(d_min, d_max, 9)
    cbar_d.set_ticks(d_ticks)
    cbar_d.set_ticklabels([f"{v * 1e3:.1f}" for v in d_ticks])
    cbar_d.set_label(r"$\delta\ [10^{-3}]$", fontsize=16)
    cbar_d.ax.tick_params(labelsize=14)

    sm = cm.ScalarMappable(cmap=wp_cmap, norm=wp_norm)
    sm.set_array([])
    cbar_wp = fig.colorbar(sm, cax=cax_wp)
    wp_ticks = np.linspace(min(scan_vals), max(scan_vals), min(7, len(scan_vals)))
    cbar_wp.set_ticks(wp_ticks)
    cbar_wp.set_ticklabels([f"{v:.3f}" for v in wp_ticks])
    cbar_wp.set_label(cbar_label, fontsize=16)
    cbar_wp.ax.tick_params(labelsize=14)

    td.finalize(
        ax,
        extra_handles=[],
        xlabel=r"$Q_x$",
        ylabel=r"$Q_y$",
        legend_loc="upper left",
    )
    ax.set_title(f"Sweep trajectories — {scan_type} — {case_name}", fontsize=13)
    finalize_colorbar_heights(fig, ax, [cax_d, cax_wp])
    # Re-apply after canvas.draw() in finalize_colorbar_heights
    cbar_d.set_label(r"$\delta\ [10^{-3}]$", fontsize=16)
    cbar_d.ax.tick_params(labelsize=14)
    cbar_wp.set_label(cbar_label, fontsize=16)
    cbar_wp.ax.tick_params(labelsize=14)
    ax.set_xlabel(r"$Q_x$", fontsize=16)
    ax.set_ylabel(r"$Q_y$", fontsize=16)
    ax.tick_params(labelsize=14)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    for scan_cfg in ACTIVE_SCANS:
        scan_type = scan_cfg["type"]
        print(f"\n{'═' * 72}")
        print(f"Scan: {scan_type}")
        print("═" * 72)

        for case_name in MAP_CASES:
            print(f"\nLoading case: {case_name}")
            maps = load_maps(scan_cfg, case_name)
            if not maps:
                print(f"  No maps found — skipping plot for {scan_type}/{case_name}.")
                continue

            fig = plot_tune_diagram(maps, scan_cfg, case_name)
            fig_path = _scan_output_pdf_for_cfg(scan_cfg, case_name)
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"  Saved → {fig_path}")
            plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
