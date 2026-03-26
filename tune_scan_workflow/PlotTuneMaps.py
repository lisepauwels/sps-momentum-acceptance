"""
plot_tune_maps.py
=================
Loads pre-computed TuneMaps from SweepTrajectoryMaps/ and reproduces the
sweep trajectory tune-diagram plots.

This version is generic and supports all scan types produced by TuneScan.py:
    - QxScan
    - QyScan
    - ChromaScanX
    - ChromaScanY

It only loads and plots TuneMap files. It does not use measurements.

Expected directory structure:
    SweepTrajectoryMaps/
        QxScan/
            WithErrors/
            WithoutErrors/
            Simplified/
        QyScan/
            ...
        ChromaScanX/
            ...
        ChromaScanY/
            ...

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
sys.path.insert(0, '/Users/lisepauwels/sps_simulations/Studies/MomentumAcceptance/HelperFunctions')

from tune_diagram import TuneDiagram, TuneMap
from workflow_common import (
    CHROMA_SCAN_X,
    CHROMA_SCAN_Y,
    MAP_CASES,
    MAX_ORDER,
    OUTPUT_ROOT,
    QX_SCAN,
    QY_SCAN,
    colorbar_inset_positions,
    iter_scan_entries,
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
    CHROMA_SCAN_Y,
    CHROMA_SCAN_X,
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
    return os.path.join(OUTPUT_ROOT, scan_type, case_name)


def _scan_output_pdf(scan_type: str, case_name: str) -> str:
    return os.path.join(_case_dir(scan_type, case_name), f"tune_diagram_{case_name}.pdf")


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

    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    cax_d, cax_wp = [ax.inset_axes(pos) for pos in colorbar_inset_positions(2)]

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
    cbar_d.set_label(r"$\delta\ [10^{-3}]$", fontsize=11)
    d_ticks = np.linspace(d_min, d_max, 9)
    cbar_d.set_ticks(d_ticks)
    cbar_d.set_ticklabels([f"{v * 1e3:.1f}" for v in d_ticks])

    sm = cm.ScalarMappable(cmap=wp_cmap, norm=wp_norm)
    sm.set_array([])
    cbar_wp = fig.colorbar(sm, cax=cax_wp)
    cbar_wp.set_label(cbar_label, fontsize=11)
    wp_ticks = np.linspace(min(scan_vals), max(scan_vals), min(7, len(scan_vals)))
    cbar_wp.set_ticks(wp_ticks)
    cbar_wp.set_ticklabels([f"{v:.3f}" for v in wp_ticks])

    td.finalize(
        ax,
        extra_handles=[],
        xlabel=r"$Q_x$",
        ylabel=r"$Q_y$",
        legend_loc="upper left",
    )
    ax.set_title(f"Sweep trajectories — {scan_type} — {case_name}", fontsize=13)
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

        os.makedirs(os.path.join(OUTPUT_ROOT, scan_type), exist_ok=True)

        for case_name in MAP_CASES:
            print(f"\nLoading case: {case_name}")
            maps = load_maps(scan_cfg, case_name)
            if not maps:
                print(f"  No maps found — skipping plot for {scan_type}/{case_name}.")
                continue

            fig = plot_tune_diagram(maps, scan_cfg, case_name)
            fig_path = _scan_output_pdf(scan_type, case_name)
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"  Saved → {fig_path}")
            plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
