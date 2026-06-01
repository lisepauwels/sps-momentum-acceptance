"""
plot_qx_scan.py
===============
Plot all QxScan sweep trajectories for the new fitted error model.

Reads maps from:
    prototyping/new_model_sweeps/maps/WithErrors/

Output:
    prototyping/new_model_sweeps/plots/tune_diagram_QxScan_WithErrors.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "helper_functions" / "tune_diagram_helpers"))
sys.path.insert(0, str(REPO_ROOT / "helper_functions"))

from tune_diagram import TuneDiagram, TuneMap
from workflow_common import MAX_ORDER, make_tune_diagram_figure, finalize_colorbar_heights

QX_TUNES  = [
    20.07, 20.075, 20.08, 20.085, 20.09, 20.095,
    20.10, 20.105, 20.11, 20.115, 20.12, 20.125,
    20.13, 20.135, 20.14, 20.145, 20.15, 20.155,
    20.16, 20.165, 20.17, 20.175,
]
QY        = 20.18
XI_X      = 0.505
XI_Y      = 0.300
MAP_DIR   = THIS_DIR / "maps" / "WithErrors"
OUT_DIR   = THIS_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
N_SAMPLE  = 300


def load_maps() -> dict[float, TuneMap]:
    maps = {}
    for qx in QX_TUNES:
        fname = f"tune_map_Qx{qx:.3f}_Qy{QY:.3f}_xix{XI_X:.3f}_xiy{XI_Y:.3f}.npz"
        path = MAP_DIR / fname
        if path.exists():
            maps[qx] = TuneMap.load(str(path))
        else:
            print(f"  [WARNING] Missing: {path}")
    print(f"Loaded {len(maps)}/{len(QX_TUNES)} maps.")
    return maps


def plot_qx_scan(maps: dict[float, TuneMap]) -> plt.Figure:
    if not maps:
        raise ValueError("No maps loaded.")

    qx_ref = min(maps, key=lambda q: abs(q - 20.13))
    tm_ref = maps[qx_ref]
    qx0, qy0 = tm_ref(0.0)

    td = TuneDiagram(qx0=qx0, qy0=qy0, half_range=0.5, max_order=MAX_ORDER, skew=True)
    fig, ax, (cax_d, cax_wp) = make_tune_diagram_figure(n_cbars=2)
    td.plot(ax=ax, show_working_point=False)
    ax.set_aspect("equal")

    wp_cmap = cm.viridis
    wp_norm = Normalize(vmin=min(maps), vmax=max(maps))

    d_min = min(tm.delta_min for tm in maps.values())
    d_max = max(tm.delta_max for tm in maps.values())
    d_norm = Normalize(vmin=d_min, vmax=d_max)

    last_lc = None
    for qx in sorted(maps):
        tm = maps[qx]
        color = wp_cmap(wp_norm(qx))
        d_arr, qx_arr, qy_arr = tm.sample(N_SAMPLE)
        points   = np.array([qx_arr, qy_arr]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        d_mid    = 0.5 * (d_arr[:-1] + d_arr[1:])
        lc = LineCollection(segments, cmap="coolwarm", norm=d_norm,
                            lw=2.0, zorder=3, alpha=0.85)
        lc.set_array(d_mid)
        ax.add_collection(lc)
        last_lc = lc
        qx_wp, qy_wp = tm(0.0)
        ax.scatter(qx_wp, qy_wp, color=color, zorder=5, s=45,
                   edgecolors="k", linewidths=0.5)

    cbar_d = fig.colorbar(last_lc, cax=cax_d)
    d_ticks = np.linspace(d_min, d_max, 9)
    cbar_d.set_ticks(d_ticks)
    cbar_d.set_ticklabels([f"{v * 1e3:.1f}" for v in d_ticks])
    cbar_d.set_label(r"$\delta\ [10^{-3}]$", fontsize=16)
    cbar_d.ax.tick_params(labelsize=14)

    sm = cm.ScalarMappable(cmap=wp_cmap, norm=wp_norm)
    sm.set_array([])
    cbar_wp = fig.colorbar(sm, cax=cax_wp)
    wp_ticks = np.linspace(min(maps), max(maps), min(7, len(maps)))
    cbar_wp.set_ticks(wp_ticks)
    cbar_wp.set_ticklabels([f"{v:.3f}" for v in wp_ticks])
    cbar_wp.set_label(r"$Q_x$", fontsize=16)
    cbar_wp.ax.tick_params(labelsize=14)

    td.finalize(ax, extra_handles=[], xlabel=r"$Q_x$", ylabel=r"$Q_y$",
                legend_loc="upper left")
    ax.set_title("Sweep trajectories — QxScan — WithErrors (new model)", fontsize=13)

    finalize_colorbar_heights(fig, ax, [cax_d, cax_wp])
    cbar_d.set_label(r"$\delta\ [10^{-3}]$", fontsize=16)
    cbar_d.ax.tick_params(labelsize=14)
    cbar_wp.set_label(r"$Q_x$", fontsize=16)
    cbar_wp.ax.tick_params(labelsize=14)
    ax.set_xlabel(r"$Q_x$", fontsize=16)
    ax.set_ylabel(r"$Q_y$", fontsize=16)
    ax.tick_params(labelsize=14)

    return fig


def main() -> None:
    maps = load_maps()
    if not maps:
        print("Nothing to plot.")
        return
    fig = plot_qx_scan(maps)
    out = OUT_DIR / "tune_diagram_QxScan_WithErrors.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
