"""
plot_md_intensity_single_point.py
==================================
Loss rate and intensity tune diagram plots for a single working point,
using the new fitted error model map.

Equivalent to PlotMDIntensity plot1 and plot2, but with one working point
so only one colorbar (delta, no Qx colorbar).

Output: prototyping/new_model_sweeps/plots/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "helper_functions" / "tune_diagram_helpers"))
sys.path.insert(0, str(REPO_ROOT / "helper_functions" / "intensity_helpers"))
sys.path.insert(0, str(REPO_ROOT / "helper_functions"))

from tune_diagram import TuneDiagram, TuneMap
from workflow_common import (
    QX_SCAN_MACHINE, MAX_ORDER,
    make_tune_diagram_figure, finalize_colorbar_heights,
)
from PlotMDIntensity import (
    load_rep_data,
    _common_delta, _interp_stack,
    SAVGOL_WINDOW, SAVGOL_POLYORDER,
    SIZE_PERCENTILE, DOT_SCALE,
)

QX, QY     = 20.13, 20.18
XI_X, XI_Y = 0.600, 0.600
MAP_PATH   = THIS_DIR / "maps" / "WithErrors" / f"tune_map_Qx{QX:.3f}_Qy{QY:.3f}_xix{XI_X:.3f}_xiy{XI_Y:.3f}.npz"
OUT_DIR    = THIS_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")


def _make_fig():
    fig, ax, (cax,) = make_tune_diagram_figure(n_cbars=1)
    td = TuneDiagram(qx0=QX, qy0=QY, half_range=0.4, max_order=MAX_ORDER, skew=True)
    td.plot(ax=ax, show_working_point=False)
    ax.set_aspect("equal")
    return fig, ax, cax, td


def _apply_labels(fig, ax, cax, cbar):
    finalize_colorbar_heights(fig, ax, [cax])
    cbar.ax.tick_params(labelsize=14)
    ax.set_xlabel(r"$Q_x$", fontsize=16)
    ax.set_ylabel(r"$Q_y$", fontsize=16)
    ax.tick_params(labelsize=14)


def plot1_loss_rate(tm: TuneMap, reps: list[dict]) -> plt.Figure:
    fig, ax, cax, td = _make_fig()

    delta_c = _common_delta(reps, tm=tm)
    _, loss_mean, _ = _interp_stack(reps, delta_c, "di_ddelta", transform=np.abs)
    qx_c, qy_c = tm(np.clip(delta_c * 1e-3, tm.delta_min, tm.delta_max))

    d_norm = Normalize(vmin=-1e-2, vmax=1e-2)

    # Trajectory colored by delta
    points   = np.array([qx_c, qy_c]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="coolwarm", norm=d_norm,
                        lw=1.0, zorder=2, alpha=0.35)
    lc.set_array(0.5 * (delta_c[:-1] + delta_c[1:]) * 1e-3)
    ax.add_collection(lc)

    # Scatter sized by loss rate
    size_ref = np.percentile(np.abs(loss_mean), SIZE_PERCENTILE)
    if size_ref == 0:
        size_ref = 1.0
    MIN_SIZE, MAX_SIZE = 4, DOT_SCALE

    def _size(arr):
        return MIN_SIZE + (MAX_SIZE - MIN_SIZE) * np.clip(np.abs(arr) / size_ref, 0, 1)

    n_bins = 15
    bins = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        norm_loss = np.abs(loss_mean) / size_ref
        mask = (norm_loss >= bins[i]) & (norm_loss < bins[i + 1])
        if mask.sum() == 0:
            continue
        a = float(np.clip(0.5 * (bins[i] + bins[i + 1]), 0.05, 0.9))
        ax.scatter(qx_c[mask], qy_c[mask],
                   c=delta_c[mask] * 1e-3, cmap="coolwarm", norm=d_norm,
                   s=_size(loss_mean[mask]), zorder=4, alpha=a,
                   edgecolors="none", linewidths=0)

    qx_wp, qy_wp = tm(0.0)
    ax.scatter(qx_wp, qy_wp, color="k", zorder=5, s=50, edgecolors="k", linewidths=0.5)

    cbar = fig.colorbar(lc, cax=cax)
    d_ticks = np.linspace(-1e-2, 1e-2, 9)
    cbar.set_ticks(d_ticks)
    cbar.set_ticklabels([f"{v:.3f}" for v in d_ticks])
    cbar.set_label(r"$\delta$", fontsize=16)

    td.finalize(ax, extra_handles=[], xlabel=r"$Q_x$", ylabel=r"$Q_y$")
    ax.set_title(f"Loss rate — WithErrors new model — Qx={QX} Qy={QY} xi={XI_X}/{XI_Y}", fontsize=12)
    _apply_labels(fig, ax, cax, cbar)
    cbar.set_label(r"$\delta$", fontsize=16)
    return fig


def plot2_intensity(tm: TuneMap, reps: list[dict]) -> plt.Figure:
    fig, ax, cax, td = _make_fig()

    delta_c = _common_delta(reps, tm=tm)
    _, int_mean, _ = _interp_stack(reps, delta_c, "intensity")
    qx_c, qy_c = tm(np.clip(delta_c * 1e-3, tm.delta_min, tm.delta_max))

    i_norm = Normalize(vmin=0, vmax=1)
    sc = ax.scatter(qx_c, qy_c, c=int_mean, cmap="RdYlGn", norm=i_norm,
                    s=18, zorder=4, linewidths=0)

    thresh_styles = {0.75: ("^", 12, "75%"), 0.50: ("D", 12, "50%"), 0.25: ("v", 12, "25%")}
    for thresh, (marker, ms, _) in thresh_styles.items():
        crossings = []
        for i in range(len(int_mean) - 1):
            if (int_mean[i] - thresh) * (int_mean[i+1] - thresh) <= 0:
                crossings.append(0.5 * (delta_c[i] + delta_c[i+1]) * 1e-3)
        for d_cross in crossings:
            qx_cross, qy_cross = tm(np.clip(d_cross, tm.delta_min, tm.delta_max))
            ax.scatter(qx_cross, qy_cross, marker=marker, s=ms,
                       color="k", zorder=6, linewidths=0.5, edgecolors="k")

    qx_wp, qy_wp = tm(0.0)
    ax.scatter(qx_wp, qy_wp, color="k", zorder=5, s=50, edgecolors="k", linewidths=0.5)

    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Normalised Intensity [-]", fontsize=16)

    thresh_handles = [
        mlines.Line2D([], [], marker=m, color="gray", ls="None",
                      markersize=4, markeredgecolor="k",
                      markeredgewidth=0.5, label=f"I = {lbl}")
        for _, (m, _, lbl) in thresh_styles.items()
    ]
    td.finalize(ax, extra_handles=thresh_handles, xlabel=r"$Q_x$", ylabel=r"$Q_y$")
    ax.set_title(f"Intensity — WithErrors new model — Qx={QX} Qy={QY} xi={XI_X}/{XI_Y}", fontsize=12)
    _apply_labels(fig, ax, cax, cbar)
    cbar.set_label("Normalised Intensity [-]", fontsize=16)
    return fig


def main():
    print(f"Loading map: {MAP_PATH}")
    tm = TuneMap.load(str(MAP_PATH))

    print(f"Loading MD data for Qx={QX} ...")
    reps = load_rep_data(QX_SCAN_MACHINE, QX)
    if reps is None:
        print("No MD data found — aborting.")
        return

    print(f"  {len(reps)} reps loaded.")

    fig1 = plot1_loss_rate(tm, reps)
    out1 = OUT_DIR / f"plot1_loss_rate_new_model_Qx{QX:.3f}_Qy{QY:.3f}_xix{XI_X:.3f}_xiy{XI_Y:.3f}.pdf"
    fig1.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {out1}")

    fig2 = plot2_intensity(tm, reps)
    out2 = OUT_DIR / f"plot2_intensity_new_model_Qx{QX:.3f}_Qy{QY:.3f}_xix{XI_X:.3f}_xiy{XI_Y:.3f}.pdf"
    fig2.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
