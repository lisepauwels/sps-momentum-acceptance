from __future__ import annotations

import argparse
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "helper_functions" / "intensity_helpers"))

import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from TuneScan import (
    LINE_PATH,
    _case_dir,
    _plot_filename,
    _setup_cavities,
    _tune_map_filename,
    install_errors,
    optimise_tune_chroma,
    TRAJ_STEP,
)
from workflow_common import MAX_ORDER, plot_case_root_for_scan_type
from tune_diagram import SweepTrajectory, TuneDiagram


SCAN_CFG = {
    "label": "SinglePointMachine",
    "type": "SinglePoint",
    "tunes": [20.13],
    "fixed_qy": 20.18,
    "xi_x": 0.505,
    "xi_y": 0.300,
    "include_chroma_in_filename": True,
    "plot_suffix": "xix0.505_xiy0.300",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one machine-profile tune map and tune diagram."
    )
    parser.add_argument("--qx", type=float, default=20.13)
    parser.add_argument("--qy", type=float, default=20.18)
    parser.add_argument("--xi-x", type=float, default=0.505, dest="xi_x")
    parser.add_argument("--xi-y", type=float, default=0.300, dest="xi_y")
    parser.add_argument(
        "--case",
        choices=["WithErrors", "WithoutErrors", "Simplified"],
        default="WithoutErrors",
    )
    parser.add_argument("--delta-min", type=float, default=-1e-2)
    parser.add_argument("--delta-max", type=float, default=1e-2)
    return parser.parse_args()


def plot_single_point_tune_diagram(qx: float, qy: float, tm, xi_x: float, xi_y: float) -> plt.Figure:
    td = TuneDiagram(
        qx0=qx,
        qy0=qy,
        half_range=0.4,
        max_order=MAX_ORDER,
        skew=True,
    )
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    cax_d = ax.inset_axes([1.04, 0.0, 0.03, 1.0])

    td.plot(ax=ax, show_working_point=False)
    ax.set_aspect("equal")

    d_arr, qx_arr, qy_arr = tm.sample(300)
    points = np.array([qx_arr, qy_arr]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    d_mid = 0.5 * (d_arr[:-1] + d_arr[1:])
    d_norm = Normalize(vmin=tm.delta_min, vmax=tm.delta_max)

    lc = LineCollection(
        segments,
        cmap="coolwarm",
        norm=d_norm,
        lw=2.0,
        zorder=3,
        alpha=0.85,
    )
    lc.set_array(d_mid)
    ax.add_collection(lc)

    ax.scatter(
        qx,
        qy,
        color="k",
        zorder=5,
        s=45,
        edgecolors="k",
        linewidths=0.5,
    )

    cbar_d = fig.colorbar(lc, cax=cax_d)
    cbar_d.set_label(r"$\delta$", fontsize=11)
    d_ticks = np.linspace(tm.delta_min, tm.delta_max, 9)
    cbar_d.set_ticks(d_ticks)
    cbar_d.set_ticklabels([f"{v:.3f}" for v in d_ticks])

    td.finalize(
        ax,
        extra_handles=[
            mlines.Line2D(
                [], [],
                color="k",
                marker="o",
                ls="None",
                markersize=6,
                label=rf"Working point: $Q_x={qx:.3f}$, $Q_y={qy:.3f}$, $\xi_x={xi_x:.3f}$, $\xi_y={xi_y:.3f}$",
            ),
        ],
        xlabel=r"$Q_x$",
        ylabel=r"$Q_y$",
    )
    return fig


def main() -> None:
    args = parse_args()

    line_base = xt.load(LINE_PATH)
    _setup_cavities(line_base)
    tt_aper = line_base.get_aperture_table()

    scan_cfg = dict(SCAN_CFG)
    scan_cfg["tunes"] = [args.qx]
    scan_cfg["fixed_qy"] = args.qy
    scan_cfg["xi_x"] = args.xi_x
    scan_cfg["xi_y"] = args.xi_y
    scan_cfg["plot_suffix"] = f"xix{args.xi_x:.3f}_xiy{args.xi_y:.3f}"
    out_dir = Path(_case_dir(scan_cfg["label"], args.case))
    plot_dir = plot_case_root_for_scan_type(scan_cfg["label"], args.case)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    line = xt.load(LINE_PATH)
    _setup_cavities(line)

    use_errors = args.case == "WithErrors"
    simplified = args.case == "Simplified"
    if use_errors:
        install_errors(line, "all")
    optimise_tune_chroma(line, args.xi_x, args.xi_y, args.qx, args.qy)

    delta_range = (args.delta_min, args.delta_max)
    if simplified:
        tw0 = line.twiss4d()
        n_pts = max(10, int(round((args.delta_max - args.delta_min) / TRAJ_STEP)) + 1)
        sweep = SweepTrajectory.from_chroma(
            args.qx,
            args.qy,
            tw0.dqx,
            tw0.dqy,
            delta_range=delta_range,
            n_points=n_pts,
        )
    else:
        sweep = SweepTrajectory.from_twiss_scan(
            line,
            delta_range=delta_range,
            step=TRAJ_STEP,
            verbose=False,
        )

    tm = sweep.build_map()
    map_path = out_dir / _tune_map_filename(args.qx, args.qy, xi_x=args.xi_x, xi_y=args.xi_y)
    tm.save(map_path)
    print(f"Saved single-point map -> {map_path}")

    fig = plot_single_point_tune_diagram(args.qx, args.qy, tm, args.xi_x, args.xi_y)
    fig_path = plot_dir / _plot_filename("tune_diagram", scan_cfg, args.case)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved single-point diagram -> {fig_path}")


if __name__ == "__main__":
    main()
