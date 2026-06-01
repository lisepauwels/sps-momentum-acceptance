from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "helper_functions" / "tune_diagram_helpers"))
sys.path.insert(0, str(REPO_ROOT / "helper_functions"))

from tune_diagram import TuneDiagram

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build combined slow-scan NAFF and dead-particle tune diagrams from saved outputs."
    )
    parser.add_argument("scan_dir", help="Saved scan directory, e.g. three_speed_scan_20260414")
    return parser.parse_args()


def load_map_df(scan_dir: Path) -> pd.DataFrame:
    pos_te = pd.read_parquet(scan_dir / "DPpos_slow" / "tune_estimate.parquet")
    neg_te = pd.read_parquet(scan_dir / "DPneg_slow" / "tune_estimate.parquet")
    return (
        pd.concat(
            [
                pos_te[["delta_center", "qx_map", "qy_map"]],
                neg_te[["delta_center", "qx_map", "qy_map"]],
            ],
            ignore_index=True,
        )
        .sort_values("delta_center")
        .drop_duplicates("delta_center")
    )


def tune_diagram_axes(map_df: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    center_idx = int(np.argmin(np.abs(map_df["delta_center"].to_numpy(dtype=float))))
    qx0 = float(map_df["qx_map"].iloc[center_idx])
    qy0 = float(map_df["qy_map"].iloc[center_idx])
    td = TuneDiagram(qx0=qx0, qy0=qy0, half_range=0.4, max_order=3, skew=True)
    fig, ax = td.plot(figsize=(8.5, 7.5), show_working_point=True)
    ax.plot(map_df["qx_map"], map_df["qy_map"], color="0.65", linewidth=1.8, label="Sweep trajectory")
    td.finalize(ax, extra_handles=None)
    return fig, ax


def plot_combined_naff(scan_dir: Path) -> Path:
    pos = pd.read_parquet(scan_dir / "DPpos_slow" / "tune_estimate.parquet")
    neg = pd.read_parquet(scan_dir / "DPneg_slow" / "tune_estimate.parquet")
    map_df = load_map_df(scan_dir)
    out = scan_dir / "combined_naff_tune_diagram_slow.png"

    fig, ax = tune_diagram_axes(map_df)
    turn_min = float(min(pos["window_center"].min(), neg["window_center"].min()))
    turn_max = float(max(pos["window_center"].max(), neg["window_center"].max()))

    scatter = ax.scatter(
        pos["qx_estimate_abs_full"].to_numpy(dtype=float),
        pos["qy_estimate_abs_full"].to_numpy(dtype=float),
        c=pos["window_center"].to_numpy(dtype=float),
        cmap="plasma",
        vmin=turn_min,
        vmax=turn_max,
        s=24,
        alpha=0.8,
        marker="o",
        label="DPpos slow NAFF",
        zorder=6,
    )
    ax.scatter(
        neg["qx_estimate_abs_full"].to_numpy(dtype=float),
        neg["qy_estimate_abs_full"].to_numpy(dtype=float),
        c=neg["window_center"].to_numpy(dtype=float),
        cmap="plasma",
        vmin=turn_min,
        vmax=turn_max,
        s=24,
        alpha=0.8,
        marker="s",
        label="DPneg slow NAFF",
        zorder=6,
    )
    ax.legend(loc="best", frameon=True)

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Window center [turn]")
    fig.suptitle("Combined NAFF tune estimates for DPpos/DPneg slow scan")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_combined_dead(scan_dir: Path) -> Path:
    pos = pd.read_parquet(scan_dir / "DPpos_slow" / "dead_particles.parquet")
    neg = pd.read_parquet(scan_dir / "DPneg_slow" / "dead_particles.parquet")
    map_df = load_map_df(scan_dir)
    out = scan_dir / "combined_dead_particle_tune_diagram_slow_absdelta.png"
    all_delta_abs = np.abs(
        np.concatenate(
            [pos["delta"].to_numpy(dtype=float), neg["delta"].to_numpy(dtype=float)]
        )
    )

    fig, ax = tune_diagram_axes(map_df)
    scatter = ax.scatter(
        pos["qx_estimate_without_errors"].to_numpy(dtype=float),
        pos["qy_estimate_without_errors"].to_numpy(dtype=float),
        c=np.abs(pos["delta"].to_numpy(dtype=float)),
        cmap="cool",
        vmin=float(np.min(all_delta_abs)),
        vmax=float(np.max(all_delta_abs)),
        s=24,
        alpha=0.8,
        marker="o",
        label="DPpos slow dead particles",
        zorder=6,
    )
    ax.scatter(
        neg["qx_estimate_without_errors"].to_numpy(dtype=float),
        neg["qy_estimate_without_errors"].to_numpy(dtype=float),
        c=np.abs(neg["delta"].to_numpy(dtype=float)),
        cmap="cool",
        vmin=float(np.min(all_delta_abs)),
        vmax=float(np.max(all_delta_abs)),
        s=24,
        alpha=0.8,
        marker="s",
        label="DPneg slow dead particles",
        zorder=6,
    )
    ax.legend(loc="best", frameon=True)

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label(r"Dead-particle $|\delta|$", fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    ax.tick_params(labelsize=14)
    fig.suptitle("Dead particles from DPpos and DPneg slow scan on one tune diagram")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    scan_dir = Path(args.scan_dir).resolve()
    naff_out = plot_combined_naff(scan_dir)
    dead_out = plot_combined_dead(scan_dir)
    print(naff_out)
    print(dead_out)


if __name__ == "__main__":
    main()
