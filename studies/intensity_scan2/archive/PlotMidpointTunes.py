from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "helper_functions" / "tune_diagram_helpers"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PlotMDIntensity import load_tune_map
from PlotScanSummaries import _common_delta, _interp_stack, load_side_reps
from workflow_common import (
    CHROMA_SCAN_X_MACHINE,
    CHROMA_SCAN_Y_MACHINE,
    scan_keys_and_labels,
    scan_param_value,
    scan_plot_suffix,
)


OUTPUT_ROOT = Path("summary_plots_midpoint_tunes")
CASES = ["WithoutErrors", "WithErrors"]
PLANES = ["DPpos", "DPneg"]
CASE_STYLES = {
    "WithoutErrors": {"color": "royalblue"},
    "WithErrors": {"color": "crimson"},
}
PLANE_MARKERS = {
    "DPpos": "o",
    "DPneg": "s",
}


def midpoint_delta_for(scan_cfg: dict, key, plane: str) -> float:
    reps = load_side_reps(scan_cfg, key, plane)
    if reps is None:
        return np.nan
    delta_c = _common_delta(reps)
    if delta_c is None:
        return np.nan
    _, int_mean, _ = _interp_stack(reps, delta_c, "intensity")
    midpoint = safe_interpolate_percentile_val(delta_c, int_mean, percentile=0.5)
    return np.nan if midpoint is None else float(midpoint)


def safe_interpolate_percentile_val(xvals, yvals, percentile=0.5):
    xvals = np.asarray(xvals, dtype=float)
    yvals = np.asarray(yvals, dtype=float)
    below = np.where(yvals <= percentile)[0]
    if below.size == 0:
        return None
    idx_below = int(below[0])
    if idx_below == 0:
        return float(xvals[0])
    idx_above = idx_below - 1
    return float(
        np.interp(
            percentile,
            [yvals[idx_above], yvals[idx_below]],
            [xvals[idx_above], xvals[idx_below]],
        )
    )


def build_midpoint_tunes(scan_cfg: dict) -> dict[str, dict]:
    keys, _, _ = scan_keys_and_labels(scan_cfg)
    result: dict[str, dict] = {case: {} for case in CASES}
    for case in CASES:
        for key in keys:
            midpoint_by_plane = {
                plane: midpoint_delta_for(scan_cfg, key, plane)
                for plane in PLANES
            }
            tm = load_tune_map(scan_cfg, key, case)
            result[case][key] = {
                "delta": midpoint_by_plane,
                "qx": {plane: np.nan for plane in PLANES},
                "qy": {plane: np.nan for plane in PLANES},
            }
            if tm is None:
                continue
            for plane, delta_val in midpoint_by_plane.items():
                if np.isnan(delta_val):
                    continue
                try:
                    qx_val, qy_val = tm(float(delta_val))
                except ValueError:
                    continue
                result[case][key]["qx"][plane] = float(qx_val)
                result[case][key]["qy"][plane] = float(qy_val)
    return result


def make_plot(scan_cfg: dict, tune_axis: str, tune_label: str) -> Path:
    data = build_midpoint_tunes(scan_cfg)
    keys, _, cbar_label = scan_keys_and_labels(scan_cfg)

    fig, ax = plt.subplots(figsize=(8, 6))
    for case in CASES:
        color = CASE_STYLES[case]["color"]
        for plane in PLANES:
            xs = np.array([scan_param_value(scan_cfg, key) for key in keys], dtype=float)
            ys = np.array([data[case][key][tune_axis][plane] for key in keys], dtype=float)
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle="-" if plane == "DPpos" else "--",
                marker=PLANE_MARKERS[plane],
                markersize=5,
                label=f"{case} - {plane}",
            )

    ax.grid(alpha=0.3)
    ax.set_xlabel(cbar_label)
    ax.set_ylabel(tune_label)
    ax.legend(ncols=2, fontsize=10, frameon=True)
    fig.tight_layout()

    out_dir = OUTPUT_ROOT / scan_cfg["label"]
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = scan_plot_suffix(scan_cfg)
    stem = f"midpoint_{tune_axis}_vs_xi_compare_errors"
    if suffix:
        stem = f"{stem}_{suffix}"
    out_path = out_dir / f"{stem}.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for scan_cfg in [CHROMA_SCAN_Y_MACHINE, CHROMA_SCAN_X_MACHINE]:
        qx_path = make_plot(scan_cfg, "qx", r"$Q_x(\delta_{50\%})$")
        qy_path = make_plot(scan_cfg, "qy", r"$Q_y(\delta_{50\%})$")
        print(f"Saved {qx_path}")
        print(f"Saved {qy_path}")


if __name__ == "__main__":
    main()
