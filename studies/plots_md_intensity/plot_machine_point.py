"""
plot_machine_point.py
======================
Regenerate sweep-trajectory and MD-intensity plots for the machine
chromaticity point (xi_x=0.505, xi_y=0.3), WithoutErrors case only.

Outputs
-------
  sps-chromaticity-maps/plots/qx_scan/without_errors/
      tune_diagram_WithoutErrors_xix0.505_xiy0.300.pdf

  studies/plots_md_intensity/QxScan/WithoutErrors/
      plot1_tune_diagram_sized_xix0.505_xiy0.300.pdf
      plot2_tune_diagram_intensity_xix0.505_xiy0.300.pdf
      plot3_loss_vs_delta_xix0.505_xiy0.300.pdf
      plot4_tunes_vs_delta_xix0.505_xiy0.300.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "helper_functions" / "tune_diagram_helpers"))
sys.path.insert(0, str(REPO_ROOT / "helper_functions" / "intensity_helpers"))
sys.path.insert(0, str(REPO_ROOT / "helper_functions"))

from workflow_common import QX_SCAN_MACHINE, MAP_CASES
from PlotTuneMaps import load_maps, plot_tune_diagram, _scan_output_pdf_for_cfg
from PlotMDIntensity import (
    OUTPUT_ROOT as MD_OUTPUT_ROOT,
    MAP_CASES as MD_MAP_CASES,
    _scan_keys_and_labels,
    _plot_filename,
    load_tune_map,
    load_rep_data,
    scan_param_value,
    plot1_sized_by_loss,
    plot2_coloured_by_intensity,
    plot3_loss_vs_delta,
    plot4_tunes_vs_delta,
)

SCAN       = QX_SCAN_MACHINE
CASE       = "WithoutErrors"


def run_sweep_trajectory() -> None:
    print("=== Sweep trajectory ===")
    maps = load_maps(SCAN, CASE)
    if not maps:
        print("  No maps found — skipping.")
        return
    fig = plot_tune_diagram(maps, SCAN, CASE)
    out_dir = REPO_ROOT / "sps-chromaticity-maps" / "plots" / "single_point_machine"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"tune_diagram_{CASE}_xix{SCAN['xi_x']:.3f}_xiy{SCAN['xi_y']:.3f}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def run_md_intensity() -> None:
    print("\n=== MD intensity plots ===")
    keys, scan_vals, cbar_label = _scan_keys_and_labels(SCAN)
    out_dir = MD_OUTPUT_ROOT / SCAN["label"] / CASE
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data = {}
    for key in keys:
        print(f"  Loading {key} ...", end=" ")
        tm   = load_tune_map(SCAN, key, CASE)
        reps = load_rep_data(SCAN, key)
        if tm is None or reps is None:
            all_data[key] = None
            print("skipped.")
        else:
            all_data[key] = {"tm": tm, "reps": reps, "scan_val": scan_param_value(SCAN, key)}
            print("ok.")

    n_loaded = sum(1 for v in all_data.values() if v is not None)
    print(f"  {n_loaded}/{len(keys)} loaded.")
    if n_loaded == 0:
        print("  Nothing to plot.")
        return

    args = (all_data, scan_vals, CASE, cbar_label, SCAN)
    for fn, stem in [
        (plot1_sized_by_loss,        "plot1_tune_diagram_sized"),
        (plot2_coloured_by_intensity, "plot2_tune_diagram_intensity"),
        (plot3_loss_vs_delta,         "plot3_loss_vs_delta"),
        (plot4_tunes_vs_delta,        "plot4_tunes_vs_delta"),
    ]:
        fig = fn(*args)
        fname = _plot_filename(SCAN, stem)
        fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


if __name__ == "__main__":
    run_sweep_trajectory()
    run_md_intensity()
    print("\nDone.")
