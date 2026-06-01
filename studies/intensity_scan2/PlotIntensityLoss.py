"""
PlotIntensityLoss.py
====================
Generate only the intensity-loss-vs-delta summary plot for the tune-scan
workflow.

This is a thin wrapper around the data-loading and plotting logic already
implemented in PlotMDIntensity.py. It saves one PDF per scan type and optics
case:

    plot3_loss_vs_delta.pdf

for:

    - QxScan
    - QyScan
    - ChromaScanX
    - ChromaScanY
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "helper_functions" / "tune_diagram_helpers"))

from PlotMDIntensity import (
    MAP_CASES,
    OUTPUT_ROOT,
    SCANS,
    _scan_keys_and_labels,
    load_rep_data,
    load_tune_map,
    plot3_loss_vs_delta,
    scan_plot_suffix,
    scan_param_value,
)


def _plot_filename(scan_cfg: dict, stem: str) -> str:
    suffix = scan_plot_suffix(scan_cfg)
    if suffix:
        return f"{stem}_{suffix}.pdf"
    return f"{stem}.pdf"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for scan in SCANS:
        scan_label = scan["label"]
        keys, scan_vals, cbar_label = _scan_keys_and_labels(scan)

        print(f"\n{'=' * 60}")
        print(f"Scan: {scan_label}")
        print("=" * 60)

        for map_case in MAP_CASES:
            print(f"\n  Optics case: {map_case}")
            out_dir = OUTPUT_ROOT / scan_label / map_case
            out_dir.mkdir(parents=True, exist_ok=True)

            all_data = {}
            for key in keys:
                print(f"    Loading {key} ...", end=" ")
                tm = load_tune_map(scan, key, map_case)
                reps = load_rep_data(scan, key)
                if tm is None or reps is None:
                    all_data[key] = None
                    print("skipped.")
                else:
                    all_data[key] = {
                        "tm": tm,
                        "reps": reps,
                        "scan_val": scan_param_value(scan, key),
                    }
                    print("ok.")

            n_loaded = sum(1 for v in all_data.values() if v is not None)
            print(f"    -> {n_loaded}/{len(keys)} loaded.")
            if n_loaded == 0:
                print("    Nothing to plot -- skipping.")
                continue

            fig = plot3_loss_vs_delta(all_data, scan_vals, map_case, cbar_label, scan)
            fname = _plot_filename(scan, "plot3_loss_vs_delta")
            fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved {fname}")

    print(f"\nDone. Plots saved under '{OUTPUT_ROOT}/'")


if __name__ == "__main__":
    main()
