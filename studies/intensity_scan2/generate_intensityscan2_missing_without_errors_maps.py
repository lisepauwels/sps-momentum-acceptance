from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import xtrack as xt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "helper_functions" / "tune_diagram_helpers"))

from TuneScan import (
    LINE_PATH,
    _build_and_save,
    _case_dir,
    _plot_filename,
    _setup_cavities,
    _setup_dirs,
    plot_tune_diagram,
)


DEFAULT_STUDY_ROOT = Path(
    "/Users/lisepauwels/sps_simulations/Studies/MomentumAcceptance/IntensityScan2"
)
DEFAULT_QX = 20.13
DEFAULT_QY = 20.18
DEFAULT_FIXED_XI_Y = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate IntensityScan2 tune maps for a selected optics case. "
            "Chromaticity points are read from combined_linear_*.json.gzip."
        )
    )
    parser.add_argument(
        "--study-root",
        type=Path,
        default=DEFAULT_STUDY_ROOT,
        help="IntensityScan2 directory containing study_results/.",
    )
    parser.add_argument("--qx", type=float, default=DEFAULT_QX)
    parser.add_argument("--qy", type=float, default=DEFAULT_QY)
    parser.add_argument(
        "--fixed-xi-y",
        type=float,
        default=DEFAULT_FIXED_XI_Y,
        help="Fixed vertical normalised chromaticity for the horizontal chroma scan.",
    )
    parser.add_argument(
        "--coupled-xi",
        action="store_true",
        help="Use xi_x = xi_y = study chroma for every generated map.",
    )
    parser.add_argument(
        "--case",
        choices=["WithErrors", "WithoutErrors"],
        default="WithoutErrors",
        help="Optics case to generate.",
    )
    parser.add_argument(
        "--error-variant",
        choices=[
            "dipole_b3",
            "dipole_b5",
            "dipole_b3b5",
            "quadrupole_b4",
            "quadrupole_b6",
            "quadrupole_b4b6",
            "dipole_b3_quadrupole_b4",
            "all",
        ],
        default="all",
        help="Error component to install when --case WithErrors is used.",
    )
    parser.add_argument(
        "--with-diagram",
        action="store_true",
        help="Also regenerate a summary tune diagram after the missing maps are built.",
    )
    return parser.parse_args()


def load_intensityscan2_chromas(study_root: Path) -> list[float]:
    study_results = study_root / "study_results"
    chromas: set[float] = set()
    for path in sorted(study_results.glob("combined_linear_*.json.gzip")):
        stem = path.name.replace(".json.gzip", "")
        chromas.add(float(stem.split("_")[-1]))
    if not chromas:
        raise FileNotFoundError(f"No combined_linear_*.json.gzip files found in {study_results}")
    return sorted(chromas)


def build_scan_cfg(
    chromas: list[float],
    qx: float,
    qy: float,
    fixed_xi_y: float,
    *,
    coupled_xi: bool,
) -> dict:
    xi_pairs = [(c, c) for c in chromas] if coupled_xi else [(c, fixed_xi_y) for c in chromas]
    suffix = "intensityscan2_coupled_xi" if coupled_xi else f"intensityscan2_xiy{fixed_xi_y:.3f}"
    return {
        "label": "IntensityScan2Qx",
        "type": "ChromaScanX",
        "xi_pairs": xi_pairs,
        "fixed_qx": qx,
        "fixed_qy": qy,
        "plot_suffix": suffix,
    }


def main() -> None:
    args = parse_args()
    chromas = load_intensityscan2_chromas(args.study_root.resolve())
    scan_cfg = build_scan_cfg(
        chromas,
        args.qx,
        args.qy,
        args.fixed_xi_y,
        coupled_xi=args.coupled_xi,
    )
    scan_type = scan_cfg["type"]

    _setup_dirs(scan_type)

    line_base = xt.load(LINE_PATH)
    _setup_cavities(line_base)
    tt_aper = line_base.get_aperture_table()

    if args.coupled_xi:
        print(
            "Generating IntensityScan2 coupled-chroma maps with "
            f"Qx={args.qx:.3f}, Qy={args.qy:.3f}, xi_x = xi_y = study chroma"
        )
    else:
        print(
            "Generating IntensityScan2 horizontal-chroma maps with "
            f"Qx={args.qx:.3f}, Qy={args.qy:.3f}, fixed xi_y={args.fixed_xi_y:.3f}"
        )
    print(f"Study chroma points: {len(chromas)} values from {chromas[0]:.3f} to {chromas[-1]:.3f}")

    case_name = args.case
    if args.case == "WithErrors" and args.error_variant != "all":
        case_name = f"{args.case}_{args.error_variant}"
    line = xt.load(LINE_PATH)
    _setup_cavities(line)

    maps = {}
    for xi_x in chromas:
        xi_y = xi_x if args.coupled_xi else args.fixed_xi_y
        key = (xi_x, xi_y)
        print(f"\n  Qx={args.qx:.3f}  Qy={args.qy:.3f}  xi_x={xi_x:.3f}  xi_y={xi_y:.3f}")
        maps[key] = _build_and_save(
            line,
            tt_aper,
            args.qx,
            args.qy,
            xi_x,
            xi_y,
            case_name,
            scan_type,
            use_errors=case_name == "WithErrors",
            simplified=False,
            xi_x_label=xi_x,
            xi_y_label=xi_y,
            error_variant=args.error_variant,
        )

    if args.with_diagram:
        fig = plot_tune_diagram(maps, scan_cfg, case_name)
        fig_path = os.path.join(
            _case_dir(scan_type, case_name),
            _plot_filename("tune_diagram", scan_cfg, case_name),
        )
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved diagram -> {fig_path}")


if __name__ == "__main__":
    main()
