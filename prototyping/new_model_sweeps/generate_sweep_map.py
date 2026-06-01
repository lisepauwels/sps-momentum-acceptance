"""
generate_sweep_map.py
=====================
Generate sweep trajectory TuneMaps for the new fitted error model.

Error values come from the new fit (replacing the old nominal install_errors).
Maps are saved locally under:
    prototyping/new_model_sweeps/maps/WithErrors/
    prototyping/new_model_sweeps/maps/WithoutErrors/

Usage
-----
    # Single nominal point:
    python generate_sweep_map.py --qx 20.13 --qy 20.18 --xi_x 0.505 --xi_y 0.3

    # Full QxScan (after verifying the nominal point):
    python generate_sweep_map.py --scan qx
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import xtrack as xt

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "helper_functions" / "tune_diagram_helpers"))
sys.path.insert(0, str(REPO_ROOT / "helper_functions"))

from load_paths import get_path
from tune_diagram import SweepTrajectory, TuneMap
from workflow_common import QX_SCAN_MACHINE

MAP_ROOT   = THIS_DIR / "maps"
LINE_PATH  = get_path(
    "line_with_ap_path",
    default=str(Path(get_path("legacy_workspace_root",
                              default="/Users/lisepauwels/sps_simulations"))
                / "injection_lines"
                / "sps_with_aperture_inj_q20_beam_sagitta4.json"),
)

DELTA_SCAN_MAX  = 1e-2
DELTA_SCAN_NPTS = 20
TRAJ_STEP       = 1e-3


def install_new_model_errors(line) -> None:
    env = line.env
    b1, b2, b3, b4, b5, b6 = 1, 1, 1, 1, 1, 1
    tte = env.elements.get_table()
    mask_rbends = tte.element_type == "RBend"
    mask_quads  = tte.element_type == "Quadrupole"

    mba = tte.rows[mask_rbends].rows["mba.*"].name
    mbb = tte.rows[mask_rbends].rows["mbb.*"].name
    qf  = tte.rows[mask_quads].rows["qf.*"].name
    qd  = tte.rows[mask_quads].rows["qd.*"].name

    env.vars["qph_setvalue"] = 0.0
    env.vars["qpv_setvalue"] = 0.0

    for nn in mba:
        env[nn].knl = np.array([b1*0., b2*0., b3*8.92863062376558e-06,   b4*0., b5*-1.6271899631371554,  b6*0.])
    for nn in mbb:
        env[nn].knl = np.array([b1*0., b2*0., b3*-2.606543641240525e-05, b4*0., b5*-1.1800693058580867,  b6*0.])
    for nn in qf:
        env[nn].knl = np.array([b1*0., b2*0., b3*0., b4*0.025520870077609926,  b5*0., b6*-114.8869977018476])
    for nn in qd:
        env[nn].knl = np.array([b1*0., b2*0., b3*0., b4*-0.029783873625363777, b5*0., b6*-749.9978975055122])


def optimise_tune_chroma(line, xi_x: float, xi_y: float,
                          qx: float, qy: float) -> None:
    env = line.env
    env.vars["qph_setvalue"] = 0.0
    env.vars["qpv_setvalue"] = 0.0
    line.match(
        method="6d",
        vary=[
            xt.VaryList(["kqf0", "kqd0"],                 step=1e-8, tag="quad"),
            xt.VaryList(["qph_setvalue", "qpv_setvalue"],  step=1e-4, tag="sext"),
        ],
        targets=[
            xt.TargetSet(qx=qx,       qy=qy,       tol=1e-6, tag="tune"),
            xt.TargetSet(dqx=xi_x*qx, dqy=xi_y*qy, tol=1e-2, tag="chrom"),
        ],
    )


def _setup_cavities(line) -> None:
    _, cavity_names = line.get_elements_of_type(xt.Cavity)
    for name in cavity_names:
        line[name].frequency = 200e6
        line[name].lag       = 180
        line[name].voltage   = 0
    line["actcse.31632"].voltage = 3.0e6


def build_map(qx: float, qy: float, xi_x: float, xi_y: float,
              case: str, tt_aper) -> Path:
    """Build and save one TuneMap. Returns the output path."""
    out_dir = MAP_ROOT / case
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"tune_map_Qx{qx:.3f}_Qy{qy:.3f}_xix{xi_x:.3f}_xiy{xi_y:.3f}.npz"
    out   = out_dir / fname

    if out.exists():
        print(f"  Exists → {out} (skipping)")
        return out

    print(f"\nBuilding {case}: Qx={qx:.3f} Qy={qy:.3f} xi_x={xi_x:.3f} xi_y={xi_y:.3f}")
    line = xt.load(LINE_PATH)
    _setup_cavities(line)

    if case == "WithErrors":
        install_new_model_errors(line)

    optimise_tune_chroma(line, xi_x, xi_y, qx, qy)

    delta_pos = SweepTrajectory.find_delta_limit(line, +1, tt_aper,
                    max_delta_scan=DELTA_SCAN_MAX, n_scan_points=DELTA_SCAN_NPTS)
    delta_neg = SweepTrajectory.find_delta_limit(line, -1, tt_aper,
                    max_delta_scan=DELTA_SCAN_MAX, n_scan_points=DELTA_SCAN_NPTS)
    print(f"  δ range: [{delta_neg:.4g}, {delta_pos:.4g}]")

    sweep = SweepTrajectory.from_twiss_scan(
        line, delta_range=(delta_neg, delta_pos), step=TRAJ_STEP, verbose=False,
    )
    tm = sweep.build_map()
    tm.save(str(out))
    print(f"  Saved → {out}")
    return out


QX_SCAN_TUNES = [
    20.07, 20.075, 20.08, 20.085, 20.09, 20.095,
    20.10, 20.105, 20.11, 20.115, 20.12, 20.125,
    20.13, 20.135, 20.14, 20.145, 20.15, 20.155,
    20.16, 20.165, 20.17, 20.175,
]
QX_SCAN_QY    = 20.18
QX_SCAN_XI_X  = 0.505
QX_SCAN_XI_Y  = 0.300


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qx",   type=float, default=20.13)
    parser.add_argument("--qy",   type=float, default=20.18)
    parser.add_argument("--xi_x", type=float, default=0.505)
    parser.add_argument("--xi_y", type=float, default=0.300)
    parser.add_argument("--cases", nargs="+",
                        default=["WithErrors"],
                        choices=["WithErrors", "WithoutErrors"])
    parser.add_argument("--scan", choices=["qx"],
                        help="Run full QxScan instead of a single point")
    args = parser.parse_args()

    print(f"Loading line from {LINE_PATH} ...")
    line_base = xt.load(LINE_PATH)
    _setup_cavities(line_base)
    tt_aper = line_base.get_aperture_table()

    if args.scan == "qx":
        points = [(qx, QX_SCAN_QY, QX_SCAN_XI_X, QX_SCAN_XI_Y)
                  for qx in QX_SCAN_TUNES]
    else:
        points = [(args.qx, args.qy, args.xi_x, args.xi_y)]

    for qx, qy, xi_x, xi_y in points:
        for case in args.cases:
            build_map(qx, qy, xi_x, xi_y, case, tt_aper)

    print("\nDone.")


if __name__ == "__main__":
    main()
