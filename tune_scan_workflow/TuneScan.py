"""
tune_scan.py
============
Compute aperture-limited sweep trajectories and TuneMaps for three scan types:

    QxScan    — scan Qx at fixed Qy, fixed xi
    QyScan    — scan Qy at fixed Qx, fixed xi
    ChromaScan — scan xi_x (and xi_y) at fixed Qx, Qy

For each scan type, three optics cases are produced:
    WithErrors    — line with magnet errors installed, fully matched
    WithoutErrors — bare line (no errors), fully matched
    Simplified    — from_chroma linear approximation (no twiss scan)

Directory structure:
    SweepTrajectoryMaps/
        QxScan/
            WithErrors/      tune_map_Qx<XX.XXX>_Qy<YY.YYY>.npz
            WithoutErrors/   ...
            Simplified/      ...
        QyScan/
            WithErrors/      tune_map_Qx<XX.XXX>_Qy<YY.YYY>.npz
            ...
        ChromaScan/
            WithErrors/      tune_map_Qx<XX.XXX>_Qy<YY.YYY>_xi<ZZ.ZZZ>.npz
            ...

One tune diagram per (scan type × optics case) is saved alongside the maps.

Usage
-----
    python tune_scan.py

To select which scans to run, edit ACTIVE_SCANS at the bottom of the
configuration section.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

import xtrack as xt

import sys
sys.path.insert(0, '/Users/lisepauwels/sps_simulations/Studies/MomentumAcceptance/HelperFunctions')

from tune_diagram import TuneDiagram, SweepTrajectory, TuneMap
from workflow_common import (
    CHROMA_SCAN_X,
    CHROMA_SCAN_Y,
    MAP_CASES,
    MAX_ORDER,
    OUTPUT_ROOT,
    QX_SCAN,
    QY_SCAN,
    scan_param_value,
    tune_diagram_spec,
    tune_map_filename,
)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

LINE_PATH = "../../injection_lines/sps_with_aperture_inj_q20_beam_sagitta4.json"

# ── Active scans ──────────────────────────────────────────────────────────────
# Comment out any scan you don't want to run.
ACTIVE_SCANS = [
    # QX_SCAN,
    # QY_SCAN,
    CHROMA_SCAN_Y,
    # CHROMA_SCAN_X,
]

# δ scan parameters for find_delta_limit
DELTA_SCAN_MAX  = 1e-2
DELTA_SCAN_NPTS = 20

# trajectory step for from_twiss_scan
TRAJ_STEP = 1e-3


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def install_errors(line, error_variant_name: str) -> None:
    env = line.env
    error_variants = {
        "dipole_b3":               [0, 0, 1, 0, 0, 0],
        "dipole_b5":               [0, 0, 0, 0, 1, 0],
        "dipole_b3b5":             [0, 0, 1, 0, 1, 0],
        "quadrupole_b4":           [0, 0, 0, 1, 0, 0],
        "quadrupole_b6":           [0, 0, 0, 0, 0, 1],
        "quadrupole_b4b6":         [0, 0, 0, 1, 0, 1],
        "dipole_b3_quadrupole_b4": [0, 0, 1, 1, 0, 0],
        "all":                     [0, 0, 1, 1, 1, 1],
    }
    b1, b2, b3, b4, b5, b6 = error_variants[error_variant_name]
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
        env[nn].knl = np.array([b1*0., b2*0., b3*2.12e-3,   b4*0.,       b5*-5.74,  b6*0.])
    for nn in mbb:
        env[nn].knl = np.array([b1*0., b2*0., b3*-3.19e-3,  b4*0.,       b5*-5.10,  b6*0.])
    for nn in qf:
        env[nn].knl = np.array([b1*0., b2*0., b3*0.,        b4*0.75e-1,  b5*0.,     b6*-0.87e3])
    for nn in qd:
        env[nn].knl = np.array([b1*0., b2*0., b3*0.,        b4*-2.03e-1, b5*0.,     b6*2.04e3])


def optimise_tune_chroma(line, xi_x: float, xi_y: float,
                          qx: float, qy: float) -> None:
    env = line.env
    env.vars["qph_setvalue"] = 0.0
    env.vars["qpv_setvalue"] = 0.0
    line.match(
        method="6d",
        vary=[
            xt.VaryList(["kqf0", "kqd0"],                step=1e-8, tag="quad"),
            xt.VaryList(["qph_setvalue", "qpv_setvalue"], step=1e-4, tag="sext"),
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


def _tune_map_filename(qx: float, qy: float,
                       xi_x: float | None = None,
                       xi_y: float | None = None) -> str:
    return tune_map_filename(qx, qy, xi_x=xi_x, xi_y=xi_y)


def _case_dir(scan_type: str, case_name: str) -> str:
    return os.path.join(OUTPUT_ROOT, scan_type, case_name)


def _setup_dirs(scan_type: str) -> None:
    for case in MAP_CASES:
        os.makedirs(_case_dir(scan_type, case), exist_ok=True)
    print(f"Directories ready under '{OUTPUT_ROOT}/{scan_type}/'")


# ──────────────────────────────────────────────────────────────────────────────
# Core map builder — shared by all scan types
# ──────────────────────────────────────────────────────────────────────────────

def _build_and_save(
    line,
    tt_aper,
    qx: float,
    qy: float,
    xi_x: float,
    xi_y: float,
    case_name: str,
    scan_type: str,
    use_errors: bool,
    simplified: bool,
    xi_x_label: float | None = None,   # if not None, included in filename
    xi_y_label: float | None = None,
) -> TuneMap:
    """Match, scan, build and save one TuneMap. Returns the TuneMap."""
    if use_errors:
        install_errors(line, "all")
    optimise_tune_chroma(line, xi_x, xi_y, qx, qy)

    delta_pos = SweepTrajectory.find_delta_limit(
        line, +1, tt_aper,
        max_delta_scan=DELTA_SCAN_MAX, n_scan_points=DELTA_SCAN_NPTS,
    )
    delta_neg = SweepTrajectory.find_delta_limit(
        line, -1, tt_aper,
        max_delta_scan=DELTA_SCAN_MAX, n_scan_points=DELTA_SCAN_NPTS,
    )
    print(f"  δ range: [{delta_neg:.4g}, {delta_pos:.4g}]")

    if simplified:
        tw0 = line.twiss4d()
        chroma_x, chroma_y = tw0.dqx, tw0.dqy
        print(f"  Q'x={chroma_x:.3f}  Q'y={chroma_y:.3f}")
        n_pts = max(10, int(round((delta_pos - delta_neg) / TRAJ_STEP)) + 1)
        sweep = SweepTrajectory.from_chroma(
            qx, qy, chroma_x, chroma_y,
            delta_range=(delta_neg, delta_pos),
            n_points=n_pts,
        )
    else:
        sweep = SweepTrajectory.from_twiss_scan(
            line,
            delta_range=(delta_neg, delta_pos),
            step=TRAJ_STEP,
            verbose=False,
        )

    tm    = sweep.build_map()
    fname = _tune_map_filename(qx, qy, xi_x=xi_x_label, xi_y=xi_y_label)
    out   = os.path.join(_case_dir(scan_type, case_name), fname)
    tm.save(out)
    print(f"  Saved → {out}")
    return tm


# ──────────────────────────────────────────────────────────────────────────────
# Tune diagram plotter — generic, works for all scan types
# ──────────────────────────────────────────────────────────────────────────────

def plot_tune_diagram(
    maps: dict,           # key → TuneMap  (key can be float or tuple)
    scan_cfg: dict,
    case_name: str,
    n_sample: int = 300,
) -> plt.Figure:
    """
    One tune diagram for a single (scan, case) combination.

    The viridis colormap cycles over the scanned parameter (Qx, Qy, or xi).
    The coolwarm colormap shows δ along each trajectory.
    """
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
    cax_d  = ax.inset_axes([1.04, 0.0, 0.03, 1.0])   # δ colorbar
    cax_wp = ax.inset_axes([1.20, 0.0, 0.03, 1.0])   # scanned param colorbar

    td.plot(ax=ax, show_working_point=False)
    ax.set_aspect("equal")

    wp_cmap = cm.viridis
    wp_norm = Normalize(vmin=scan_vals[0], vmax=scan_vals[-1])
    d_cmap  = "coolwarm"
    d_min   = min(tm.delta_min for tm in maps.values())
    d_max   = max(tm.delta_max for tm in maps.values())
    d_norm  = Normalize(vmin=d_min, vmax=d_max)

    last_lc = None
    for key, tm in maps.items():
        color = wp_cmap(wp_norm(scan_val_of(key)))
        d_arr, qx_arr, qy_arr = tm.sample(n_sample)

        points   = np.array([qx_arr, qy_arr]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        d_mid    = 0.5 * (d_arr[:-1] + d_arr[1:])

        lc = LineCollection(segments, cmap=d_cmap, norm=d_norm,
                            lw=2.0, zorder=3, alpha=0.85)
        lc.set_array(d_mid)
        ax.add_collection(lc)
        last_lc = lc

        qx_wp, qy_wp = tm(0.0)
        ax.scatter(qx_wp, qy_wp, color=color, zorder=5,
                   s=45, edgecolors="k", linewidths=0.5)

    # δ colorbar
    cbar_d = fig.colorbar(last_lc, cax=cax_d)
    cbar_d.set_label(r"$\delta\ [10^{-3}]$", fontsize=11)
    d_ticks = np.linspace(d_min, d_max, 9)
    cbar_d.set_ticks(d_ticks)
    cbar_d.set_ticklabels([f"{v * 1e3:.1f}" for v in d_ticks])

    # Scanned parameter colorbar
    sm = cm.ScalarMappable(cmap=wp_cmap, norm=wp_norm)
    sm.set_array([])
    cbar_wp = fig.colorbar(sm, cax=cax_wp)
    cbar_wp.set_label(cbar_label, fontsize=11)
    wp_ticks = np.linspace(scan_vals[0], scan_vals[-1], 7)
    cbar_wp.set_ticks(wp_ticks)
    cbar_wp.set_ticklabels([f"{v:.3f}" for v in wp_ticks])

    td.finalize(ax, extra_handles=[], xlabel=r"$Q_x$", ylabel=r"$Q_y$",
                legend_loc="upper left")
    ax.set_title(f"Sweep trajectories — {scan_type} — {case_name}", fontsize=13)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Scan runners
# ──────────────────────────────────────────────────────────────────────────────

def run_tune_scan(scan_cfg: dict, tt_aper) -> None:
    """Run a QxScan or QyScan."""
    scan_type = scan_cfg["type"]
    _setup_dirs(scan_type)

    is_qx = scan_type == "QxScan"
    tunes = scan_cfg["tunes"]
    xi_x  = scan_cfg["xi_x"]
    xi_y  = scan_cfg["xi_y"]

    for case_name in MAP_CASES:
        print(f"\n{'═' * 60}")
        print(f"{scan_type} — {case_name}")
        print("═" * 60)

        use_errors = case_name == "WithErrors"
        simplified = case_name == "Simplified"
        line = xt.load(LINE_PATH)
        _setup_cavities(line)

        maps = {}
        for tune in tunes:
            qx = tune if is_qx else scan_cfg["fixed_qx"]
            qy = scan_cfg["fixed_qy"] if is_qx else tune
            key = tune
            if qx == qy:
                print(f"\n  Qx={qx:.3f}  Qy={qy:.3f} — skipped (Qx = Qy, degenerate working point)")
                continue
            print(f"\n  Qx={qx:.3f}  Qy={qy:.3f}")
            maps[key] = _build_and_save(
                line, tt_aper, qx, qy, xi_x, xi_y,
                case_name, scan_type,
                use_errors, simplified,
            )  # no xi labels for tune scans

        # Tune diagram
        fig      = plot_tune_diagram(maps, scan_cfg, case_name)
        fig_path = os.path.join(_case_dir(scan_type, case_name),
                                f"tune_diagram_{case_name}.pdf")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved diagram → {fig_path}")


def run_chroma_scan(scan_cfg: dict, tt_aper) -> None:
    """Run a ChromaScan: fixed Qx, Qy; scan xi."""
    scan_type = scan_cfg["type"]
    _setup_dirs(scan_type)

    qx        = scan_cfg["fixed_qx"]
    qy        = scan_cfg["fixed_qy"]
    xi_pairs  = scan_cfg["xi_pairs"]

    for case_name in MAP_CASES:
        print(f"\n{'═' * 60}")
        print(f"{scan_type} — {case_name}")
        print("═" * 60)

        use_errors = case_name == "WithErrors"
        simplified = case_name == "Simplified"
        line = xt.load(LINE_PATH)
        _setup_cavities(line)

        maps = {}
        for xi_x, xi_y in xi_pairs:
            print(f"\n  Qx={qx:.3f}  Qy={qy:.3f}  xi_x={xi_x:.3f}  xi_y={xi_y:.3f}")
            maps[(xi_x, xi_y)] = _build_and_save(
                line, tt_aper, qx, qy, xi_x, xi_y,
                case_name, scan_type,
                use_errors, simplified,
                xi_x_label=xi_x, xi_y_label=xi_y,
            )

        # Tune diagram
        fig      = plot_tune_diagram(maps, scan_cfg, case_name)
        fig_path = os.path.join(_case_dir(scan_type, case_name),
                                f"tune_diagram_{case_name}.pdf")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved diagram → {fig_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print(f"Loading aperture table from {LINE_PATH} ...")
    line_base = xt.load(LINE_PATH)
    _setup_cavities(line_base)
    tt_aper = line_base.get_aperture_table()

    for scan_cfg in ACTIVE_SCANS:
        scan_type = scan_cfg["type"]
        print(f"\n{'═' * 60}")
        print(f"Starting scan: {scan_type}")
        print("═" * 60)

        if scan_type in ("QxScan", "QyScan"):
            run_tune_scan(scan_cfg, tt_aper)
        elif scan_type in ("ChromaScanY", "ChromaScanX"):
            run_chroma_scan(scan_cfg, tt_aper)
        else:
            raise ValueError(f"Unknown scan type: {scan_type}")

    print("\nAll scans done.")


if __name__ == "__main__":
    main()
