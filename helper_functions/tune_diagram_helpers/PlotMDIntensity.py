"""
plot_md_intensity.py
====================
Loads MD intensity data and TuneMaps to produce four plots per optics case
(WithErrors, WithoutErrors, Simplified) — one figure each, all working points
overlaid:

    Plot 1 — Tune diagram: mean scatter points on each trajectory, sized by
              |dI/dδ| (percentile-normalised), coloured by δ (coolwarm).
              δ colorbar added. Reps removed for clarity.

    Plot 2 — Tune diagram: mean scatter points on each trajectory, coloured
              by normalised intensity (RdYlGn), opaque. Intensity thresholds
              at 75%, 50%, 25% marked on each trajectory.

    Plot 3 — |dI/dδ| vs δ: all working points, viridis by scanned tune.
              Mean only (no reps). Log y-scale. Y-limit set at 95th
              percentile of mean values across all working points; working
              points whose peak exceeds this are warned.

    Plot 4 — Qx(δ) and Qy(δ) vs δ: from TuneMap. Peak-loss δ (= δ where
              |dI/dδ| is max) marked as vertical dotted lines.

Data folder naming convention (decimal places vary, auto-detected):
    results/TUNE_20.085_20.18_NEG/  id1.json  id2.json  id3.json
    results/TUNE_20.085_20.18_POS/  id1.json  id2.json  id3.json

Output:
    studies/plots_md_intensity/
        QxScan/
            WithErrors/   plot1...pdf  plot2...pdf  plot3...pdf  plot4...pdf
            WithoutErrors/ ...
            Simplified/    ...

Map lookup:
    - tune scans:          SweepTrajectoryMaps/QxScan/... and QyScan/...
    - chromaticity maps:   sps-chromaticity-maps/{with_errors,without_errors,simplified}/

Usage
-----
    python plot_md_intensity.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter

# ── path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "helper_functions"))
sys.path.insert(0, str(REPO_ROOT / "helper_functions" / "intensity_helpers"))
sys.path.insert(0, str(THIS_DIR))
from load_paths import get_path

HELPER_DIR = get_path(
    "frederik_offmom_scans_root",
    default="/Users/lisepauwels/phd/code/sps-offmom-scans",
)
TUNE_DIR = REPO_ROOT / "helper_functions" / "intensity_helpers"
sys.path.insert(0, str(HELPER_DIR))
sys.path.insert(0, str(TUNE_DIR))

from bct import get_data_from_file
from plot import dr_to_delta
from tune_diagram import TuneDiagram, TuneMap
from workflow_common import (
    CHROMA_SCAN_X_MACHINE,
    CHROMA_SCAN_Y_MACHINE,
    MAP_CASES,
    MAX_ORDER,
    QX_SCAN,
    QX_SCAN_MACHINE,
    QY_SCAN,
    QY_SCAN_MACHINE,
    finalize_colorbar_heights,
    is_tune_scan,
    make_tune_diagram_figure,
    map_case_root_for_scan_type,
    scan_key_label,
    scan_key_to_chroma,
    scan_key_to_filename_chroma,
    scan_key_to_map_chroma,
    scan_key_to_measurement_chroma,
    scan_plot_suffix,
    scan_key_to_working_point,
    scan_keys_and_labels,
    scan_param_value,
    tune_diagram_spec,
    tune_map_filename,
)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DATA_ROOT = get_path(
    "offmom_scans_results_root",
    default=str(HELPER_DIR / "results"),
)

# Active scans
SCANS = [
    QX_SCAN,
    QY_SCAN,
    QX_SCAN_MACHINE,
    QY_SCAN_MACHINE,
    CHROMA_SCAN_Y_MACHINE,
    CHROMA_SCAN_X_MACHINE,
]


# ── Hard exclusion list ──────────────────────────────────────────────────────
# Tune scans:   (qx, qy, side, rep_id)
# Chroma scans: (xi_x, xi_y, side, rep_id)
EXCLUDED_FILES = {
    # Tune scan exclusions
    (20.175, 20.18, 'POS', 1),
    (20.155, 20.18, 'POS', 1),
    (20.140, 20.18, 'NEG', 1),
    # Chroma scan exclusions — add here as needed
    # e.g. (0.5, 0.20, 'NEG', 2),
}

REP_IDS    = [1, 2, 3]
N_SAMPLE   = 200    # points for TuneMap sampling on tune diagrams

# Savitzky-Golay smoothing for dI/dδ
SAVGOL_WINDOW    = 21
SAVGOL_POLYORDER = 3

# Percentile used for dot size normalisation in plot 1
# (clips outliers so one bad trajectory doesn't blow up the scale)
SIZE_PERCENTILE = 95

# Intensity thresholds to mark on plot 2 (fraction of full beam)
INTENSITY_THRESHOLDS = [0.75, 0.50, 0.25]

# Percentile used to cap y-axis in plot 3 (mean |dI/dδ| values)
YLIM_PERCENTILE = 95

# Minimum outlier ratio: if a working point's peak loss exceeds
# OUTLIER_FACTOR × the 95th-percentile y-limit, warn about it
OUTLIER_FACTOR = 3.0

# Maximum dot size for plot 1 (pt²). Minimum is always 4 pt².
# Increase if dots are too small, decrease if too crowded.
DOT_SCALE = 80

# Minimum data points per rep
MIN_POINTS = 10

OUTPUT_ROOT = REPO_ROOT / "studies" / "plots_md_intensity"


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def _map_filename(qx: float, qy: float,
                  xi_x: float | None = None,
                  xi_y: float | None = None) -> str:
    return tune_map_filename(qx, qy, xi_x=xi_x, xi_y=xi_y)


def _legacy_map_filename(qx: float, qy: float) -> str:
    return tune_map_filename(qx, qy, xi_x=None, xi_y=None)


def _folder_name_tune(qx: float, qy: float, side: str) -> Path | None:
    """Locate TUNE_* folder, trying common decimal formats then glob."""
    for qx_fmt in [f"{qx:.3f}", f"{qx:.2f}", f"{qx:.1f}"]:
        for qy_fmt in [f"{qy:.2f}", f"{qy:.3f}", f"{qy:.1f}"]:
            c = DATA_ROOT / f"TUNE_{qx_fmt}_{qy_fmt}_{side}"
            if c.exists():
                return c
    qx_str = f"{qx:.3f}".rstrip("0").rstrip(".")
    matches = list(DATA_ROOT.glob(f"TUNE_{qx_str}*_{side}"))
    return matches[0] if matches else None


def _folder_name_chroma(xi_x: float, xi_y: float, side: str) -> Path | None:
    """Locate CHROM_* folder, trying common decimal formats then glob."""
    for xx_fmt in [f"{xi_x:.3f}", f"{xi_x:.2f}", f"{xi_x:.1f}"]:
        for xy_fmt in [f"{xi_y:.3f}", f"{xi_y:.2f}", f"{xi_y:.1f}"]:
            c = DATA_ROOT / f"CHROM_{xx_fmt}_{xy_fmt}_{side}"
            if c.exists():
                return c
    xx_str = f"{xi_x:.3f}".rstrip("0").rstrip(".")
    matches = list(DATA_ROOT.glob(f"CHROM_{xx_str}*_{side}"))
    return matches[0] if matches else None


def load_tune_map(scan_cfg: dict, key, map_case: str) -> TuneMap | None:
    """Load a TuneMap for any scan type."""
    scan_label = scan_cfg["label"]
    scan_type = scan_cfg["type"]
    qx, qy = scan_key_to_working_point(scan_cfg, key)
    xi_x, xi_y = scan_key_to_filename_chroma(scan_cfg, key)
    fname = _map_filename(qx, qy, xi_x=xi_x, xi_y=xi_y)
    path = map_case_root_for_scan_type(scan_type, map_case) / fname
    if not path.exists() and is_tune_scan(scan_cfg) and xi_x is not None:
        legacy_path = map_case_root_for_scan_type(scan_type, map_case) / _legacy_map_filename(qx, qy)
        if legacy_path.exists():
            path = legacy_path
    if not path.exists():
        print(f"  [WARNING] TuneMap not found: {path}")
        return None
    return TuneMap.load(str(path))


def _smooth(y: np.ndarray) -> np.ndarray:
    """Savitzky-Golay smooth, falls back gracefully if array is too short."""
    win = min(SAVGOL_WINDOW, len(y) if len(y) % 2 == 1 else len(y) - 1)
    win = max(win, SAVGOL_POLYORDER + 2)
    if win % 2 == 0:
        win -= 1
    if len(y) < win:
        return y
    return savgol_filter(y, win, SAVGOL_POLYORDER)


def _is_valid_rep(intensity: np.ndarray) -> bool:
    if len(intensity) < MIN_POINTS:
        return False
    if intensity.max() < 0.5:
        return False
    if intensity.max() - intensity.min() < 0.1:
        return False
    return True


def load_rep_data(scan_cfg: dict, key) -> list[dict] | None:
    """
    Load 3 NEG + 3 POS repetitions for one scan point.

    key is the scan parameter value:
      - tune scan:   key = qx or qy (float)
      - chroma scan: key = (xi_x, xi_y) tuple

    Returns list of valid dicts with keys:
        delta     : δ in 10⁻³, sorted ascending
        intensity : normalised intensity clipped to [0, 1]
        di_ddelta : Savitzky-Golay smoothed dI/dδ
    Returns None if no valid reps found.
    """
    scan_type = scan_cfg["type"]

    if is_tune_scan(scan_cfg):
        qx, qy = scan_key_to_working_point(scan_cfg, key)
        folder_neg = _folder_name_tune(qx, qy, "NEG")
        folder_pos = _folder_name_tune(qx, qy, "POS")
        excl_key_neg = lambda r: (round(qx, 3), round(qy, 2), "NEG", r)
        excl_key_pos = lambda r: (round(qx, 3), round(qy, 2), "POS", r)
    elif scan_type in ("ChromaScanX", "ChromaScanY"):
        xi_x, xi_y = scan_key_to_measurement_chroma(scan_cfg, key)
        folder_neg = _folder_name_chroma(xi_x, xi_y, "NEG")
        folder_pos = _folder_name_chroma(xi_x, xi_y, "POS")
        excl_key_neg = lambda r: (round(xi_x, 3), round(xi_y, 2), "NEG", r)
        excl_key_pos = lambda r: (round(xi_x, 3), round(xi_y, 2), "POS", r)
    else:
        raise ValueError(f"Unsupported scan type: {scan_type}")

    label = scan_key_label(scan_cfg, key)

    if folder_neg is None or folder_pos is None:
        print(f"  [WARNING] Folder not found for {label}")
        return None

    reps = []
    for rep_id in REP_IDS:
        try:
            skip_neg = excl_key_neg(rep_id) in EXCLUDED_FILES
            skip_pos = excl_key_pos(rep_id) in EXCLUDED_FILES

            if skip_neg:
                print(f"  [EXCLUDED] {label} NEG id{rep_id}")
            if skip_pos:
                print(f"  [EXCLUDED] {label} POS id{rep_id}")

            # Skip rep entirely if both sides excluded
            if skip_neg and skip_pos:
                continue

            # Load whichever sides are needed
            if not skip_neg:
                _, _, R_neg, I_neg = get_data_from_file(folder_neg / f"id{rep_id}.json")
                delta_neg = 1000 * dr_to_delta(R_neg)

            if not skip_pos:
                _, _, R_pos, I_pos = get_data_from_file(folder_pos / f"id{rep_id}.json")
                delta_pos = 1000 * dr_to_delta(R_pos)

            # Combine available sides
            if skip_neg:
                delta         = delta_pos
                intensity_arr = I_pos
            elif skip_pos:
                delta         = delta_neg[::-1]
                intensity_arr = I_neg[::-1]
            else:
                delta         = np.concatenate([delta_neg[::-1], delta_pos])
                intensity_arr = np.concatenate([I_neg[::-1], I_pos])

        except FileNotFoundError as e:
            print(f"  [WARNING] {e}")
            continue

        order     = np.argsort(delta)
        delta     = delta[order]
        intensity = np.clip(intensity_arr[order], 0.0, 1.0)

        # Remove duplicate δ values (cause divide-by-zero in np.gradient)
        _, unique_idx = np.unique(delta, return_index=True)
        delta     = delta[unique_idx]
        intensity = intensity[unique_idx]

        if not _is_valid_rep(intensity):
            print(f"  [WARNING] Rep {rep_id} {label} failed validity check — skipping.")
            continue

        di_ddelta = _smooth(np.gradient(intensity, delta))

        reps.append({"delta": delta, "intensity": intensity, "di_ddelta": di_ddelta})

    return reps if reps else None


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────

def _common_delta(reps: list[dict], n: int = N_SAMPLE,
                  tm: TuneMap | None = None) -> np.ndarray | None:
    """
    If tm is given, use the full TuneMap δ range for display — this avoids
    the range shrinking when some reps are excluded or only cover one side.
    If tm is None, use the overlap of all reps (for loss-rate plots where
    we need actual data coverage).
    """
    if tm is not None:
        return np.linspace(tm.delta_min * 1e3, tm.delta_max * 1e3, n)
    d_min = max(r["delta"].min() for r in reps)
    d_max = min(r["delta"].max() for r in reps)
    if d_min >= d_max:
        return None
    return np.linspace(d_min, d_max, n)


def _interp_stack(reps, delta_c, key, transform=None):
    stack = np.array([
        np.interp(delta_c, r["delta"],
                  transform(r[key]) if transform else r[key])
        for r in reps
    ])
    return stack, stack.mean(axis=0), stack.std(axis=0)


def _td_and_fig(scan, scan_vals, n_cbars=1):
    """Create TuneDiagram + figure with n_cbars colorbar axes via GridSpec."""
    spec = tune_diagram_spec(scan)
    td = TuneDiagram(qx0=spec["qx0"], qy0=spec["qy0"],
                     half_range=spec["half_range"],
                     max_order=MAX_ORDER, skew=True)
    fig, ax, caxes = make_tune_diagram_figure(n_cbars=n_cbars)
    td.plot(ax=ax, show_working_point=False)
    ax.set_aspect("equal")
    return fig, ax, td, caxes


def _threshold_delta(delta, intensity, threshold):
    """Return δ where intensity crosses threshold (interpolated), or None."""
    # Find sign changes of (intensity - threshold)
    residual = intensity - threshold
    crossings = []
    for i in range(len(residual) - 1):
        if residual[i] * residual[i + 1] < 0:
            # Linear interpolation
            d = delta[i] + (delta[i + 1] - delta[i]) * (-residual[i]) / (residual[i + 1] - residual[i])
            crossings.append(d)
    return crossings


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1 — Tune diagram: mean only, sized by |dI/dδ|, coloured by δ
# ──────────────────────────────────────────────────────────────────────────────

def plot1_sized_by_loss(all_data, scan_vals, map_case, cbar_label, scan):
    from matplotlib.collections import LineCollection

    # Two colorbars: δ (coolwarm) and Qx (viridis)
    fig, ax, td, (cax_d, cax_wp) = _td_and_fig(scan, scan_vals, n_cbars=2)

    wp_cmap = cm.viridis
    wp_norm = Normalize(vmin=scan_vals[0], vmax=scan_vals[-1])
    d_min   = min(d["tm"].delta_min * 1e3 for d in all_data.values() if d)
    d_max   = max(d["tm"].delta_max * 1e3 for d in all_data.values() if d)
    d_norm  = Normalize(vmin=d_min, vmax=d_max)

    # Size normalisation: use percentile across all working points so one
    # outlier doesn't dominate. Sizes are mapped to [MIN_SIZE, MAX_SIZE] pt².
    all_loss_vals = np.concatenate([
        np.abs(r["di_ddelta"])
        for d in all_data.values() if d
        for r in d["reps"]
    ])
    size_ref = np.percentile(all_loss_vals, SIZE_PERCENTILE)
    if size_ref == 0:
        size_ref = 1.0
    MIN_SIZE, MAX_SIZE = 4, DOT_SCALE

    def _size(loss_arr):
        """Map loss rate to marker area in [MIN_SIZE, MAX_SIZE] pt²."""
        norm = np.clip(np.abs(loss_arr) / size_ref, 0.0, 1.0)
        return MIN_SIZE + (MAX_SIZE - MIN_SIZE) * norm

    last_lc = None
    for key, data in all_data.items():
        if data is None:
            continue
        tm, reps = data["tm"], data["reps"]
        delta_c  = _common_delta(reps, tm=tm)
        if delta_c is None:
            continue
        _, loss_mean, _ = _interp_stack(reps, delta_c, "di_ddelta", transform=np.abs)
        qx_c, qy_c = tm(np.clip(delta_c * 1e-3, tm.delta_min, tm.delta_max))

        # Faint trajectory line (coolwarm for δ)
        points   = np.array([qx_c, qy_c]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap="coolwarm", norm=d_norm,
                            lw=1.0, zorder=2, alpha=0.35)
        lc.set_array(0.5 * (delta_c[:-1] + delta_c[1:]))
        ax.add_collection(lc)
        last_lc = lc

        # Mean — size AND alpha both scale with loss rate
        # Per-point alpha requires binning (matplotlib limitation)
        sizes  = _size(loss_mean)
        alphas = np.clip(np.abs(loss_mean) / size_ref, 0.05, 0.9)
        n_bins = 15
        bins   = np.linspace(0.0, 1.0, n_bins + 1)
        for i in range(n_bins):
            norm_loss = np.abs(loss_mean) / size_ref
            mask = (norm_loss >= bins[i]) & (norm_loss < bins[i + 1])
            if mask.sum() == 0:
                continue
            a = float(np.clip(0.5 * (bins[i] + bins[i + 1]), 0.05, 0.9))
            ax.scatter(qx_c[mask], qy_c[mask],
                       c=delta_c[mask], cmap="coolwarm", norm=d_norm,
                       s=sizes[mask], zorder=4,
                       alpha=a, edgecolors="none", linewidths=0)

        # Working point marker at δ=0
        qx_wp, qy_wp = tm(0.0)
        ax.scatter(qx_wp, qy_wp, color=wp_cmap(wp_norm(data["scan_val"])),
                   zorder=5, s=50, edgecolors="k", linewidths=0.5)

    # δ colorbar (inner)
    if last_lc is not None:
        cbar_d = fig.colorbar(last_lc, cax=cax_d)
        cbar_d.set_label(r"$\delta\ [10^{-3}]$", fontsize=16)
        cbar_d.ax.tick_params(labelsize=14)
        d_ticks = np.linspace(d_min, d_max, 9)
        cbar_d.set_ticks(d_ticks)
        cbar_d.set_ticklabels([f"{v:.1f}" for v in d_ticks])

    # working-point colorbar (outer)
    sm = cm.ScalarMappable(cmap=wp_cmap, norm=wp_norm)
    sm.set_array([])
    cbar_wp = fig.colorbar(sm, cax=cax_wp)
    cbar_wp.set_label(cbar_label, fontsize=16)
    cbar_wp.ax.tick_params(labelsize=14)
    ticks = np.linspace(scan_vals[0], scan_vals[-1], 7)
    cbar_wp.set_ticks(ticks)
    cbar_wp.set_ticklabels([f"{v:.3f}" for v in ticks])

    # Legend: show size + alpha together for three loss levels
    for frac, label in [(0.25, "25%"), (0.5, "50%"), (1.0, f"p{SIZE_PERCENTILE}")]:
        val = frac * size_ref
        a   = float(np.clip(frac, 0.05, 0.9))
        ax.scatter([], [], s=_size(np.array([val]))[0], color="gray",
                   label=rf"$|dI/d\delta|$ = {label}",
                   alpha=a, edgecolors="none")

    td.finalize(ax, extra_handles=[
        mlines.Line2D([], [], color="gray", marker="o", ls="None",
                      markersize=5, label="Mean loss rate"),
    ], xlabel=r"$Q_x$", ylabel=r"$Q_y$")
    ax.set_title(f"Loss rate on trajectory — {map_case}", fontsize=12)
    finalize_colorbar_heights(fig, ax, [cax_d, cax_wp])
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Plot 2 — Tune diagram: mean only, coloured by intensity, + thresholds
# ──────────────────────────────────────────────────────────────────────────────

def plot2_coloured_by_intensity(all_data, scan_vals, map_case, cbar_label, scan):
    fig, ax, td, (cax_wp, cax_i) = _td_and_fig(scan, scan_vals, n_cbars=2)

    wp_cmap = cm.viridis
    wp_norm = Normalize(vmin=scan_vals[0], vmax=scan_vals[-1])
    i_norm  = Normalize(vmin=0, vmax=1)
    i_cmap  = cm.RdYlGn

    # Threshold marker styles: 75%, 50%, 25%
    thresh_styles = {
        0.75: ("^", 12, "75%"),
        0.50: ("D", 12, "50%"),
        0.25: ("v", 12, "25%"),
    }

    last_sc = None
    for key, data in all_data.items():
        if data is None:
            continue
        tm, reps = data["tm"], data["reps"]
        delta_c  = _common_delta(reps, tm=tm)
        if delta_c is None:
            continue
        _, int_mean, _ = _interp_stack(reps, delta_c, "intensity")
        qx_c, qy_c = tm(np.clip(delta_c * 1e-3, tm.delta_min, tm.delta_max))
        color = wp_cmap(wp_norm(data["scan_val"]))

        # Thin trajectory line in working-point colour
        ax.plot(qx_c, qy_c, color=color, lw=0.8, zorder=2, alpha=0.35)

        # Mean scatter — fully opaque, no alpha needed
        sc = ax.scatter(qx_c, qy_c, c=int_mean, cmap=i_cmap, norm=i_norm,
                        s=18, zorder=4, linewidths=0)
        last_sc = sc

        # Intensity threshold markers (75%, 50%, 25%)
        for thresh, (marker, ms, _) in thresh_styles.items():
            crossings = _threshold_delta(delta_c, int_mean, thresh)
            for d_cross in crossings:
                qx_cross, qy_cross = tm(np.clip(d_cross * 1e-3, tm.delta_min, tm.delta_max))
                ax.scatter(qx_cross, qy_cross, marker=marker, s=ms,
                           color=color, zorder=6, linewidths=0.5,
                           edgecolors="k")

        # Working point at δ=0
        qx_wp, qy_wp = tm(0.0)
        ax.scatter(qx_wp, qy_wp, color=color, zorder=5,
                   s=40, edgecolors="k", linewidths=0.5)

    # Intensity colorbar (inner)
    if last_sc is not None:
        cbar_i = fig.colorbar(last_sc, cax=cax_i)
        cbar_i.set_label("Normalised Intensity [-]", fontsize=16)
        cbar_i.ax.tick_params(labelsize=14)

    # working-point colorbar (outer)
    sm = cm.ScalarMappable(cmap=wp_cmap, norm=wp_norm)
    sm.set_array([])
    cbar_wp = fig.colorbar(sm, cax=cax_wp)
    cbar_wp.set_label(cbar_label, fontsize=16)
    cbar_wp.ax.tick_params(labelsize=14)
    ticks = np.linspace(scan_vals[0], scan_vals[-1], 7)
    cbar_wp.set_ticks(ticks)
    cbar_wp.set_ticklabels([f"{v:.3f}" for v in ticks])

    # Threshold legend (one entry per threshold, colour-neutral)
    thresh_handles = [
        mlines.Line2D([], [], marker=m, color="gray", ls="None",
                      markersize=4, markeredgecolor="k",
                      markeredgewidth=0.5, label=f"I = {lbl}")
        for _, (m, _, lbl) in thresh_styles.items()
    ]
    td.finalize(ax, extra_handles=thresh_handles,
                xlabel=r"$Q_x$", ylabel=r"$Q_y$")
    ax.set_title(f"Intensity on trajectory — {map_case}", fontsize=12)
    finalize_colorbar_heights(fig, ax, [cax_wp, cax_i])
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Plot 3 — |dI/dδ| vs δ, mean only, log scale, auto y-limit with outlier warn
# ──────────────────────────────────────────────────────────────────────────────

def plot3_loss_vs_delta(all_data, scan_vals, map_case, cbar_label, scan):
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    wp_cmap = cm.viridis
    wp_norm = Normalize(vmin=scan_vals[0], vmax=scan_vals[-1])

    # Pre-compute all mean loss peaks to set a sensible y-limit
    peak_losses = {}
    for key, data in all_data.items():
        if data is None:
            continue
        reps    = data["reps"]
        delta_c = _common_delta(reps)
        if delta_c is None:
            continue
        _, loss_mean, _ = _interp_stack(reps, delta_c, "di_ddelta", transform=np.abs)
        peak_losses[key] = loss_mean.max()

    if not peak_losses:
        return fig

    all_peaks = np.array(list(peak_losses.values()))
    ylim_top  = np.percentile(all_peaks, YLIM_PERCENTILE)

    # Warn about outliers
    for k, peak in peak_losses.items():
        if peak > OUTLIER_FACTOR * ylim_top:
            warnings.warn(
                f"Key={k}: peak |dI/dδ| = {peak:.2f} is "
                f"{peak/ylim_top:.1f}× the {YLIM_PERCENTILE}th-percentile "
                f"y-limit ({ylim_top:.2f}). Clipped in plot 3.",
                stacklevel=2,
            )

    for key, data in all_data.items():
        if data is None:
            continue
        reps    = data["reps"]
        color   = wp_cmap(wp_norm(data["scan_val"]))
        delta_c = _common_delta(reps)
        if delta_c is None:
            continue
        _, loss_mean, loss_std = _interp_stack(reps, delta_c, "di_ddelta",
                                               transform=np.abs)

        # Mean only (no individual reps on this plot)
        ax.plot(delta_c, loss_mean, color=color, lw=1.8)
        ax.fill_between(delta_c,
                        np.maximum(loss_mean - loss_std, 1e-6),
                        loss_mean + loss_std,
                        color=color, alpha=0.15)

    sm = cm.ScalarMappable(cmap=wp_cmap, norm=wp_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    ticks = np.linspace(scan_vals[0], scan_vals[-1], 7)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{v:.3f}" for v in ticks])

    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-3, top=ylim_top * 1.2)
    ax.set_xlabel(r"$\delta\ [10^{-3}]$", fontsize=16)
    ax.set_ylabel(r"$|dI/d\delta|$", fontsize=16)
    ax.set_title(r"Loss rate vs $\delta$ — " + map_case, fontsize=14)
    ax.tick_params(labelsize=14)
    ax.grid(True, which="both", alpha=0.3)
    ax.grid(which="minor", alpha=0.15)

    # Note if any curves are clipped
    n_clipped = sum(1 for p in peak_losses.values() if p > OUTLIER_FACTOR * ylim_top)
    if n_clipped:
        ax.text(0.01, 0.97,
                f"{n_clipped} working point(s) clipped\n(peak > {OUTLIER_FACTOR:.0f}× y-limit)",
                transform=ax.transAxes, fontsize=8, va="top",
                color="tomato", style="italic")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Plot 4 — Qx(δ) and Qy(δ) from TuneMap, peak-loss δ as vertical lines
# ──────────────────────────────────────────────────────────────────────────────

def plot4_tunes_vs_delta(all_data, scan_vals, map_case, cbar_label, scan):
    fig, (ax_qx, ax_qy) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, constrained_layout=True
    )
    fig.suptitle(r"$Q_x(\delta)$, $Q_y(\delta)$ — " + map_case, fontsize=12)

    wp_cmap = cm.viridis
    wp_norm = Normalize(vmin=scan_vals[0], vmax=scan_vals[-1])

    for key, data in all_data.items():
        if data is None:
            continue
        tm, reps = data["tm"], data["reps"]
        color    = wp_cmap(wp_norm(data["scan_val"]))
        delta_c  = _common_delta(reps, tm=tm)
        if delta_c is None:
            continue
        delta_cl = np.clip(delta_c * 1e-3, tm.delta_min, tm.delta_max)
        qx_arr, qy_arr = tm(delta_cl)

        ax_qx.plot(delta_c, qx_arr, color=color, lw=1.5, alpha=0.8)
        ax_qy.plot(delta_c, qy_arr, color=color, lw=1.5, alpha=0.8)

        # Peak-loss δ: δ where mean |dI/dδ| is maximum
        _, loss_mean, _ = _interp_stack(reps, delta_c, "di_ddelta", transform=np.abs)
        if loss_mean.max() > 0:
            peak_delta = delta_c[np.argmax(loss_mean)]
            ax_qx.axvline(peak_delta, color=color, lw=0.8, ls=":", alpha=0.7)
            ax_qy.axvline(peak_delta, color=color, lw=0.8, ls=":", alpha=0.7)

    sm = cm.ScalarMappable(cmap=wp_cmap, norm=wp_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_qx, ax_qy], pad=0.02)
    cbar.set_label(cbar_label, fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    ticks = np.linspace(scan_vals[0], scan_vals[-1], 7)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{v:.3f}" for v in ticks])

    ax_qx.set_ylabel(r"$Q_x$", fontsize=16)
    ax_qx.tick_params(labelsize=14)
    ax_qx.grid(True, alpha=0.4)
    ax_qx.minorticks_on()
    ax_qx.grid(which="minor", alpha=0.2)
    ax_qx.plot([], [], color="gray", lw=0.8, ls=":",
               label=r"$\delta$ of peak $|dI/d\delta|$")
    ax_qx.legend(fontsize=9, loc="best")

    ax_qy.set_ylabel(r"$Q_y$", fontsize=16)
    ax_qy.set_xlabel(r"$\delta\ [10^{-3}]$", fontsize=16)
    ax_qy.tick_params(labelsize=14)
    ax_qy.grid(True, alpha=0.4)
    ax_qy.minorticks_on()
    ax_qy.grid(which="minor", alpha=0.2)

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def _scan_keys_and_labels(scan):
    return scan_keys_and_labels(scan)


def _plot_filename(scan_cfg: dict, stem: str) -> str:
    suffix = scan_plot_suffix(scan_cfg)
    if suffix:
        return f"{stem}_{suffix}.pdf"
    return f"{stem}.pdf"


def main():
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
                tm   = load_tune_map(scan, key, map_case)
                reps = load_rep_data(scan, key)
                if tm is None or reps is None:
                    all_data[key] = None
                    print("skipped.")
                else:
                    sv = scan_param_value(scan, key)
                    all_data[key] = {"tm": tm, "reps": reps, "scan_val": sv}
                    print("ok.")

            n_loaded = sum(1 for v in all_data.values() if v is not None)
            print(f"    -> {n_loaded}/{len(keys)} loaded.")
            if n_loaded == 0:
                print("    Nothing to plot -- skipping.")
                continue

            args = (all_data, scan_vals, map_case, cbar_label, scan)
            for fn, fname in [
                (plot1_sized_by_loss,         "plot1_tune_diagram_sized.pdf"),
                (plot2_coloured_by_intensity,  "plot2_tune_diagram_intensity.pdf"),
                (plot3_loss_vs_delta,          "plot3_loss_vs_delta.pdf"),
                (plot4_tunes_vs_delta,         "plot4_tunes_vs_delta.pdf"),
            ]:
                fig = fn(*args)
                stem = fname.removesuffix(".pdf")
                fig.savefig(out_dir / _plot_filename(scan, stem), dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"    Saved {_plot_filename(scan, stem)}")

    print(f"\nDone. Plots saved under '{OUTPUT_ROOT}/'")


if __name__ == "__main__":
    main()
