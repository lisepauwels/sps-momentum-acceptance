from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xcoll as xc
import xobjects as xo
import xpart as xp
import xtrack as xt
import sys


DEFAULT_LINE_PATH = (
    "/Users/lisepauwels/phd/code/sps-xsuite-model/"
    "sps_with_aperture_inj_q20_beam_sagitta4.json"
)

COORDINATES = ("x", "px", "y", "py", "zeta", "delta")
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "helper_functions"))
sys.path.insert(0, str(REPO_ROOT / "tune_scan_workflow"))

from tune_diagram import TuneDiagram, TuneMap
from workflow_common import tune_map_filename

error_variants = {
    'none': [0, 0, 0, 0, 0, 0],
    'dipole_b3' : [0, 0, 1, 0, 0, 0],
    'dipole_b5' : [0, 0, 0, 0, 1, 0],
    'dipole_b3b5': [0, 0, 1, 0, 1, 0],
    'quadrupole_b4': [0, 0, 0, 1, 0, 0],
    'quadrupole_b6': [0, 0, 0, 0, 0, 1],
    'quadrupole_b4b6': [0, 0, 0, 1, 0, 1],
    'dipole_b3_quadrupole_b4': [0, 0, 1, 1, 0, 0],
    'all': [0, 0, 1, 1, 1, 1]
}

def df_to_delta(df_hz: np.ndarray | float) -> np.ndarray | float:
    sps_gtr = 17.95
    sps_g0 = 27.643
    f0 = 200e6
    eta = 1 / sps_gtr**2 - 1 / sps_g0**2
    return -np.asarray(df_hz) / (f0 * eta)


def repo_convention_signed_sweep(total_sweep_hz: float, plane: str) -> float:
    if plane not in {"DPpos", "DPneg"}:
        raise ValueError("plane must be either 'DPpos' or 'DPneg'")
    return -abs(total_sweep_hz) if plane == "DPpos" else abs(total_sweep_hz)


def build_output_dir(base_dir: Path, case_name: str | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir.mkdir(parents=True, exist_ok=True)
    label = case_name or f"rf_sweep_{stamp}"
    outdir = base_dir / label
    if outdir.exists():
        outdir = base_dir / f"{label}_{stamp}"
    outdir.mkdir(parents=True, exist_ok=False)
    return outdir


def ensure_tidp(line: xt.Line, tidp_name: str = "tidp.11434") -> None:
    if tidp_name in line.element_names:
        return

    tidp_ap_tot = 147
    block_mvt = 29
    tidp = xc.EverestCollimator(
        length=4.3,
        material=xc.materials.Carbon,
        jaw_L=tidp_ap_tot / 2 + block_mvt,
        jaw_R=-tidp_ap_tot / 2 + block_mvt,
    )

    line.discard_tracker()
    line.collimators.install(names=[tidp_name], elements=[tidp])


def configure_line(
    line_path: str,
    qx: float,
    qy: float,
    xi_x: float,
    xi_y: float,
    omp_threads: str,
    point_monitor_element: str | None,
    error_variant_name: str,
) -> xt.Line:
    print(f"[rf_sweep_speed_scan] Loading lattice from {line_path}")
    line = xt.load(line_path)
    env = line.env

    cavity_elements, cavity_names = line.get_elements_of_type(xt.Cavity)
    for name in cavity_names:
        line[name].frequency = 200e6
        line[name].lag = 180
        line[name].voltage = 0
    # line['acl.31735'].voltage = 0 #setting 800 cav to 0V
    line['actcse.31632'].voltage = 3.0e6

    tw = line.twiss()
    # remove_offmom_bpms_apers(line, exn=3.5e-6, nrj=21, pmass=0.938, bucket_height=3e-3, n_buckets=2)

    # Installing errors
    b1, b2, b3, b4, b5, b6 = error_variants[error_variant_name]
    tte = env.elements.get_table()
    mask_rbends = tte.element_type == 'RBend'
    mask_quads = tte.element_type == 'Quadrupole'
    mask_sextupoles = tte.element_type == 'Sextupole'

    mba = tte.rows[mask_rbends].rows['mba.*'].name
    mbb = tte.rows[mask_rbends].rows['mbb.*'].name
    qf = tte.rows[mask_quads].rows['qf.*'].name
    qd = tte.rows[mask_quads].rows['qd.*'].name
    lsf = tte.rows[mask_sextupoles].rows['lsf.*'].name
    lsd = tte.rows[mask_sextupoles].rows['lsd.*'].name

    env.vars['qph_setvalue'] = 0.0
    env.vars['qpv_setvalue'] = 0.0

    # Set the strengths according to Hannes' measurements
    for nn in mba:
        env[nn].knl = np.array([b1*0., b2*0., b3*2.12e-3, b4*0., b5*-5.74, b6*0.])

    for nn in mbb:
        env[nn].knl = np.array([b1*0., b2*0., b3*-3.19e-3, b4*0., b5*-5.10, b6*0.])

    for nn in qf:
        env[nn].knl = np.array([b1*0., b2*0., b3*0., b4*0.75e-1, b5*0., b6*-0.87e3])

    for nn in qd:
        env[nn].knl = np.array([b1*0., b2*0., b3*0., b4*-2.03e-1, b5*0., b6*2.04e3])

    
    env.vars["qph_setvalue"] = xi_x
    env.vars["qpv_setvalue"] = xi_y

    print("[rf_sweep_speed_scan] Matching tune and chromaticity")
    line.match(
        method="6d",
        vary=[
            xt.VaryList(["kqf0", "kqd0"], step=1e-8, tag="quad"),
            xt.VaryList(["qph_setvalue", "qpv_setvalue"], step=1e-4, tag="sext"),
        ],
        targets=[
            xt.TargetSet(qx=qx, qy=qy, tol=1e-6, tag="tune"),
            xt.TargetSet(dqx=xi_x * qx, dqy=xi_y * qy, tol=1e-2, tag="chrom"),
        ],
    )

    print("[rf_sweep_speed_scan] Installing TIDP and building tracker")
    ensure_tidp(line)
    if point_monitor_element is not None:
        print(f"[rf_sweep_speed_scan] Cycling line to observation point {point_monitor_element}")
        line.cycle(name_first_element=point_monitor_element, inplace=True)
    line.discard_tracker()
    if omp_threads in {"auto", "openmp"}:
        context = xo.ContextCpu(omp_num_threads=0)
    else:
        omp_threads_value = int(omp_threads)
        context = xo.ContextCpu(omp_num_threads=omp_threads_value)
    line.build_tracker(_context=context)
    return line


def generate_particles(
    line: xt.Line,
    num_particles: int,
    nemitt_x: float,
    nemitt_y: float,
    sigma_z: float,
):
    return xp.generate_matched_gaussian_bunch(
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=sigma_z,
        num_particles=num_particles,
        line=line,
    )


def get_observation_dispersion(line: xt.Line) -> dict[str, float]:
    tw = line.twiss()
    return {
        "dx": float(np.asarray(tw.dx)[0]),
        "dpx": float(np.asarray(tw.dpx)[0]),
        "dy": float(np.asarray(tw.dy)[0]),
        "dpy": float(np.asarray(tw.dpy)[0]),
    }


def alive_particle_arrays(particles) -> dict[str, np.ndarray]:
    state = np.asarray(particles.state)
    alive_mask = state > 0
    arrays = {
        "particle_id": np.flatnonzero(alive_mask),
        "alive_count": np.array(int(alive_mask.sum())),
    }
    for coord in COORDINATES:
        arrays[coord] = np.asarray(getattr(particles, coord))[alive_mask].astype(float, copy=False)
    return arrays


def particle_stats(values: np.ndarray) -> tuple[float, float, float, float]:
    if values.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    std = float(np.std(values))
    centered = values - mean
    moment3 = float(np.mean(centered**3))
    skewness = float(moment3 / std**3) if std > 0 else np.nan
    return mean, std, moment3, skewness


def collect_turn_row(particles, turn: int, sweep_per_turn_hz: float) -> dict[str, float | int]:
    arrays = alive_particle_arrays(particles)
    row: dict[str, float | int] = {
        "turn": int(turn),
        "alive_count": int(arrays["alive_count"]),
        "delta_from_sweep": float(df_to_delta(sweep_per_turn_hz * turn)),
    }
    for coord in COORDINATES:
        mean, std, moment3, skewness = particle_stats(arrays[coord])
        row[f"{coord}_mean"] = mean
        row[f"{coord}_std"] = std
        row[f"{coord}_moment3"] = moment3
        row[f"{coord}_skewness"] = skewness
        if coord == "delta":
            row["delta_min"] = float(np.min(arrays[coord])) if arrays[coord].size else np.nan
            row["delta_max"] = float(np.max(arrays[coord])) if arrays[coord].size else np.nan
    return row


def write_frame_outputs(base_path: Path, frame: pd.DataFrame) -> None:
    parquet_path = base_path.with_suffix(".parquet")
    frame.to_parquet(parquet_path, index=False)


def add_dispersion_subtracted_columns(
    summary_frame: pd.DataFrame,
    dispersion: dict[str, float],
) -> pd.DataFrame:
    frame = summary_frame.copy()
    delta_ref = frame["delta_mean"].to_numpy(dtype=float)
    frame["x_beta_mean"] = frame["x_mean"] - dispersion["dx"] * delta_ref
    frame["px_beta_mean"] = frame["px_mean"] - dispersion["dpx"] * delta_ref
    frame["y_beta_mean"] = frame["y_mean"] - dispersion["dy"] * delta_ref
    frame["py_beta_mean"] = frame["py_mean"] - dispersion["dpy"] * delta_ref
    return frame


def load_tune_map_case(
    qx: float,
    qy: float,
    xi_x: float,
    xi_y: float,
    map_case: str,
) -> TuneMap | None:
    candidate = (
        REPO_ROOT
        / "tune_scan_workflow"
        / "SweepTrajectoryMaps"
        / "ChromaScanY"
        / map_case
        / tune_map_filename(qx, qy, xi_x=xi_x, xi_y=xi_y)
    )
    if candidate.exists():
        return TuneMap.load(str(candidate))
    return None


def export_dead_particle_deltas(outdir: Path, particles, tune_map: TuneMap | None) -> None:
    state = np.asarray(particles.state)
    dead_mask = state <= 0
    dead_particle_id = np.flatnonzero(dead_mask)
    dead_delta = np.asarray(particles.delta)[dead_mask].astype(float, copy=False)
    dead_turn = np.asarray(particles.at_turn)[dead_mask].astype(int, copy=False)

    frame_dict: dict[str, np.ndarray] = {
        "particle_id": dead_particle_id,
        "at_turn": dead_turn,
        "delta": dead_delta,
    }
    if tune_map is not None and dead_delta.size > 0:
        clipped = np.clip(dead_delta, tune_map.delta_min, tune_map.delta_max)
        qx_dead, qy_dead = tune_map(clipped)
        frame_dict["qx_estimate_without_errors"] = np.asarray(qx_dead)
        frame_dict["qy_estimate_without_errors"] = np.asarray(qy_dead)
        frame_dict["delta_clipped_to_map"] = clipped

    dead_frame = pd.DataFrame(frame_dict)
    write_frame_outputs(outdir / "dead_particles", dead_frame)
    plot_dead_particle_distributions(outdir, dead_frame)
    plot_dead_particle_tune_diagram(outdir, dead_frame, tune_map)


def compute_loss_curve_from_summary(summary_frame: pd.DataFrame, num_particles: int) -> pd.DataFrame:
    alive = summary_frame["alive_count"].to_numpy(dtype=float)
    lost_count = np.zeros_like(alive, dtype=float)
    lost_count[1:] = np.clip(alive[:-1] - alive[1:], a_min=0, a_max=None)
    return pd.DataFrame(
        {
            "turn": summary_frame["turn"].to_numpy(dtype=int),
            "delta": summary_frame["delta_from_sweep"].to_numpy(dtype=float),
            "lost_count": lost_count.astype(int),
            "surviving_fraction": alive / num_particles,
        }
    )


def resolve_sweep_settings(
    plane: str,
    num_turns: int,
    total_sweep_hz: float | None,
    sweep_per_turn_hz: float | None,
) -> tuple[float, float, float]:
    if total_sweep_hz is None and sweep_per_turn_hz is None:
        total_sweep_hz = 6000.0

    if sweep_per_turn_hz is not None:
        signed_sweep_per_turn_hz = repo_convention_signed_sweep(sweep_per_turn_hz, plane)
        signed_total_sweep_hz = signed_sweep_per_turn_hz * num_turns
        requested_total_sweep_hz = abs(signed_total_sweep_hz)
    else:
        assert total_sweep_hz is not None
        signed_total_sweep_hz = repo_convention_signed_sweep(total_sweep_hz, plane)
        signed_sweep_per_turn_hz = signed_total_sweep_hz / num_turns
        requested_total_sweep_hz = total_sweep_hz

    return requested_total_sweep_hz, signed_total_sweep_hz, signed_sweep_per_turn_hz


def sliding_fft_spectrogram(
    signal: np.ndarray,
    window_size: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=float)
    signal = np.nan_to_num(signal, nan=0.0)
    n_turns = signal.size
    if n_turns < window_size or window_size < 8:
        return np.array([]), np.array([]), np.empty((0, 0))

    centers = []
    spectra = []
    window = np.hanning(window_size)
    freqs = np.fft.rfftfreq(window_size, d=1.0)

    for start in range(0, n_turns - window_size + 1, step):
        segment = signal[start:start + window_size]
        segment = segment - np.mean(segment)
        spectrum = np.abs(np.fft.rfft(segment * window))
        centers.append(start + window_size // 2)
        spectra.append(spectrum)

    return np.asarray(centers), freqs, np.asarray(spectra)


def plot_spectrogram(
    summary: dict[str, np.ndarray],
    output_path: Path,
    window_size: int,
    step: int,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, sharey=True, constrained_layout=True)

    for axis, coord, label in zip(axes, ("x_mean", "y_mean"), ("Horizontal", "Vertical")):
        centers, freqs, spectra = sliding_fft_spectrogram(summary[coord], window_size, step)
        if spectra.size == 0:
            axis.set_title(f"{label}: insufficient data")
            continue
        mesh = axis.pcolormesh(
            centers,
            freqs,
            spectra.T,
            shading="auto",
            cmap="magma",
        )
        axis.set_ylabel(f"{label}\nfrequency [turn$^-1$]")
        axis.grid(False)
        fig.colorbar(mesh, ax=axis, pad=0.02, label="FFT amplitude")

    axes[-1].set_xlabel("Turn")
    fig.suptitle(f"Sliding-window spectrum (window={window_size}, step={step})")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def sliding_naff(
    signal: np.ndarray,
    signal_px: np.ndarray | None,
    window_size: int,
    step: int,
    num_harmonics: int,
) -> pd.DataFrame | None:
    try:
        import nafflib
    except ImportError:
        return None

    signal = np.asarray(signal, dtype=float)
    signal = np.nan_to_num(signal, nan=0.0)
    signal_px_arr = None
    if signal_px is not None:
        signal_px_arr = np.asarray(signal_px, dtype=float)
        signal_px_arr = np.nan_to_num(signal_px_arr, nan=0.0)
        if signal_px_arr.size != signal.size:
            raise ValueError("signal and signal_px must have the same length")
    n_turns = signal.size
    if n_turns < window_size or window_size < 16:
        return None

    rows: list[dict[str, float | int]] = []
    for start in range(0, n_turns - window_size + 1, step):
        stop = start + window_size
        segment = signal[start:stop]
        segment = segment - np.mean(segment)
        segment_px = None
        if signal_px_arr is not None:
            segment_px = signal_px_arr[start:stop]
            segment_px = segment_px - np.mean(segment_px)
        if not np.any(np.abs(segment) > 0):
            continue

        try:
            if segment_px is None:
                amplitudes, frequencies = nafflib.harmonics(
                    segment,
                    num_harmonics=num_harmonics,
                    window_order=2,
                    window_type="hann",
                )
            else:
                amplitudes, frequencies = nafflib.harmonics(
                    segment,
                    segment_px,
                    num_harmonics=num_harmonics,
                    window_order=2,
                    window_type="hann",
                )
        except Exception:
            continue

        for harmonic_idx, (amp, freq) in enumerate(zip(amplitudes, frequencies)):
            rows.append(
                {
                    "window_start": start,
                    "window_stop": stop,
                    "window_center": start + window_size // 2,
                    "harmonic": harmonic_idx,
                    "amplitude_abs": float(np.abs(amp)),
                    "amplitude_real": float(np.real(amp)),
                    "amplitude_imag": float(np.imag(amp)),
                    "amplitude_phase": float(np.angle(amp)),
                    "frequency": float(freq),
                }
            )

    if not rows:
        return None

    return pd.DataFrame(rows)


def save_sliding_naff(
    summary: dict[str, np.ndarray],
    base_path: Path,
    window_size: int,
    step: int,
    num_harmonics: int,
) -> None:
    outputs: dict[str, dict[str, object]] = {}

    for plane, coord, coord_px in (
        ("horizontal", "x_mean", "px_mean"),
        ("vertical", "y_mean", "py_mean"),
    ):
        frame = sliding_naff(
            summary[coord],
            summary.get(coord_px),
            window_size=window_size,
            step=step,
            num_harmonics=num_harmonics,
        )
        if frame is None:
            outputs[plane] = {
                "available": False,
                "reason": "nafflib missing, insufficient data, or NAFF failed",
            }
            continue

        plane_path = base_path.parent / f"{base_path.name}_{plane}"
        write_frame_outputs(plane_path, frame)
        outputs[plane] = {
            "available": True,
            "json": str(plane_path.with_suffix(".json").name),
            "parquet": str(plane_path.with_suffix(".parquet").name),
        }

    with (base_path.parent / f"{base_path.name}_index.json").open("w", encoding="utf-8") as fh:
        json.dump(outputs, fh, indent=2)


def build_tune_estimate_from_naff(
    summary_frame: pd.DataFrame,
    signal_columns: dict[str, str],
    window_size: int,
    step: int,
    num_harmonics: int,
) -> pd.DataFrame | None:
    outputs: dict[str, pd.DataFrame] = {}
    for plane, column in signal_columns.items():
        frame = sliding_naff(
            summary_frame[column].to_numpy(dtype=float),
            summary_frame[{"horizontal": "px_beta_mean", "vertical": "py_beta_mean"}[plane]].to_numpy(dtype=float),
            window_size=window_size,
            step=step,
            num_harmonics=num_harmonics,
        )
        if frame is None or frame.empty:
            return None
        outputs[plane] = frame[frame["harmonic"] == 0].copy()
        if outputs[plane].empty:
            return None

    merged = outputs["horizontal"][
        ["window_center", "frequency", "amplitude_abs"]
    ].rename(
        columns={
            "frequency": "qx_estimate",
            "amplitude_abs": "qx_amplitude",
        }
    )
    merged = merged.merge(
        outputs["vertical"][["window_center", "frequency", "amplitude_abs"]].rename(
            columns={
                "frequency": "qy_estimate",
                "amplitude_abs": "qy_amplitude",
            }
        ),
        on="window_center",
        how="inner",
    )

    turn = summary_frame["turn"].to_numpy(dtype=float)
    delta = summary_frame["delta_from_sweep"].to_numpy(dtype=float)
    merged["delta_center"] = np.interp(
        merged["window_center"].to_numpy(dtype=float),
        turn,
        delta,
    )
    return merged


def plot_tune_estimate(
    estimate_frame: pd.DataFrame,
    output_path: Path,
    x_key: str,
    x_label: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    axes[0].plot(estimate_frame[x_key], estimate_frame["qx_estimate"], marker="o", markersize=3, linewidth=1.2)
    axes[0].set_ylabel(r"$Q_x$ estimate")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(estimate_frame[x_key], estimate_frame["qy_estimate"], marker="o", markersize=3, linewidth=1.2)
    axes[1].set_ylabel(r"$Q_y$ estimate")
    axes[1].set_xlabel(x_label)
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("Windowed NAFF tune estimate from dispersion-subtracted centroid")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def add_sweep_map_tunes_to_estimate(
    estimate_frame: pd.DataFrame,
    tune_map: TuneMap | None,
) -> pd.DataFrame:
    if tune_map is None or estimate_frame.empty:
        return estimate_frame.copy()

    frame = estimate_frame.copy()
    delta_center = frame["delta_center"].to_numpy(dtype=float)
    clipped = np.clip(delta_center, tune_map.delta_min, tune_map.delta_max)
    qx_map, qy_map = tune_map(clipped)
    qx_map = np.asarray(qx_map, dtype=float)
    qy_map = np.asarray(qy_map, dtype=float)
    qx_map_frac = np.mod(qx_map, 1.0)
    qy_map_frac = np.mod(qy_map, 1.0)
    frame["delta_center_clipped_to_map"] = clipped
    frame["qx_map"] = qx_map
    frame["qy_map"] = qy_map
    frame["qx_map_fractional"] = qx_map_frac
    frame["qy_map_fractional"] = qy_map_frac
    frame["qx_estimate_abs"] = np.abs(frame["qx_estimate"].to_numpy(dtype=float))
    frame["qy_estimate_abs"] = np.abs(frame["qy_estimate"].to_numpy(dtype=float))
    frame["qx_estimate_full"] = np.floor(qx_map) + frame["qx_estimate"].to_numpy(dtype=float)
    frame["qy_estimate_full"] = np.floor(qy_map) + frame["qy_estimate"].to_numpy(dtype=float)
    frame["qx_estimate_abs_full"] = np.floor(qx_map) + frame["qx_estimate_abs"]
    frame["qy_estimate_abs_full"] = np.floor(qy_map) + frame["qy_estimate_abs"]
    frame["qx_residual"] = frame["qx_estimate"] - frame["qx_map_fractional"]
    frame["qy_residual"] = frame["qy_estimate"] - frame["qy_map_fractional"]
    frame["qx_residual_abs"] = frame["qx_estimate_abs"] - frame["qx_map_fractional"]
    frame["qy_residual_abs"] = frame["qy_estimate_abs"] - frame["qy_map_fractional"]
    return frame


def plot_tune_estimate_vs_sweep_map(
    estimate_frame: pd.DataFrame,
    output_path: Path,
) -> None:
    required = {"delta_center", "qx_estimate", "qy_estimate", "qx_map_fractional", "qy_map_fractional"}
    if estimate_frame.empty or not required.issubset(estimate_frame.columns):
        return

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    axes[0].plot(estimate_frame["delta_center"], estimate_frame["qx_map_fractional"], linewidth=1.8, label="Sweep map frac.")
    axes[0].plot(estimate_frame["delta_center"], estimate_frame["qx_estimate"], marker="o", markersize=3, linewidth=1.2, label="NAFF")
    axes[0].set_ylabel(r"$Q_x$")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(estimate_frame["delta_center"], estimate_frame["qy_map_fractional"], linewidth=1.8, label="Sweep map frac.")
    axes[1].plot(estimate_frame["delta_center"], estimate_frame["qy_estimate"], marker="o", markersize=3, linewidth=1.2, label="NAFF")
    axes[1].set_ylabel(r"$Q_y$")
    axes[1].set_xlabel(r"$\delta$")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    fig.suptitle("NAFF tune estimate compared with sweep-map tune trajectory")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_tune_estimate_abs_vs_sweep_map(
    estimate_frame: pd.DataFrame,
    output_path: Path,
) -> None:
    required = {"delta_center", "qx_estimate_abs", "qy_estimate_abs", "qx_map_fractional", "qy_map_fractional"}
    if estimate_frame.empty or not required.issubset(estimate_frame.columns):
        return

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    axes[0].plot(estimate_frame["delta_center"], estimate_frame["qx_map_fractional"], linewidth=1.8, label="Sweep map frac.")
    axes[0].plot(estimate_frame["delta_center"], estimate_frame["qx_estimate_abs"], marker="o", markersize=3, linewidth=1.2, label="|NAFF|")
    axes[0].set_ylabel(r"$Q_x$")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(estimate_frame["delta_center"], estimate_frame["qy_map_fractional"], linewidth=1.8, label="Sweep map frac.")
    axes[1].plot(estimate_frame["delta_center"], estimate_frame["qy_estimate_abs"], marker="o", markersize=3, linewidth=1.2, label="|NAFF|")
    axes[1].set_ylabel(r"$Q_y$")
    axes[1].set_xlabel(r"$\delta$")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    fig.suptitle("Absolute NAFF tune estimate compared with sweep-map tune trajectory")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_naff_tune_diagram(
    estimate_frame: pd.DataFrame,
    tune_map: TuneMap | None,
    output_path: Path,
) -> None:
    required = {"qx_estimate_full", "qy_estimate_full"}
    if tune_map is None or estimate_frame.empty or not required.issubset(estimate_frame.columns):
        return

    d_map, qx_map, qy_map = tune_map.sample(500)
    qx0 = float(qx_map[np.argmin(np.abs(d_map))])
    qy0 = float(qy_map[np.argmin(np.abs(d_map))])

    td = TuneDiagram(qx0=qx0, qy0=qy0, half_range=0.4, max_order=3, skew=True)
    fig, ax = td.plot(figsize=(8, 7), show_working_point=True)
    ax.plot(qx_map, qy_map, color="tab:blue", linewidth=2.0, label="Sweep map")
    scatter = ax.scatter(
        estimate_frame["qx_estimate_full"].to_numpy(dtype=float),
        estimate_frame["qy_estimate_full"].to_numpy(dtype=float),
        c=estimate_frame["delta_center"].to_numpy(dtype=float),
        cmap="plasma",
        s=20,
        alpha=0.8,
        zorder=6,
        label="NAFF estimate",
    )
    td.finalize(ax, extra_handles=None)
    ax.legend(loc="best", frameon=True)
    fig.colorbar(scatter, ax=ax, pad=0.02, label=r"$\delta$ at NAFF window centre")
    fig.suptitle("NAFF tune estimates on the selected tune diagram")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_naff_abs_tune_diagram(
    estimate_frame: pd.DataFrame,
    tune_map: TuneMap | None,
    output_path: Path,
) -> None:
    required = {"qx_estimate_abs_full", "qy_estimate_abs_full"}
    if tune_map is None or estimate_frame.empty or not required.issubset(estimate_frame.columns):
        return

    d_map, qx_map, qy_map = tune_map.sample(500)
    qx0 = float(qx_map[np.argmin(np.abs(d_map))])
    qy0 = float(qy_map[np.argmin(np.abs(d_map))])

    td = TuneDiagram(qx0=qx0, qy0=qy0, half_range=0.4, max_order=3, skew=True)
    fig, ax = td.plot(figsize=(8, 7), show_working_point=True)
    ax.plot(qx_map, qy_map, color="tab:blue", linewidth=2.0, label="Sweep map")
    scatter = ax.scatter(
        estimate_frame["qx_estimate_abs_full"].to_numpy(dtype=float),
        estimate_frame["qy_estimate_abs_full"].to_numpy(dtype=float),
        c=estimate_frame["delta_center"].to_numpy(dtype=float),
        cmap="plasma",
        s=20,
        alpha=0.8,
        zorder=6,
        label="|NAFF| estimate",
    )
    td.finalize(ax, extra_handles=None)
    ax.legend(loc="best", frameon=True)
    fig.colorbar(scatter, ax=ax, pad=0.02, label=r"$\delta$ at NAFF window centre")
    fig.suptitle("Absolute NAFF tune estimates on the selected tune diagram")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_naff_tracks(
    summary: dict[str, np.ndarray],
    output_path: Path,
    window_size: int,
    step: int,
    num_harmonics: int,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, sharey=True, constrained_layout=True)
    any_available = False

    for axis, plane, coord, coord_px, label in zip(
        axes,
        ("horizontal", "vertical"),
        ("x_mean", "y_mean"),
        ("px_mean", "py_mean"),
        ("Horizontal", "Vertical"),
    ):
        frame = sliding_naff(
            summary[coord],
            summary.get(coord_px),
            window_size=window_size,
            step=step,
            num_harmonics=num_harmonics,
        )
        if frame is None or frame.empty:
            axis.set_title(f"{label}: NAFF unavailable")
            continue

        any_available = True
        for harmonic in sorted(frame["harmonic"].unique()):
            sub = frame[frame["harmonic"] == harmonic]
            axis.plot(
                sub["window_center"],
                sub["frequency"],
                marker="o",
                markersize=2,
                linewidth=1,
                label=f"h{harmonic}",
            )
        axis.set_ylabel(f"{label}\nfrequency [turn$^-1$]")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Turn")
    fig.suptitle(
        f"Sliding-window NAFF tracks (window={window_size}, step={step}, harmonics={num_harmonics})"
    )
    if any_available:
        fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_naff_harmonics_positive_vs_sweep_map(
    horizontal_frame: pd.DataFrame | None,
    vertical_frame: pd.DataFrame | None,
    summary_frame: pd.DataFrame,
    tune_map: TuneMap | None,
    output_path: Path,
) -> None:
    if tune_map is None:
        return
    if horizontal_frame is None or vertical_frame is None:
        return
    if horizontal_frame.empty or vertical_frame.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, constrained_layout=True)
    turn = summary_frame["turn"].to_numpy(dtype=float)
    delta = summary_frame["delta_from_sweep"].to_numpy(dtype=float)

    for axis, frame, qcol, label in (
        (axes[0], horizontal_frame, "qx", "Horizontal"),
        (axes[1], vertical_frame, "qy", "Vertical"),
    ):
        frame = frame.copy()
        frame["delta_center"] = np.interp(frame["window_center"].to_numpy(dtype=float), turn, delta)
        clipped = np.clip(frame["delta_center"].to_numpy(dtype=float), tune_map.delta_min, tune_map.delta_max)
        qx_map, qy_map = tune_map(clipped)
        map_fractional = np.mod(qx_map if qcol == "qx" else qy_map, 1.0)
        frame["map_fractional"] = np.asarray(map_fractional, dtype=float)
        frame = frame[frame["frequency"] > 0].copy()
        if frame.empty:
            axis.set_ylabel(f"{label}\nfrequency")
            axis.grid(True, alpha=0.25)
            continue

        harmonic_values = sorted(frame["harmonic"].unique())
        cmap = plt.get_cmap("tab10")
        for idx, harmonic in enumerate(harmonic_values):
            sub = frame[frame["harmonic"] == harmonic]
            axis.plot(
                sub["delta_center"],
                sub["frequency"],
                marker="o",
                markersize=2.5,
                linewidth=1.0,
                color=cmap(idx % 10),
                alpha=0.8,
                label=f"h{harmonic}",
            )

        reference = (
            frame[["delta_center", "map_fractional"]]
            .drop_duplicates(subset=["delta_center"])
            .sort_values("delta_center")
        )
        axis.plot(
            reference["delta_center"],
            reference["map_fractional"],
            color="black",
            linewidth=2.0,
            linestyle="--",
            label="Sweep map frac.",
        )
        axis.set_ylabel(f"{label}\nfrequency")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", ncol=4, fontsize=8)

    axes[-1].set_xlabel(r"$\delta$")
    fig.suptitle("Positive NAFF harmonics compared with sweep-map fractional tune")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_intensity_loss(loss_curve: dict[str, np.ndarray], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(loss_curve["delta"] * 1e3, loss_curve["surviving_fraction"], color="tab:blue")
    ax.set_xlabel(r"$\delta$ [$10^{-3}$]")
    ax.set_ylabel("Normalised intensity")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_delta_envelope(summary_frame: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5), constrained_layout=True)
    x = summary_frame["delta_from_sweep"].to_numpy(dtype=float)
    delta_mean = summary_frame["delta_mean"].to_numpy()
    delta_min = summary_frame["delta_min"].to_numpy()
    delta_max = summary_frame["delta_max"].to_numpy()

    lo = float(np.nanmin(np.concatenate([x, delta_min, delta_max])))
    hi = float(np.nanmax(np.concatenate([x, delta_min, delta_max])))
    ax.plot([lo, hi], [lo, hi], color="0.3", linestyle="--", linewidth=1.2, label="1:1 line")
    ax.plot(x, delta_mean, color="tab:blue", linewidth=1.5, label=r"mean actual $\delta$")
    ax.plot(x, delta_min, color="tab:orange", linewidth=1.1, label=r"min actual $\delta$")
    ax.plot(x, delta_max, color="tab:green", linewidth=1.1, label=r"max actual $\delta$")
    ax.fill_between(x, delta_min, delta_max, color="tab:blue", alpha=0.12)
    ax.set_xlabel(r"Sweep estimate $\delta$ from df_to_delta")
    ax.set_ylabel(r"Actual particle $\delta$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.suptitle("Actual particle delta vs imposed sweep delta")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_dead_particle_distributions(outdir: Path, dead_frame: pd.DataFrame) -> None:
    if dead_frame.empty:
        return

    columns_to_plot: list[tuple[str, str]] = [("delta", r"Dead-particle $\delta$")]
    if "qx_estimate_without_errors" in dead_frame.columns:
        columns_to_plot.append(("qx_estimate_without_errors", r"Estimated dead-particle $Q_x$"))
    if "qy_estimate_without_errors" in dead_frame.columns:
        columns_to_plot.append(("qy_estimate_without_errors", r"Estimated dead-particle $Q_y$"))

    fig, axes = plt.subplots(1, len(columns_to_plot), figsize=(5.0 * len(columns_to_plot), 5), constrained_layout=True)
    if len(columns_to_plot) == 1:
        axes = [axes]

    for axis, (column, title) in zip(axes, columns_to_plot):
        values = dead_frame[column].to_numpy(dtype=float)
        axis.violinplot([values], positions=[1], showmeans=True, showextrema=True, widths=0.7)
        axis.set_xticks([1])
        axis.set_xticklabels([column], rotation=20, ha="right")
        axis.set_title(title)
        axis.grid(True, alpha=0.2)

    fig.suptitle("Dead-particle distribution summary")
    fig.savefig(outdir / "dead_particle_violin_distributions.png", dpi=180)
    plt.close(fig)


def plot_dead_particle_tune_diagram(outdir: Path, dead_frame: pd.DataFrame, tune_map: TuneMap | None) -> None:
    required = {"qx_estimate_without_errors", "qy_estimate_without_errors"}
    if tune_map is None or not required.issubset(dead_frame.columns):
        return
    if dead_frame.empty:
        return

    qx_dead = dead_frame["qx_estimate_without_errors"].to_numpy(dtype=float)
    qy_dead = dead_frame["qy_estimate_without_errors"].to_numpy(dtype=float)
    d_map, qx_map, qy_map = tune_map.sample(500)

    qx0 = float(qx_map[np.argmin(np.abs(d_map))])
    qy0 = float(qy_map[np.argmin(np.abs(d_map))])
    td = TuneDiagram(qx0=qx0, qy0=qy0, half_range=0.4, max_order=3, skew=True)
    fig, ax = td.plot(figsize=(8, 7), show_working_point=True)
    ax.plot(qx_map, qy_map, color="tab:blue", linewidth=2.0, label="Sweep map")
    scatter = ax.scatter(
        qx_dead,
        qy_dead,
        c=dead_frame["delta"].to_numpy(dtype=float),
        cmap="viridis",
        s=18,
        alpha=0.7,
        label="Dead particles",
        zorder=6,
    )
    td.finalize(ax, extra_handles=None)
    ax.legend(loc="best", frameon=True)
    fig.colorbar(scatter, ax=ax, pad=0.02, label=r"Dead-particle $\delta$")
    fig.suptitle("Dead-particle tune estimates on the selected tune diagram")
    fig.savefig(outdir / "dead_particle_tune_diagram.png", dpi=180)
    plt.close(fig)


def save_particle_snapshot(snapshot_dir: Path, particles, turn: int) -> dict[str, np.ndarray]:
    arrays = alive_particle_arrays(particles)
    payload = {"turn": np.array(turn, dtype=int), **arrays}
    np.savez_compressed(snapshot_dir / f"snapshot_turn_{turn:05d}.npz", **payload)
    return arrays


def plot_violin_evolution(
    violin_dir: Path,
    snapshot_records: list[tuple[int, dict[str, np.ndarray]]],
) -> None:
    valid_records = [(turn, arrays) for turn, arrays in snapshot_records if int(arrays["alive_count"]) > 0]
    if not valid_records:
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.ravel()
    positions = np.arange(1, len(valid_records) + 1)
    turn_labels = [str(turn) for turn, _ in valid_records]

    for axis, coord in zip(axes, COORDINATES):
        data = [arrays[coord] for _, arrays in valid_records]
        axis.violinplot(dataset=data, positions=positions, showmeans=True, showextrema=True, widths=0.8)
        axis.set_title(coord)
        axis.set_xticks(positions)
        axis.set_xticklabels(turn_labels, rotation=45, ha="right")
        axis.set_xlabel("Turn")
        axis.grid(True, alpha=0.2)

    fig.suptitle("Distribution evolution at saved snapshot turns")
    fig.savefig(violin_dir / "violin_evolution.png", dpi=180)
    plt.close(fig)


def plot_violin_evolution_beta(
    violin_dir: Path,
    snapshot_records: list[tuple[int, dict[str, np.ndarray]]],
    dispersion: dict[str, float],
) -> None:
    valid_records = [(turn, arrays) for turn, arrays in snapshot_records if int(arrays["alive_count"]) > 0]
    if not valid_records:
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.ravel()
    positions = np.arange(1, len(valid_records) + 1)
    turn_labels = [str(turn) for turn, _ in valid_records]
    coord_order = ("x_beta", "px_beta", "y_beta", "py_beta", "zeta", "delta")

    for axis, coord in zip(axes, coord_order):
        data = []
        for _, arrays in valid_records:
            if coord == "x_beta":
                values = arrays["x"] - dispersion["dx"] * arrays["delta"]
            elif coord == "px_beta":
                values = arrays["px"] - dispersion["dpx"] * arrays["delta"]
            elif coord == "y_beta":
                values = arrays["y"] - dispersion["dy"] * arrays["delta"]
            elif coord == "py_beta":
                values = arrays["py"] - dispersion["dpy"] * arrays["delta"]
            else:
                values = arrays[coord]
            data.append(values)

        axis.violinplot(dataset=data, positions=positions, showmeans=True, showextrema=True, widths=0.8)
        axis.set_title(coord)
        axis.set_xticks(positions)
        axis.set_xticklabels(turn_labels, rotation=45, ha="right")
        axis.set_xlabel("Turn")
        axis.grid(True, alpha=0.2)

    fig.suptitle("Dispersion-subtracted distribution evolution at saved snapshot turns")
    fig.savefig(violin_dir / "violin_evolution_beta.png", dpi=180)
    plt.close(fig)


def plot_phase_space_snapshot(phase_dir: Path, arrays: dict[str, np.ndarray], turn: int) -> None:
    if int(arrays["alive_count"]) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    pairs = [("x", "px"), ("y", "py"), ("zeta", "delta")]
    color = arrays["delta"]
    for axis, (coord_x, coord_y) in zip(axes, pairs):
        scatter = axis.scatter(
            arrays[coord_x],
            arrays[coord_y],
            c=color,
            s=6,
            alpha=0.7,
            cmap="viridis",
        )
        axis.set_xlabel(coord_x)
        axis.set_ylabel(coord_y)
        axis.grid(True, alpha=0.2)
    fig.colorbar(scatter, ax=axes, pad=0.02, label="delta")
    fig.suptitle(f"Phase-space projections at turn {turn}")
    fig.savefig(phase_dir / f"phase_space_turn_{turn:05d}.png", dpi=180)
    plt.close(fig)


def plot_phase_space_evolution(
    phase_dir: Path,
    snapshot_records: list[tuple[int, dict[str, np.ndarray]]],
) -> None:
    valid_records = [(turn, arrays) for turn, arrays in snapshot_records if int(arrays["alive_count"]) > 0]
    if not valid_records:
        return

    pairs = [("x", "px"), ("y", "py"), ("zeta", "delta")]
    fig, axes = plt.subplots(len(valid_records), 3, figsize=(14, 3.2 * len(valid_records)), constrained_layout=True)

    if len(valid_records) == 1:
        axes = np.array([axes])

    scatter = None
    for row_idx, (turn, arrays) in enumerate(valid_records):
        for col_idx, (coord_x, coord_y) in enumerate(pairs):
            axis = axes[row_idx, col_idx]
            scatter = axis.scatter(
                arrays[coord_x],
                arrays[coord_y],
                c=arrays["delta"],
                s=6,
                alpha=0.7,
                cmap="viridis",
            )
            if row_idx == 0:
                axis.set_title(f"{coord_x} vs {coord_y}")
            axis.set_xlabel(coord_x)
            axis.set_ylabel(f"{coord_y}\nturn {turn}")
            axis.grid(True, alpha=0.2)

    if scatter is not None:
        fig.colorbar(scatter, ax=axes, pad=0.01, label="delta")
    fig.suptitle("Phase-space evolution at saved snapshot turns")
    fig.savefig(phase_dir / "phase_space_evolution.png", dpi=180)
    plt.close(fig)


def plot_phase_space_evolution_beta(
    phase_dir: Path,
    snapshot_records: list[tuple[int, dict[str, np.ndarray]]],
    dispersion: dict[str, float],
) -> None:
    valid_records = [(turn, arrays) for turn, arrays in snapshot_records if int(arrays["alive_count"]) > 0]
    if not valid_records:
        return

    fig, axes = plt.subplots(len(valid_records), 3, figsize=(14, 3.2 * len(valid_records)), constrained_layout=True)

    if len(valid_records) == 1:
        axes = np.array([axes])

    scatter = None
    for row_idx, (turn, arrays) in enumerate(valid_records):
        derived = {
            "x_beta": arrays["x"] - dispersion["dx"] * arrays["delta"],
            "px_beta": arrays["px"] - dispersion["dpx"] * arrays["delta"],
            "y_beta": arrays["y"] - dispersion["dy"] * arrays["delta"],
            "py_beta": arrays["py"] - dispersion["dpy"] * arrays["delta"],
            "zeta": arrays["zeta"],
            "delta": arrays["delta"],
        }
        for col_idx, (coord_x, coord_y) in enumerate((("x_beta", "px_beta"), ("y_beta", "py_beta"), ("zeta", "delta"))):
            axis = axes[row_idx, col_idx]
            scatter = axis.scatter(
                derived[coord_x],
                derived[coord_y],
                c=arrays["delta"],
                s=6,
                alpha=0.7,
                cmap="viridis",
            )
            if row_idx == 0:
                axis.set_title(f"{coord_x} vs {coord_y}")
            axis.set_xlabel(coord_x)
            axis.set_ylabel(f"{coord_y}\nturn {turn}")
            axis.grid(True, alpha=0.2)

    if scatter is not None:
        fig.colorbar(scatter, ax=axes, pad=0.01, label="delta")
    fig.suptitle("Dispersion-subtracted phase-space evolution at saved snapshot turns")
    fig.savefig(phase_dir / "phase_space_evolution_beta.png", dpi=180)
    plt.close(fig)


def plot_phase_space_turn_colored(
    phase_dir: Path,
    snapshot_records: list[tuple[int, dict[str, np.ndarray]]],
) -> None:
    valid_records = [(turn, arrays) for turn, arrays in snapshot_records if int(arrays["alive_count"]) > 0]
    if not valid_records:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    pairs = [("x", "px"), ("y", "py"), ("zeta", "delta")]
    cmap = plt.get_cmap("plasma")
    norm = plt.Normalize(vmin=min(turn for turn, _ in valid_records), vmax=max(turn for turn, _ in valid_records))

    for axis, (coord_x, coord_y) in zip(axes, pairs):
        for turn, arrays in valid_records:
            color = cmap(norm(turn))
            axis.scatter(
                arrays[coord_x],
                arrays[coord_y],
                color=color,
                s=6,
                alpha=0.35,
            )
        axis.set_xlabel(coord_x)
        axis.set_ylabel(coord_y)
        axis.grid(True, alpha=0.2)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, pad=0.02, label="turn")
    fig.suptitle("Phase-space overlay coloured by snapshot turn")
    fig.savefig(phase_dir / "phase_space_turn_colored_overlay.png", dpi=180)
    plt.close(fig)


def plot_phase_space_turn_colored_beta(
    phase_dir: Path,
    snapshot_records: list[tuple[int, dict[str, np.ndarray]]],
    dispersion: dict[str, float],
) -> None:
    valid_records = [(turn, arrays) for turn, arrays in snapshot_records if int(arrays["alive_count"]) > 0]
    if not valid_records:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    cmap = plt.get_cmap("plasma")
    norm = plt.Normalize(vmin=min(turn for turn, _ in valid_records), vmax=max(turn for turn, _ in valid_records))

    for axis, (coord_x, coord_y) in zip(axes, (("x_beta", "px_beta"), ("y_beta", "py_beta"), ("zeta", "delta"))):
        for turn, arrays in valid_records:
            derived = {
                "x_beta": arrays["x"] - dispersion["dx"] * arrays["delta"],
                "px_beta": arrays["px"] - dispersion["dpx"] * arrays["delta"],
                "y_beta": arrays["y"] - dispersion["dy"] * arrays["delta"],
                "py_beta": arrays["py"] - dispersion["dpy"] * arrays["delta"],
                "zeta": arrays["zeta"],
                "delta": arrays["delta"],
            }
            color = cmap(norm(turn))
            axis.scatter(
                derived[coord_x],
                derived[coord_y],
                color=color,
                s=6,
                alpha=0.35,
            )
        axis.set_xlabel(coord_x)
        axis.set_ylabel(coord_y)
        axis.grid(True, alpha=0.2)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, pad=0.02, label="turn")
    fig.suptitle("Dispersion-subtracted phase-space overlay coloured by snapshot turn")
    fig.savefig(phase_dir / "phase_space_turn_colored_overlay_beta.png", dpi=180)
    plt.close(fig)


def plot_moment_family(summary_frame: pd.DataFrame, output_path: Path, suffix: str) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True, constrained_layout=True)
    axes = axes.ravel()
    x = summary_frame["turn"].to_numpy()
    for axis, coord in zip(axes, COORDINATES):
        axis.plot(x, summary_frame[f"{coord}_{suffix}"], linewidth=1.2)
        axis.set_title(coord)
        axis.grid(True, alpha=0.25)
    axes[-2].set_xlabel("Turn")
    axes[-1].set_xlabel("Turn")
    fig.suptitle(f"{suffix} evolution at the line start")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


@dataclass
class RunConfig:
    line_path: str
    qx: float
    qy: float
    xi_x: float
    xi_y: float
    error_variant: str
    tune_map_case: str
    plane: str
    total_sweep_hz: float
    sweep_per_turn_hz_input: float | None
    num_turns: int
    num_particles: int
    nemitt_x: float
    nemitt_y: float
    sigma_z: float
    point_monitor_element: str | None
    snapshot_every: int
    loss_snapshot_every: int
    naff_harmonics: int
    fft_window: int
    fft_step: int
    omp_threads: str
    output_dir: str
    signed_sweep_hz: float
    sweep_per_turn_hz: float
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RF-sweep diagnostics with turn-by-turn beam summaries."
    )
    parser.add_argument("--line-path", default=DEFAULT_LINE_PATH)
    parser.add_argument("--qx", type=float, default=20.13)
    parser.add_argument("--qy", type=float, default=20.18)
    parser.add_argument("--xi-x", type=float, default=0.5)
    parser.add_argument("--xi-y", type=float, default=0.5)
    parser.add_argument("--error-variant", choices=sorted(error_variants), default="none")
    parser.add_argument("--tune-map-case", choices=["WithErrors", "WithoutErrors", "Simplified"], default=None)
    parser.add_argument("--plane", choices=["DPpos", "DPneg"], default="DPneg")
    parser.add_argument("--total-sweep-hz", type=float, default=None)
    parser.add_argument("--sweep-per-turn-hz", type=float, default=None)
    parser.add_argument("--num-turns", type=int, default=6000)
    parser.add_argument("--num-particles", type=int, default=512)
    parser.add_argument("--nemitt-x", type=float, default=2e-6)
    parser.add_argument("--nemitt-y", type=float, default=2e-6)
    parser.add_argument("--sigma-z", type=float, default=0.224)
    parser.add_argument("--point-monitor-element", default=None)
    parser.add_argument("--snapshot-every", type=int, default=500)
    parser.add_argument("--loss-snapshot-every", type=int, default=100)
    parser.add_argument("--naff-harmonics", type=int, default=5)
    parser.add_argument("--fft-window", type=int, default=256)
    parser.add_argument("--fft-step", type=int, default=64)
    parser.add_argument("--omp-threads", default="0")
    parser.add_argument("--output-base", default="rf_sweep_speed_outputs")
    parser.add_argument("--case-name", default=None)
    parser.add_argument(
        "--note",
        default=(
            "Plane-to-sweep sign follows the existing repository convention: "
            "DPpos -> negative sweep, DPneg -> positive sweep."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tune_map_case = args.tune_map_case
    if tune_map_case is None:
        tune_map_case = "WithErrors" if args.error_variant != "none" else "WithoutErrors"
    total_sweep_hz, signed_sweep_hz, sweep_per_turn_hz = resolve_sweep_settings(
        plane=args.plane,
        num_turns=args.num_turns,
        total_sweep_hz=args.total_sweep_hz,
        sweep_per_turn_hz=args.sweep_per_turn_hz,
    )

    base_dir = Path(__file__).resolve().parent / args.output_base
    outdir = build_output_dir(base_dir=base_dir, case_name=args.case_name)

    run_config = RunConfig(
        line_path=args.line_path,
        qx=args.qx,
        qy=args.qy,
        xi_x=args.xi_x,
        xi_y=args.xi_y,
        error_variant=args.error_variant,
        tune_map_case=tune_map_case,
        plane=args.plane,
        total_sweep_hz=total_sweep_hz,
        sweep_per_turn_hz_input=args.sweep_per_turn_hz,
        num_turns=args.num_turns,
        num_particles=args.num_particles,
        nemitt_x=args.nemitt_x,
        nemitt_y=args.nemitt_y,
        sigma_z=args.sigma_z,
        point_monitor_element=args.point_monitor_element,
        snapshot_every=args.snapshot_every,
        loss_snapshot_every=args.loss_snapshot_every,
        naff_harmonics=args.naff_harmonics,
        fft_window=args.fft_window,
        fft_step=args.fft_step,
        omp_threads=args.omp_threads,
        output_dir=str(outdir),
        signed_sweep_hz=signed_sweep_hz,
        sweep_per_turn_hz=sweep_per_turn_hz,
        note=args.note,
    )

    with (outdir / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(run_config), fh, indent=2)

    print(f"[rf_sweep_speed_scan] Output directory: {outdir}")
    print(
        "[rf_sweep_speed_scan] Sweep setup: "
        f"plane={args.plane}, num_turns={args.num_turns}, "
        f"sweep_per_turn={sweep_per_turn_hz} Hz/turn, total_sweep={signed_sweep_hz} Hz"
    )
    tune_map = load_tune_map_case(args.qx, args.qy, args.xi_x, args.xi_y, tune_map_case)
    if tune_map is None:
        print(f"[rf_sweep_speed_scan] No {tune_map_case} tune map found for dead-particle tune estimates")
    else:
        print(f"[rf_sweep_speed_scan] Loaded {tune_map_case} tune map for dead-particle tune estimates")
    line = configure_line(
        line_path=args.line_path,
        qx=args.qx,
        qy=args.qy,
        xi_x=args.xi_x,
        xi_y=args.xi_y,
        omp_threads=args.omp_threads,
        point_monitor_element=args.point_monitor_element,
        error_variant_name=args.error_variant,
    )
    observation_dispersion = get_observation_dispersion(line)
    print(
        "[rf_sweep_speed_scan] Observation dispersion: "
        f"Dx={observation_dispersion['dx']}, Dpx={observation_dispersion['dpx']}, "
        f"Dy={observation_dispersion['dy']}, Dpy={observation_dispersion['dpy']}"
    )
    particles = generate_particles(
        line=line,
        num_particles=args.num_particles,
        nemitt_x=args.nemitt_x,
        nemitt_y=args.nemitt_y,
        sigma_z=args.sigma_z,
    )
    print(
        "[rf_sweep_speed_scan] Generated bunch: "
        f"{args.num_particles} particles, sigma_z={args.sigma_z}, "
        f"nemitt_x={args.nemitt_x}, nemitt_y={args.nemitt_y}"
    )

    rf_sweep = xc.RFSweep(line)
    rf_sweep.prepare(sweep_per_turn=sweep_per_turn_hz)
    print("[rf_sweep_speed_scan] Prepared RF sweep")

    snapshot_dir = outdir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    violin_dir = outdir / "violin_plots"
    violin_dir.mkdir(exist_ok=True)
    phase_dir = outdir / "phase_space_plots"
    phase_dir.mkdir(exist_ok=True)

    snapshot_turns = set(np.arange(0, args.num_turns + 1, args.snapshot_every, dtype=int).tolist())
    snapshot_records: list[tuple[int, dict[str, np.ndarray]]] = []
    summary_rows: list[dict[str, float | int]] = []
    summary_rows.append(collect_turn_row(particles, turn=0, sweep_per_turn_hz=sweep_per_turn_hz))
    if 0 in snapshot_turns:
        arrays0 = save_particle_snapshot(snapshot_dir, particles, turn=0)
        snapshot_records.append((0, arrays0))

    if hasattr(line, "scattering"):
        line.scattering.enable()

    print("[rf_sweep_speed_scan] Starting one-turn tracking loop")
    loss_snapshot_turns: set[int] = set()
    previous_alive = int(summary_rows[0]["alive_count"])
    first_loss_turn: int | None = None
    for turn in range(1, args.num_turns + 1):
        line.track(particles=particles, num_turns=1)
        row = collect_turn_row(particles, turn=turn, sweep_per_turn_hz=sweep_per_turn_hz)
        summary_rows.append(row)
        current_alive = int(row["alive_count"])
        if first_loss_turn is None and current_alive < previous_alive:
            first_loss_turn = turn
            print(f"[rf_sweep_speed_scan] First losses detected at turn {turn}")
            loss_snapshot_turns = set(
                np.arange(turn, args.num_turns + 1, args.loss_snapshot_every, dtype=int).tolist()
            )
        previous_alive = current_alive

        snapshot_taken = False
        if turn in snapshot_turns or turn in loss_snapshot_turns:
            arrays = save_particle_snapshot(snapshot_dir, particles, turn=turn)
            snapshot_records.append((turn, arrays))
            snapshot_taken = True
        if current_alive <= 1:
            if not snapshot_taken:
                arrays = save_particle_snapshot(snapshot_dir, particles, turn=turn)
                snapshot_records.append((turn, arrays))
            print(f"[rf_sweep_speed_scan] Stopping early at turn {turn} because alive_count={current_alive}")
            break

    if hasattr(line, "scattering"):
        line.scattering.disable()
    print("[rf_sweep_speed_scan] Tracking finished, writing summaries and plots")

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame = add_dispersion_subtracted_columns(summary_frame, observation_dispersion)
    write_frame_outputs(outdir / "turn_summary", summary_frame)
    plot_moment_family(summary_frame, outdir / "mean_evolution.png", "mean")
    plot_moment_family(summary_frame, outdir / "std_evolution.png", "std")
    plot_delta_envelope(summary_frame, outdir / "delta_envelope_vs_sweep_delta.png")
    plot_violin_evolution(violin_dir, snapshot_records)
    plot_violin_evolution_beta(violin_dir, snapshot_records, observation_dispersion)
    plot_phase_space_evolution(phase_dir, snapshot_records)
    plot_phase_space_evolution_beta(phase_dir, snapshot_records, observation_dispersion)
    plot_phase_space_turn_colored(phase_dir, snapshot_records)
    plot_phase_space_turn_colored_beta(phase_dir, snapshot_records, observation_dispersion)

    loss_curve_frame = compute_loss_curve_from_summary(summary_frame, num_particles=args.num_particles)
    write_frame_outputs(outdir / "intensity_loss", loss_curve_frame)
    plot_intensity_loss(
        {
            "delta": loss_curve_frame["delta"].to_numpy(),
            "surviving_fraction": loss_curve_frame["surviving_fraction"].to_numpy(),
        },
        outdir / "intensity_loss_vs_delta.png",
    )

    summary = {column: summary_frame[column].to_numpy() for column in summary_frame.columns}
    plot_spectrogram(
        summary,
        outdir / "centroid_spectrogram.png",
        window_size=args.fft_window,
        step=args.fft_step,
    )
    plot_spectrogram(
        {
            "x_mean": summary_frame["x_beta_mean"].to_numpy(),
            "y_mean": summary_frame["y_beta_mean"].to_numpy(),
        },
        outdir / "centroid_spectrogram_beta.png",
        window_size=args.fft_window,
        step=args.fft_step,
    )
    save_sliding_naff(
        summary,
        outdir / "sliding_naff_global",
        window_size=args.fft_window,
        step=args.fft_step,
        num_harmonics=args.naff_harmonics,
    )
    save_sliding_naff(
        {
            "x_mean": summary_frame["x_beta_mean"].to_numpy(),
            "y_mean": summary_frame["y_beta_mean"].to_numpy(),
        },
        outdir / "sliding_naff_beta",
        window_size=args.fft_window,
        step=args.fft_step,
        num_harmonics=args.naff_harmonics,
    )
    plot_naff_tracks(
        summary,
        outdir / "sliding_naff_global.png",
        window_size=args.fft_window,
        step=args.fft_step,
        num_harmonics=args.naff_harmonics,
    )
    plot_naff_tracks(
        {
            "x_mean": summary_frame["x_beta_mean"].to_numpy(),
            "y_mean": summary_frame["y_beta_mean"].to_numpy(),
        },
        outdir / "sliding_naff_beta.png",
        window_size=args.fft_window,
        step=args.fft_step,
        num_harmonics=args.naff_harmonics,
    )
    tune_estimate = build_tune_estimate_from_naff(
        summary_frame,
        signal_columns={"horizontal": "x_beta_mean", "vertical": "y_beta_mean"},
        window_size=args.fft_window,
        step=args.fft_step,
        num_harmonics=args.naff_harmonics,
    )
    if tune_estimate is not None:
        write_frame_outputs(outdir / "tune_estimate", tune_estimate)
        plot_tune_estimate(
            tune_estimate,
            outdir / "tune_estimate_vs_turn.png",
            x_key="window_center",
            x_label="Turn",
        )
        plot_tune_estimate(
            tune_estimate,
            outdir / "tune_estimate_vs_delta.png",
            x_key="delta_center",
            x_label=r"$\delta$",
        )

    with (outdir / "death_turns.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "sweep_per_turn": sweep_per_turn_hz,
                "at_turn": np.asarray(particles.at_turn).tolist(),
                "final_state": np.asarray(particles.state).tolist(),
                "observation_dispersion": observation_dispersion,
            },
            fh,
            indent=2,
        )

    export_dead_particle_deltas(outdir, particles, tune_map)

    np.savez_compressed(
        outdir / "centroid_signals.npz",
        turn=summary_frame["turn"].to_numpy(),
        delta_from_sweep=summary_frame["delta_from_sweep"].to_numpy(),
        x_mean=summary_frame["x_mean"].to_numpy(),
        y_mean=summary_frame["y_mean"].to_numpy(),
        x_beta_mean=summary_frame["x_beta_mean"].to_numpy(),
        px_beta_mean=summary_frame["px_beta_mean"].to_numpy(),
        y_beta_mean=summary_frame["y_beta_mean"].to_numpy(),
        py_beta_mean=summary_frame["py_beta_mean"].to_numpy(),
        x_std=summary_frame["x_std"].to_numpy(),
        y_std=summary_frame["y_std"].to_numpy(),
        alive_count=summary_frame["alive_count"].to_numpy(),
    )
    print("[rf_sweep_speed_scan] Done")


if __name__ == "__main__":
    main()
