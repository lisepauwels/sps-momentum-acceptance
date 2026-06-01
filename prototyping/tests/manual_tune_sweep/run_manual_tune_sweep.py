from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import xcoll as xc  # noqa: F401
import xobjects as xo
import xpart as xp
import xtrack as xt


HERE = Path(__file__).resolve().parent
TESTS_DIR = HERE.parent
REPO_ROOT = TESTS_DIR.parent
RF_SWEEP_DIR = TESTS_DIR / "rf_sweep_speed_tests"
SEXT_DIR = TESTS_DIR / "sextupoles_off"
HELPER_DIR = REPO_ROOT / "helper_functions"
WORKFLOW_DIR = REPO_ROOT / "tune_scan_workflow"

sys.path.insert(0, str(WORKFLOW_DIR))
sys.path.insert(0, str(HELPER_DIR))
sys.path.insert(0, str(RF_SWEEP_DIR))
sys.path.insert(0, str(SEXT_DIR))

from rf_sweep_speed_scan import (
    COORDINATES,
    DEFAULT_LINE_PATH,
    add_dispersion_subtracted_columns,
    alive_particle_arrays,
    collect_turn_row,
    particle_stats,
    plot_moment_family,
    plot_naff_tracks,
    plot_phase_space_evolution,
    plot_phase_space_evolution_beta,
    plot_phase_space_turn_colored,
    plot_phase_space_turn_colored_beta,
    plot_spectrogram,
    plot_violin_evolution,
    plot_violin_evolution_beta,
    save_particle_snapshot,
    save_sliding_naff,
    sliding_naff,
    write_frame_outputs,
)
from recompute_saved_tune_analysis import build_tune_estimate_position_only
from tune_diagram import TuneDiagram
from run_sextupoles_off_monitor_scan import (
    apply_error_configuration,
    zero_sextupoles,
)

SCHEDULE_CACHE_VERSION = 1


@dataclass
class RunConfig:
    line_path: str
    qx: float
    qy: float
    xi_x: float
    xi_y: float
    sextupoles_mode: str
    error_variant: str
    dq_per_turn_x: float
    dq_per_turn_y: float
    num_turns: int
    num_particles: int
    nemitt_x: float
    nemitt_y: float
    sigma_z: float
    snapshot_every: int
    progress_every: int
    schedule_points: int
    naff_harmonics: int
    fft_window: int
    fft_step: int
    omp_threads: str
    output_dir: str
    schedule_cache_key: str | None
    schedule_cache_dir: str | None
    note: str


@dataclass
class ScheduleStatus:
    completed_full_path: bool
    failure_turn: float | None
    failure_qx_target: float | None
    failure_qy_target: float | None
    matched_points: int
    final_schedule_turn: float
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Manual tune sweep driven directly by dq_per_turn. "
            "At each turn, the script sets kqf0/kqd0 to follow a requested tune path, "
            "then tracks one turn without RF sweep."
        )
    )
    parser.add_argument("--line-path", default=DEFAULT_LINE_PATH)
    parser.add_argument("--qx", type=float, default=20.13)
    parser.add_argument("--qy", type=float, default=20.18)
    parser.add_argument("--xi-x", type=float, default=0.5)
    parser.add_argument("--xi-y", type=float, default=0.5)
    parser.add_argument("--sextupoles-mode", choices=["on", "off"], default="off")
    parser.add_argument("--error-variant", default="none")
    parser.add_argument("--dq-per-turn-x", type=float, default=0.0)
    parser.add_argument("--dq-per-turn-y", type=float, default=0.0)
    parser.add_argument("--num-turns", type=int, default=6000)
    parser.add_argument("--num-particles", type=int, default=100)
    parser.add_argument("--nemitt-x", type=float, default=2e-6)
    parser.add_argument("--nemitt-y", type=float, default=2e-6)
    parser.add_argument("--sigma-z", type=float, default=0.224)
    parser.add_argument("--snapshot-every", type=int, default=500)
    parser.add_argument("--progress-every", type=int, default=300)
    parser.add_argument("--schedule-points", type=int, default=101)
    parser.add_argument("--naff-harmonics", type=int, default=5)
    parser.add_argument("--fft-window", type=int, default=256)
    parser.add_argument("--fft-step", type=int, default=64)
    parser.add_argument("--omp-threads", default="0")
    parser.add_argument("--output-base", default="manual_tune_sweep_outputs")
    parser.add_argument(
        "--force-rebuild-schedule",
        action="store_true",
        help="Ignore any cached kqf0/kqd0 schedule and solve it again.",
    )
    parser.add_argument(
        "--batch-name",
        default=f"manual_tune_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--note",
        default=(
            "Direct dq-per-turn manual tune sweep. "
            "Target tunes are imposed through kqf0/kqd0, with optional sextupoles and errors."
        ),
    )
    return parser.parse_args()


def build_schedule_cache_key(
    *,
    line_path: str,
    qx: float,
    qy: float,
    xi_x: float,
    xi_y: float,
    sextupoles_mode: str,
    error_variant: str,
    dq_per_turn_x: float,
    dq_per_turn_y: float,
    num_turns: int,
    schedule_points: int,
) -> str:
    line_file = Path(line_path).expanduser().resolve()
    stat = line_file.stat()
    payload = {
        "version": SCHEDULE_CACHE_VERSION,
        "line_path": str(line_file),
        "line_mtime_ns": stat.st_mtime_ns,
        "line_size": stat.st_size,
        "qx": qx,
        "qy": qy,
        "xi_x": xi_x,
        "xi_y": xi_y,
        "sextupoles_mode": sextupoles_mode,
        "error_variant": error_variant,
        "dq_per_turn_x": dq_per_turn_x,
        "dq_per_turn_y": dq_per_turn_y,
        "num_turns": num_turns,
        "schedule_points": schedule_points,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]


def load_cached_schedule(cache_dir: Path) -> tuple[pd.DataFrame, ScheduleStatus] | None:
    schedule_path = cache_dir / "knob_schedule.parquet"
    status_path = cache_dir / "schedule_status.json"
    if not schedule_path.exists() or not status_path.exists():
        return None
    schedule = pd.read_parquet(schedule_path)
    with status_path.open("r", encoding="utf-8") as fh:
        status = ScheduleStatus(**json.load(fh))
    return schedule, status


def save_cached_schedule(
    cache_dir: Path,
    *,
    schedule: pd.DataFrame,
    schedule_status: ScheduleStatus,
    cache_metadata: dict[str, object],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_frame_outputs(cache_dir / "knob_schedule", schedule)
    with (cache_dir / "schedule_status.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(schedule_status), fh, indent=2)
    with (cache_dir / "cache_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(cache_metadata, fh, indent=2)


def configure_base_line(
    *,
    line_path: str,
    qx: float,
    qy: float,
    xi_x: float,
    xi_y: float,
    sextupoles_mode: str,
    error_variant_name: str,
) -> xt.Line:
    line = xt.load(line_path)
    env = line.env

    cavity_elements, cavity_names = line.get_elements_of_type(xt.Cavity)
    for name in cavity_names:
        line[name].frequency = 200e6
        line[name].lag = 180
        line[name].voltage = 0
    line["actcse.31632"].voltage = 3.0e6

    apply_error_configuration(line, error_variant_name=error_variant_name)

    if sextupoles_mode == "off":
        zero_sextupoles(line)
        line.match(
            method="6d",
            vary=[xt.VaryList(["kqf0", "kqd0"], step=1e-8, tag="quad")],
            targets=[xt.TargetSet(qx=qx, qy=qy, tol=1e-6, tag="tune")],
        )
    else:
        env.vars["qph_setvalue"] = xi_x
        env.vars["qpv_setvalue"] = xi_y
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
    return line


def get_line_start_dispersion(line: xt.Line) -> dict[str, float]:
    tw = line.twiss()
    return {
        "dx": float(np.asarray(tw.dx)[0]),
        "dpx": float(np.asarray(tw.dpx)[0]),
        "dy": float(np.asarray(tw.dy)[0]),
        "dpy": float(np.asarray(tw.dpy)[0]),
    }


def build_target_tune_table(
    *,
    qx0: float,
    qy0: float,
    dq_per_turn_x: float,
    dq_per_turn_y: float,
    num_turns: int,
) -> pd.DataFrame:
    turns = np.arange(num_turns + 1, dtype=float)
    return pd.DataFrame(
        {
            "turn": turns.astype(int),
            "dqx_target": dq_per_turn_x * turns,
            "dqy_target": dq_per_turn_y * turns,
            "qx_target": qx0 + dq_per_turn_x * turns,
            "qy_target": qy0 + dq_per_turn_y * turns,
        }
    )


def build_knob_schedule(
    *,
    line_path: str,
    qx0: float,
    qy0: float,
    xi_x: float,
    xi_y: float,
    sextupoles_mode: str,
    error_variant_name: str,
    target_table: pd.DataFrame,
    schedule_points: int,
) -> tuple[pd.DataFrame, ScheduleStatus]:
    line = configure_base_line(
        line_path=line_path,
        qx=qx0,
        qy=qy0,
        xi_x=xi_x,
        xi_y=xi_y,
        sextupoles_mode=sextupoles_mode,
        error_variant_name=error_variant_name,
    )
    env = line.env

    sample_turns = np.linspace(0, int(target_table["turn"].iloc[-1]), schedule_points)
    qx_target_interp = np.interp(sample_turns, target_table["turn"], target_table["qx_target"])
    qy_target_interp = np.interp(sample_turns, target_table["turn"], target_table["qy_target"])

    rows: list[dict[str, float]] = []
    failure_info: dict[str, float | None] | None = None
    for turn, qx_target, qy_target in zip(sample_turns, qx_target_interp, qy_target_interp):
        try:
            line.match(
                method="6d",
                vary=[xt.VaryList(["kqf0", "kqd0"], step=1e-8, tag="quad")],
                targets=[xt.TargetSet(qx=float(qx_target), qy=float(qy_target), tol=1e-6, tag="tune")],
            )
        except RuntimeError:
            failure_info = {
                "failure_turn": float(turn),
                "failure_qx_target": float(qx_target),
                "failure_qy_target": float(qy_target),
            }
            print(
                "[manual_tune_sweep] Schedule match failed at "
                f"turn={turn:.1f}, qx_target={qx_target:.6f}, qy_target={qy_target:.6f}. "
                "Freezing knobs at the last matched point."
            )
            break
        tw = line.twiss()
        rows.append(
            {
                "turn": float(turn),
                "qx_target": float(qx_target),
                "qy_target": float(qy_target),
                "qx_matched": float(tw.qx),
                "qy_matched": float(tw.qy),
                "dqx_match_error": float(tw.qx - qx_target),
                "dqy_match_error": float(tw.qy - qy_target),
                "kqf0": float(env.vars["kqf0"]._value),
                "kqd0": float(env.vars["kqd0"]._value),
            }
        )
    if not rows:
        raise RuntimeError("Manual tune schedule failed before any valid matched point was found.")

    final_turn = float(target_table["turn"].iloc[-1])
    if rows[-1]["turn"] < final_turn:
        frozen_row = dict(rows[-1])
        frozen_row["turn"] = final_turn
        frozen_row["qx_target"] = float(target_table["qx_target"].iloc[-1])
        frozen_row["qy_target"] = float(target_table["qy_target"].iloc[-1])
        frozen_row["qx_matched"] = float(rows[-1]["qx_matched"])
        frozen_row["qy_matched"] = float(rows[-1]["qy_matched"])
        frozen_row["dqx_match_error"] = float(frozen_row["qx_matched"] - frozen_row["qx_target"])
        frozen_row["dqy_match_error"] = float(frozen_row["qy_matched"] - frozen_row["qy_target"])
        rows.append(frozen_row)

    status = ScheduleStatus(
        completed_full_path=failure_info is None,
        failure_turn=None if failure_info is None else float(failure_info["failure_turn"]),
        failure_qx_target=None if failure_info is None else float(failure_info["failure_qx_target"]),
        failure_qy_target=None if failure_info is None else float(failure_info["failure_qy_target"]),
        matched_points=len(rows),
        final_schedule_turn=float(rows[-1]["turn"]),
        note=(
            "All requested schedule points matched successfully."
            if failure_info is None
            else "Schedule matching failed partway through; knobs are frozen at the last matched point."
        ),
    )
    return pd.DataFrame(rows), status


def build_schedule_interpolators(schedule: pd.DataFrame) -> dict[str, CubicSpline]:
    turns = schedule["turn"].to_numpy(dtype=float)
    if turns.size == 1:
        turns = np.array([turns[0], turns[0] + 1.0], dtype=float)
        kqf0 = np.repeat(schedule["kqf0"].to_numpy(dtype=float), 2)
        kqd0 = np.repeat(schedule["kqd0"].to_numpy(dtype=float), 2)
        return {
            "kqf0": CubicSpline(turns, kqf0),
            "kqd0": CubicSpline(turns, kqd0),
        }
    return {
        "kqf0": CubicSpline(turns, schedule["kqf0"].to_numpy(dtype=float)),
        "kqd0": CubicSpline(turns, schedule["kqd0"].to_numpy(dtype=float)),
    }


def save_snapshot_with_initial_delta(
    snapshot_dir: Path,
    particles,
    turn: int,
    initial_delta_by_id: np.ndarray,
) -> dict[str, np.ndarray]:
    arrays = alive_particle_arrays(particles)
    arrays["initial_delta"] = initial_delta_by_id[arrays["particle_id"]]
    payload = {"turn": np.array(turn, dtype=int), **arrays}
    np.savez_compressed(snapshot_dir / f"snapshot_turn_{turn:05d}.npz", **payload)
    return arrays


def plot_phase_space_initial_delta_overlay(
    phase_dir: Path,
    snapshot_records: list[tuple[int, dict[str, np.ndarray]]],
) -> None:
    valid_records = [(turn, arrays) for turn, arrays in snapshot_records if int(arrays["alive_count"]) > 0]
    if not valid_records:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    pairs = [("x", "px"), ("y", "py"), ("zeta", "delta")]
    scatter = None
    for axis, (coord_x, coord_y) in zip(axes, pairs):
        for _, arrays in valid_records:
            scatter = axis.scatter(
                arrays[coord_x],
                arrays[coord_y],
                c=arrays["initial_delta"],
                s=6,
                alpha=0.35,
                cmap="viridis",
            )
        axis.set_xlabel(coord_x)
        axis.set_ylabel(coord_y)
        axis.grid(True, alpha=0.2)
    if scatter is not None:
        fig.colorbar(scatter, ax=axes, pad=0.02, label="initial delta")
    fig.suptitle("Phase-space overlay coloured by initial delta")
    fig.savefig(phase_dir / "phase_space_initial_delta_overlay.png", dpi=180)
    plt.close(fig)


def lift_fractional_tune_to_reference(
    fractional_tune: np.ndarray,
    reference_tune: np.ndarray,
) -> np.ndarray:
    fractional = np.mod(np.asarray(fractional_tune, dtype=float), 1.0)
    reference = np.asarray(reference_tune, dtype=float)
    integer_part = np.floor(reference)

    candidates = np.stack(
        [
            integer_part - 1.0 + fractional,
            integer_part + fractional,
            integer_part + 1.0 + fractional,
        ],
        axis=1,
    )
    offsets = np.abs(candidates - reference[:, None])
    best_index = np.argmin(offsets, axis=1)
    return candidates[np.arange(reference.shape[0]), best_index]


def build_manual_tune_estimate(
    summary_frame: pd.DataFrame,
    *,
    window_size: int,
    step: int,
    num_harmonics: int,
) -> pd.DataFrame | None:
    outputs: dict[str, pd.DataFrame] = {}
    for plane, coord, coord_px in (
        ("horizontal", "x_beta_mean", "px_beta_mean"),
        ("vertical", "y_beta_mean", "py_beta_mean"),
    ):
        frame = sliding_naff(
            summary_frame[coord].to_numpy(dtype=float),
            summary_frame[coord_px].to_numpy(dtype=float),
            window_size=window_size,
            step=step,
            num_harmonics=num_harmonics,
        )
        if frame is None or frame.empty:
            return None
        outputs[plane] = frame[frame["harmonic"] == 0].copy()
        if outputs[plane].empty:
            return None

    merged = outputs["horizontal"][["window_center", "frequency", "amplitude_abs"]].rename(
        columns={"frequency": "qx_estimate", "amplitude_abs": "qx_amplitude"}
    )
    merged = merged.merge(
        outputs["vertical"][["window_center", "frequency", "amplitude_abs"]].rename(
            columns={"frequency": "qy_estimate", "amplitude_abs": "qy_amplitude"}
        ),
        on="window_center",
        how="inner",
    )

    turn = summary_frame["turn"].to_numpy(dtype=float)
    for column in ("qx_target", "qy_target", "dqx_target", "dqy_target"):
        merged[column] = np.interp(
            merged["window_center"].to_numpy(dtype=float),
            turn,
            summary_frame[column].to_numpy(dtype=float),
        )
    merged["qx_estimate_fractional"] = np.mod(merged["qx_estimate"], 1.0)
    merged["qy_estimate_fractional"] = np.mod(merged["qy_estimate"], 1.0)
    merged["qx_estimate_abs"] = np.abs(merged["qx_estimate"])
    merged["qy_estimate_abs"] = np.abs(merged["qy_estimate"])
    merged["qx_estimate_full"] = lift_fractional_tune_to_reference(
        merged["qx_estimate_fractional"].to_numpy(dtype=float),
        merged["qx_target"].to_numpy(dtype=float),
    )
    merged["qy_estimate_full"] = lift_fractional_tune_to_reference(
        merged["qy_estimate_fractional"].to_numpy(dtype=float),
        merged["qy_target"].to_numpy(dtype=float),
    )
    merged["qx_estimate_abs_full"] = lift_fractional_tune_to_reference(
        np.mod(merged["qx_estimate_abs"].to_numpy(dtype=float), 1.0),
        merged["qx_target"].to_numpy(dtype=float),
    )
    merged["qy_estimate_abs_full"] = lift_fractional_tune_to_reference(
        np.mod(merged["qy_estimate_abs"].to_numpy(dtype=float), 1.0),
        merged["qy_target"].to_numpy(dtype=float),
    )
    merged["qx_residual"] = merged["qx_estimate_full"] - merged["qx_target"]
    merged["qy_residual"] = merged["qy_estimate_full"] - merged["qy_target"]
    merged["qx_residual_abs"] = merged["qx_estimate_abs_full"] - merged["qx_target"]
    merged["qy_residual_abs"] = merged["qy_estimate_abs_full"] - merged["qy_target"]
    return merged


def plot_tune_estimate_vs_target(
    estimate_frame: pd.DataFrame,
    output_path: Path,
    *,
    absolute: bool,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    qx_key = "qx_estimate_abs_full" if absolute else "qx_estimate_full"
    qy_key = "qy_estimate_abs_full" if absolute else "qy_estimate_full"
    label = "|NAFF| lifted" if absolute else "NAFF lifted"

    axes[0].plot(estimate_frame["window_center"], estimate_frame["qx_target"], linewidth=1.8, label="target")
    axes[0].plot(estimate_frame["window_center"], estimate_frame[qx_key], marker="o", markersize=3, linewidth=1.2, label=label)
    axes[0].set_ylabel(r"$Q_x$")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(estimate_frame["window_center"], estimate_frame["qy_target"], linewidth=1.8, label="target")
    axes[1].plot(estimate_frame["window_center"], estimate_frame[qy_key], marker="o", markersize=3, linewidth=1.2, label=label)
    axes[1].set_ylabel(r"$Q_y$")
    axes[1].set_xlabel("Turn")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    fig.suptitle("Windowed NAFF tune estimate compared with target tune path")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_naff_tune_diagram_manual(
    estimate_frame: pd.DataFrame,
    target_table: pd.DataFrame,
    output_path: Path,
    *,
    absolute: bool,
) -> None:
    if estimate_frame is None or estimate_frame.empty:
        return

    qx_key = "qx_estimate_abs_full" if absolute else "qx_estimate_full"
    qy_key = "qy_estimate_abs_full" if absolute else "qy_estimate_full"
    label = "|NAFF| estimate (lifted)" if absolute else "NAFF estimate (lifted)"

    qx0 = float(target_table["qx_target"].iloc[0])
    qy0 = float(target_table["qy_target"].iloc[0])
    td = TuneDiagram(qx0=qx0, qy0=qy0, half_range=0.4, max_order=3, skew=True)
    fig, ax = td.plot(figsize=(8, 7), show_working_point=True)
    ax.plot(
        target_table["qx_target"].to_numpy(dtype=float),
        target_table["qy_target"].to_numpy(dtype=float),
        color="tab:blue",
        linewidth=2.0,
        label="Target path",
    )
    scatter = ax.scatter(
        estimate_frame[qx_key].to_numpy(dtype=float),
        estimate_frame[qy_key].to_numpy(dtype=float),
        c=estimate_frame["window_center"].to_numpy(dtype=float),
        cmap="plasma",
        s=20,
        alpha=0.8,
        zorder=6,
        label=label,
    )
    td.finalize(ax, extra_handles=None)
    ax.legend(loc="best", frameon=True)
    fig.colorbar(scatter, ax=ax, pad=0.02, label="Turn")
    fig.suptitle("NAFF tune estimates on the manual target tune path")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def compute_loss_curve_from_particles(
    particles,
    *,
    target_table: pd.DataFrame,
    num_particles: int,
) -> pd.DataFrame:
    state = np.asarray(particles.state)
    at_turn = np.asarray(particles.at_turn).astype(int, copy=False)
    dead_turns = at_turn[state <= 0]

    turn_axis = target_table["turn"].to_numpy(dtype=int)
    lost_count = np.zeros_like(turn_axis)
    if dead_turns.size > 0:
        clipped_turns = np.clip(dead_turns, 0, turn_axis[-1])
        unique_turns, counts = np.unique(clipped_turns, return_counts=True)
        lost_count[unique_turns] = counts

    surviving_fraction = 1.0 - np.cumsum(lost_count) / float(num_particles)
    frame = target_table.copy()
    frame["lost_count"] = lost_count.astype(int)
    frame["surviving_fraction"] = surviving_fraction
    return frame


def plot_intensity_vs_target(
    loss_curve_frame: pd.DataFrame,
    output_path: Path,
    *,
    x_key: str,
    x_label: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(loss_curve_frame[x_key].to_numpy(dtype=float), loss_curve_frame["surviving_fraction"].to_numpy(dtype=float), color="tab:blue")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Normalised intensity")
    ax.grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def export_dead_particles_manual(
    outdir: Path,
    particles,
    *,
    target_table: pd.DataFrame,
) -> pd.DataFrame:
    state = np.asarray(particles.state)
    dead_mask = state <= 0
    dead_particle_id = np.flatnonzero(dead_mask)
    dead_turn = np.asarray(particles.at_turn)[dead_mask].astype(int, copy=False)
    dead_delta = np.asarray(particles.delta)[dead_mask].astype(float, copy=False)

    turn_clipped = np.clip(dead_turn, 0, int(target_table["turn"].iloc[-1]))
    qx_target = target_table["qx_target"].to_numpy(dtype=float)[turn_clipped]
    qy_target = target_table["qy_target"].to_numpy(dtype=float)[turn_clipped]
    dqx_target = target_table["dqx_target"].to_numpy(dtype=float)[turn_clipped]
    dqy_target = target_table["dqy_target"].to_numpy(dtype=float)[turn_clipped]

    dead_frame = pd.DataFrame(
        {
            "particle_id": dead_particle_id,
            "at_turn": dead_turn,
            "delta": dead_delta,
            "qx_target": qx_target,
            "qy_target": qy_target,
            "dqx_target": dqx_target,
            "dqy_target": dqy_target,
        }
    )
    write_frame_outputs(outdir / "dead_particles", dead_frame)

    if not dead_frame.empty:
        columns_to_plot = [
            ("delta", r"Dead-particle $\delta$"),
            ("qx_target", r"Target $Q_x$ at death"),
            ("qy_target", r"Target $Q_y$ at death"),
        ]
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

        qx0 = float(target_table["qx_target"].iloc[0])
        qy0 = float(target_table["qy_target"].iloc[0])
        td = TuneDiagram(qx0=qx0, qy0=qy0, half_range=0.4, max_order=3, skew=True)
        fig, ax = td.plot(figsize=(8, 7), show_working_point=True)
        ax.plot(target_table["qx_target"], target_table["qy_target"], color="tab:blue", linewidth=2.0, label="Target path")
        scatter = ax.scatter(
            dead_frame["qx_target"].to_numpy(dtype=float),
            dead_frame["qy_target"].to_numpy(dtype=float),
            c=dead_frame["delta"].to_numpy(dtype=float),
            cmap="viridis",
            s=18,
            alpha=0.75,
            zorder=6,
            label="Dead particles",
        )
        td.finalize(ax, extra_handles=None)
        ax.legend(loc="best", frameon=True)
        fig.colorbar(scatter, ax=ax, pad=0.02, label=r"Dead-particle $\delta$")
        fig.suptitle("Dead particles on the manual target tune path")
        fig.savefig(outdir / "dead_particle_tune_diagram.png", dpi=180)
        plt.close(fig)

    return dead_frame


def main() -> None:
    args = parse_args()
    batch_dir = HERE / args.output_base / args.batch_name
    outdir = batch_dir
    if outdir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = HERE / args.output_base / f"{args.batch_name}_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=False)

    schedule_cache_key = build_schedule_cache_key(
        line_path=args.line_path,
        qx=args.qx,
        qy=args.qy,
        xi_x=args.xi_x,
        xi_y=args.xi_y,
        sextupoles_mode=args.sextupoles_mode,
        error_variant=args.error_variant,
        dq_per_turn_x=args.dq_per_turn_x,
        dq_per_turn_y=args.dq_per_turn_y,
        num_turns=args.num_turns,
        schedule_points=args.schedule_points,
    )
    schedule_cache_dir = HERE / "schedule_cache" / schedule_cache_key

    run_config = RunConfig(
        line_path=args.line_path,
        qx=args.qx,
        qy=args.qy,
        xi_x=args.xi_x,
        xi_y=args.xi_y,
        sextupoles_mode=args.sextupoles_mode,
        error_variant=args.error_variant,
        dq_per_turn_x=args.dq_per_turn_x,
        dq_per_turn_y=args.dq_per_turn_y,
        num_turns=args.num_turns,
        num_particles=args.num_particles,
        nemitt_x=args.nemitt_x,
        nemitt_y=args.nemitt_y,
        sigma_z=args.sigma_z,
        snapshot_every=args.snapshot_every,
        progress_every=args.progress_every,
        schedule_points=args.schedule_points,
        naff_harmonics=args.naff_harmonics,
        fft_window=args.fft_window,
        fft_step=args.fft_step,
        omp_threads=args.omp_threads,
        output_dir=str(outdir),
        schedule_cache_key=schedule_cache_key,
        schedule_cache_dir=str(schedule_cache_dir),
        note=args.note,
    )
    with (outdir / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(run_config), fh, indent=2)

    target_table = build_target_tune_table(
        qx0=args.qx,
        qy0=args.qy,
        dq_per_turn_x=args.dq_per_turn_x,
        dq_per_turn_y=args.dq_per_turn_y,
        num_turns=args.num_turns,
    )
    write_frame_outputs(outdir / "target_tune_path", target_table)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    axes[0].plot(target_table["turn"], target_table["qx_target"], linewidth=1.5)
    axes[0].set_ylabel(r"$Q_x$")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(target_table["turn"], target_table["qy_target"], linewidth=1.5, color="tab:orange")
    axes[1].set_ylabel(r"$Q_y$")
    axes[1].set_xlabel("Turn")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle("Requested manual tune path")
    fig.savefig(outdir / "target_tune_path.png", dpi=180)
    plt.close(fig)

    schedule_cache_info: dict[str, object]
    cached_schedule = None if args.force_rebuild_schedule else load_cached_schedule(schedule_cache_dir)
    if cached_schedule is not None:
        print(f"[manual_tune_sweep] Reusing cached kqf0/kqd0 schedule: {schedule_cache_key}")
        schedule, schedule_status = cached_schedule
        schedule_cache_info = {
            "cache_key": schedule_cache_key,
            "cache_dir": str(schedule_cache_dir),
            "cache_hit": True,
            "force_rebuild_schedule": bool(args.force_rebuild_schedule),
        }
    else:
        print("[manual_tune_sweep] Solving kqf0/kqd0 schedule")
        schedule, schedule_status = build_knob_schedule(
            line_path=args.line_path,
            qx0=args.qx,
            qy0=args.qy,
            xi_x=args.xi_x,
            xi_y=args.xi_y,
            sextupoles_mode=args.sextupoles_mode,
            error_variant_name=args.error_variant,
            target_table=target_table,
            schedule_points=args.schedule_points,
        )
        cache_metadata = {
            "cache_key": schedule_cache_key,
            "created_at": datetime.now().isoformat(),
            "line_path": str(Path(args.line_path).expanduser().resolve()),
            "qx": args.qx,
            "qy": args.qy,
            "xi_x": args.xi_x,
            "xi_y": args.xi_y,
            "sextupoles_mode": args.sextupoles_mode,
            "error_variant": args.error_variant,
            "dq_per_turn_x": args.dq_per_turn_x,
            "dq_per_turn_y": args.dq_per_turn_y,
            "num_turns": args.num_turns,
            "schedule_points": args.schedule_points,
        }
        save_cached_schedule(
            schedule_cache_dir,
            schedule=schedule,
            schedule_status=schedule_status,
            cache_metadata=cache_metadata,
        )
        schedule_cache_info = {
            "cache_key": schedule_cache_key,
            "cache_dir": str(schedule_cache_dir),
            "cache_hit": False,
            "force_rebuild_schedule": bool(args.force_rebuild_schedule),
        }

    write_frame_outputs(outdir / "knob_schedule", schedule)
    with (outdir / "schedule_status.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(schedule_status), fh, indent=2)
    with (outdir / "schedule_cache_info.json").open("w", encoding="utf-8") as fh:
        json.dump(schedule_cache_info, fh, indent=2)
    if not schedule_status.completed_full_path:
        print(
            "[manual_tune_sweep] Warning: requested tune path is only partially reachable. "
            f"Schedule froze after turn={schedule_status.failure_turn:.1f}."
        )
    schedule_interpolators = build_schedule_interpolators(schedule)

    line = configure_base_line(
        line_path=args.line_path,
        qx=args.qx,
        qy=args.qy,
        xi_x=args.xi_x,
        xi_y=args.xi_y,
        sextupoles_mode=args.sextupoles_mode,
        error_variant_name=args.error_variant,
    )
    line.discard_tracker()
    if args.omp_threads in {"auto", "openmp"}:
        context = xo.ContextCpu(omp_num_threads=0)
    else:
        context = xo.ContextCpu(omp_num_threads=int(args.omp_threads))
    line.build_tracker(_context=context)
    observation_dispersion = get_line_start_dispersion(line)

    particles = xp.generate_matched_gaussian_bunch(
        nemitt_x=args.nemitt_x,
        nemitt_y=args.nemitt_y,
        sigma_z=args.sigma_z,
        num_particles=args.num_particles,
        line=line,
    )
    initial_delta = np.asarray(particles.delta).astype(float, copy=True)

    snapshot_dir = outdir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    violin_dir = outdir / "violin_plots"
    violin_dir.mkdir(exist_ok=True)
    phase_dir = outdir / "phase_space_plots"
    phase_dir.mkdir(exist_ok=True)

    summary_rows: list[dict[str, float | int]] = []
    snapshot_records: list[tuple[int, dict[str, np.ndarray]]] = []

    print(
        "[manual_tune_sweep] Starting tracking: "
        f"particles={args.num_particles}, turns={args.num_turns}, "
        f"dq_per_turn_x={args.dq_per_turn_x}, dq_per_turn_y={args.dq_per_turn_y}"
    )
    start_time = time.time()
    if hasattr(line, "scattering"):
        line.scattering.enable()

    for turn in range(args.num_turns + 1):
        line.vars["kqf0"] = float(schedule_interpolators["kqf0"](turn))
        line.vars["kqd0"] = float(schedule_interpolators["kqd0"](turn))

        row = collect_turn_row(particles, turn=turn, sweep_per_turn_hz=0.0)
        row["qx_target"] = float(target_table["qx_target"].iloc[turn])
        row["qy_target"] = float(target_table["qy_target"].iloc[turn])
        row["dqx_target"] = float(target_table["dqx_target"].iloc[turn])
        row["dqy_target"] = float(target_table["dqy_target"].iloc[turn])
        row["kqf0"] = float(line.vars["kqf0"]._value)
        row["kqd0"] = float(line.vars["kqd0"]._value)
        summary_rows.append(row)

        if turn % args.snapshot_every == 0 or turn == args.num_turns:
            arrays = save_snapshot_with_initial_delta(snapshot_dir, particles, turn, initial_delta)
            snapshot_records.append((turn, arrays))

        if turn == args.num_turns:
            break

        line.track(particles=particles, num_turns=1)
        if (turn + 1) % max(1, args.progress_every) == 0 or (turn + 1) == args.num_turns:
            print(
                "[manual_tune_sweep] Progress: "
                f"{turn + 1}/{args.num_turns} turns in {time.time() - start_time:.1f}s"
            )

    if hasattr(line, "scattering"):
        line.scattering.disable()
    print(f"[manual_tune_sweep] Tracking finished in {time.time() - start_time:.1f}s")

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame = add_dispersion_subtracted_columns(summary_frame, observation_dispersion)
    write_frame_outputs(outdir / "turn_summary", summary_frame)

    plot_moment_family(summary_frame, outdir / "mean_evolution.png", "mean")
    plot_moment_family(summary_frame, outdir / "std_evolution.png", "std")

    plot_violin_evolution(violin_dir, snapshot_records)
    plot_violin_evolution_beta(violin_dir, snapshot_records, observation_dispersion)
    plot_phase_space_evolution(phase_dir, snapshot_records)
    plot_phase_space_evolution_beta(phase_dir, snapshot_records, observation_dispersion)
    plot_phase_space_turn_colored(phase_dir, snapshot_records)
    plot_phase_space_turn_colored_beta(phase_dir, snapshot_records, observation_dispersion)
    plot_phase_space_initial_delta_overlay(phase_dir, snapshot_records)

    summary = {column: summary_frame[column].to_numpy() for column in summary_frame.columns}
    beta_summary = {
        "x_mean": summary_frame["x_beta_mean"].to_numpy(),
        "px_mean": summary_frame["px_beta_mean"].to_numpy(),
        "y_mean": summary_frame["y_beta_mean"].to_numpy(),
        "py_mean": summary_frame["py_beta_mean"].to_numpy(),
    }
    plot_spectrogram(summary, outdir / "centroid_spectrogram.png", window_size=args.fft_window, step=args.fft_step)
    plot_spectrogram(beta_summary, outdir / "centroid_spectrogram_beta.png", window_size=args.fft_window, step=args.fft_step)

    save_sliding_naff(summary, outdir / "sliding_naff_global", window_size=args.fft_window, step=args.fft_step, num_harmonics=args.naff_harmonics)
    save_sliding_naff(beta_summary, outdir / "sliding_naff_beta", window_size=args.fft_window, step=args.fft_step, num_harmonics=args.naff_harmonics)
    plot_naff_tracks(summary, outdir / "sliding_naff_global.png", window_size=args.fft_window, step=args.fft_step, num_harmonics=args.naff_harmonics)
    plot_naff_tracks(beta_summary, outdir / "sliding_naff_beta.png", window_size=args.fft_window, step=args.fft_step, num_harmonics=args.naff_harmonics)

    tune_estimate = build_manual_tune_estimate(
        summary_frame,
        window_size=args.fft_window,
        step=args.fft_step,
        num_harmonics=args.naff_harmonics,
    )
    if tune_estimate is not None:
        write_frame_outputs(outdir / "tune_estimate", tune_estimate)
        plot_tune_estimate_vs_target(tune_estimate, outdir / "tune_estimate_vs_turn.png", absolute=False)
        plot_tune_estimate_vs_target(tune_estimate, outdir / "tune_estimate_abs_vs_turn.png", absolute=True)
        plot_naff_tune_diagram_manual(tune_estimate, target_table, outdir / "naff_tune_diagram.png", absolute=False)
        plot_naff_tune_diagram_manual(tune_estimate, target_table, outdir / "naff_abs_tune_diagram.png", absolute=True)

    tune_estimate_xonly_h3 = build_tune_estimate_position_only(
        summary_frame.assign(delta_from_sweep=summary_frame["dqx_target"]),
        signal_columns={"horizontal": "x_beta_mean", "vertical": "y_beta_mean"},
        window_size=args.fft_window,
        step=args.fft_step,
        num_harmonics=3,
    )
    if tune_estimate_xonly_h3 is not None:
        write_frame_outputs(outdir / "tune_estimate_xonly_h3", tune_estimate_xonly_h3)

    loss_curve_frame = compute_loss_curve_from_particles(
        particles,
        target_table=target_table,
        num_particles=args.num_particles,
    )
    write_frame_outputs(outdir / "intensity_loss", loss_curve_frame)
    plot_intensity_vs_target(
        loss_curve_frame,
        outdir / "intensity_loss_vs_turn.png",
        x_key="turn",
        x_label="Turn",
        title="Intensity loss vs turn",
    )
    plot_intensity_vs_target(
        loss_curve_frame,
        outdir / "intensity_loss_vs_qx.png",
        x_key="qx_target",
        x_label=r"$Q_x$ target",
        title="Intensity loss vs target $Q_x$",
    )
    plot_intensity_vs_target(
        loss_curve_frame,
        outdir / "intensity_loss_vs_qy.png",
        x_key="qy_target",
        x_label=r"$Q_y$ target",
        title="Intensity loss vs target $Q_y$",
    )

    export_dead_particles_manual(outdir, particles, target_table=target_table)

    with (outdir / "death_turns.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "at_turn": np.asarray(particles.at_turn).tolist(),
                "final_state": np.asarray(particles.state).tolist(),
                "target_path": target_table.to_dict(orient="list"),
            },
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
