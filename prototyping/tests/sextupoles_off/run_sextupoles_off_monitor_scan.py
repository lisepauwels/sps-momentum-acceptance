from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import xcoll as xc
import xobjects as xo
import xpart as xp
import xtrack as xt


REPO_ROOT = Path(__file__).resolve().parents[2]
RF_SWEEP_DIR = REPO_ROOT / "tests" / "rf_sweep_speed_tests"
HELPER_DIR = REPO_ROOT / "helper_functions"
WORKFLOW_DIR = REPO_ROOT / "tune_scan_workflow"
sys.path.insert(0, str(WORKFLOW_DIR))
sys.path.insert(0, str(HELPER_DIR))
sys.path.insert(0, str(RF_SWEEP_DIR))

from recompute_saved_tune_analysis import build_tune_estimate_position_only
from rf_sweep_speed_scan import (
    COORDINATES,
    DEFAULT_LINE_PATH,
    add_dispersion_subtracted_columns,
    add_sweep_map_tunes_to_estimate,
    build_tune_estimate_from_naff,
    collect_turn_row,
    compute_loss_curve_from_summary,
    df_to_delta,
    ensure_tidp,
    error_variants,
    export_dead_particle_deltas,
    load_tune_map_case,
    particle_stats,
    plot_delta_envelope,
    plot_intensity_loss,
    plot_moment_family,
    plot_naff_abs_tune_diagram,
    plot_naff_harmonics_positive_vs_sweep_map,
    plot_naff_tracks,
    plot_naff_tune_diagram,
    plot_spectrogram,
    plot_tune_estimate,
    plot_tune_estimate_abs_vs_sweep_map,
    plot_tune_estimate_vs_sweep_map,
    repo_convention_signed_sweep,
    save_sliding_naff,
    write_frame_outputs,
)
from tune_diagram import SweepTrajectory, TuneMap


DEFAULT_PLANES = ("DPpos", "DPneg")
DEFAULT_OUTPUT_BASE = "sextupoles_off_monitor_outputs"
DEFAULT_MONITOR_ELEMENT = "qd.31110"
SHARED_TUNE_MAP_PATH = Path(__file__).resolve().parent / "tune_map_sextupoles_off_shared.npz"
SHARED_TUNE_MAP_FIGURE_PATH = Path(__file__).resolve().parent / "tune_map_sextupoles_off_shared.png"


@dataclass
class CaseConfig:
    line_path: str
    qx: float
    qy: float
    xi_x: float
    xi_y: float
    error_variant: str
    tune_map_case: str
    monitor_element: str
    plane: str
    total_sweep_hz: float
    sweep_per_turn_hz: float
    num_turns: int
    num_particles: int
    nemitt_x: float
    nemitt_y: float
    sigma_z: float
    naff_harmonics: int
    fft_window: int
    fft_step: int
    omp_threads: str
    sextupoles_off: bool
    output_dir: str
    tune_map_path: str | None
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run DPpos/DPneg momentum-acceptance sweeps with sextupoles off and "
            "a real ParticlesMonitor at a chosen element."
        )
    )
    parser.add_argument("--line-path", default=DEFAULT_LINE_PATH)
    parser.add_argument("--qx", type=float, default=20.13)
    parser.add_argument("--qy", type=float, default=20.18)
    parser.add_argument("--xi-x", type=float, default=0.5)
    parser.add_argument("--xi-y", type=float, default=0.5)
    parser.add_argument("--error-variant", choices=sorted(error_variants), default="none")
    parser.add_argument("--tune-map-case", choices=["WithErrors", "WithoutErrors", "Simplified"], default=None)
    parser.add_argument("--planes", nargs="+", choices=DEFAULT_PLANES, default=list(DEFAULT_PLANES))
    parser.add_argument("--monitor-element", default=DEFAULT_MONITOR_ELEMENT)
    parser.add_argument(
        "--observation-mode",
        choices=["monitor", "rf_style"],
        default="monitor",
        help=(
            "monitor uses a real ParticlesMonitor at the requested element; "
            "rf_style cycles the line to that element and collects turn summaries "
            "with a one-turn loop like rf_sweep_speed_tests."
        ),
    )
    parser.add_argument(
        "--no-cycle",
        action="store_true",
        help="With --observation-mode rf_style, keep the original line start instead of cycling to the monitor element.",
    )
    parser.add_argument("--total-sweep-hz", type=float, default=3000.0)
    parser.add_argument("--num-turns", type=int, default=6000)
    parser.add_argument("--num-particles", type=int, default=100)
    parser.add_argument("--nemitt-x", type=float, default=2e-6)
    parser.add_argument("--nemitt-y", type=float, default=2e-6)
    parser.add_argument("--sigma-z", type=float, default=0.224)
    parser.add_argument("--naff-harmonics", type=int, default=5)
    parser.add_argument("--fft-window", type=int, default=256)
    parser.add_argument("--fft-step", type=int, default=64)
    parser.add_argument("--omp-threads", default="0")
    parser.add_argument("--output-base", default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--batch-name", default="sextupoles_off_monitor")
    parser.add_argument(
        "--reuse-tune-map",
        action="store_true",
        help="Reuse the shared sextupoles-off tune map if it already exists.",
    )
    parser.add_argument(
        "--force-rebuild-tune-map",
        action="store_true",
        help="Rebuild the shared sextupoles-off tune map even if one already exists.",
    )
    parser.add_argument(
        "--track-only",
        action="store_true",
        help="Skip NAFF / tune-diagram / plot-heavy post-processing and only save core tracking tables.",
    )
    parser.add_argument(
        "--loss-only",
        action="store_true",
        help="Do a straight tracking run and save only survival/loss outputs, skipping all turn-by-turn summary collection.",
    )
    parser.add_argument(
        "--note",
        default=(
            "Sextupoles are explicitly zeroed before matching, tunes are rematched "
            "with quadrupoles only, and turn-by-turn data come from a ParticlesMonitor "
            "inserted upstream of the requested element."
        ),
    )
    return parser.parse_args()


def build_case_dir(batch_dir: Path, plane: str) -> Path:
    candidate = batch_dir / plane
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = batch_dir / f"{plane}_{stamp}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def apply_error_configuration(line: xt.Line, error_variant_name: str) -> None:
    env = line.env
    b1, b2, b3, b4, b5, b6 = error_variants[error_variant_name]
    tte = env.elements.get_table()
    mask_rbends = tte.element_type == "RBend"
    mask_quads = tte.element_type == "Quadrupole"

    mba = tte.rows[mask_rbends].rows["mba.*"].name
    mbb = tte.rows[mask_rbends].rows["mbb.*"].name
    qf = tte.rows[mask_quads].rows["qf.*"].name
    qd = tte.rows[mask_quads].rows["qd.*"].name

    for nn in mba:
        env[nn].knl = np.array([b1 * 0.0, b2 * 0.0, b3 * 2.12e-3, b4 * 0.0, b5 * -5.74, b6 * 0.0])

    for nn in mbb:
        env[nn].knl = np.array([b1 * 0.0, b2 * 0.0, b3 * -3.19e-3, b4 * 0.0, b5 * -5.10, b6 * 0.0])

    for nn in qf:
        env[nn].knl = np.array([b1 * 0.0, b2 * 0.0, b3 * 0.0, b4 * 0.75e-1, b5 * 0.0, b6 * -0.87e3])

    for nn in qd:
        env[nn].knl = np.array([b1 * 0.0, b2 * 0.0, b3 * 0.0, b4 * -2.03e-1, b5 * 0.0, b6 * 2.04e3])


def zero_sextupoles(line: xt.Line) -> None:
    tt_vars = line.vars.get_table()
    for kk in tt_vars.rows["kls.*"].name:
        line[kk] = 0.0

    for kk in tt_vars.rows["kl.*"].name:
        line[kk] = 0.0

    for kk in tt_vars.rows["ks.*"].name:
        line[kk] = 0.0

    tw = line.twiss()
    print(
        "[sextupoles_off] zero_sextupoles optics: "
        f"qx={tw.qx:.6f}, qy={tw.qy:.6f}, dqx={tw.dqx:.6f}, dqy={tw.dqy:.6f}"
    )


def configure_line_for_observation(
    *,
    line_path: str,
    qx: float,
    qy: float,
    omp_threads: str,
    monitor_element: str,
    monitor_name: str,
    num_turns: int,
    num_particles: int,
    error_variant_name: str,
    observation_mode: str,
    no_cycle: bool,
) -> tuple[xt.Line, str | None]:
    print(f"[sextupoles_off] Loading lattice from {line_path}")
    line = xt.load(line_path)

    cavity_elements, cavity_names = line.get_elements_of_type(xt.Cavity)
    for name in cavity_names:
        line[name].frequency = 200e6
        line[name].lag = 180
        line[name].voltage = 0
    line["actcse.31632"].voltage = 3.0e6

    apply_error_configuration(line, error_variant_name=error_variant_name)
    zero_sextupoles(line)

    print("[sextupoles_off] Matching tunes with quadrupoles only")
    line.match(
        method="6d",
        vary=[xt.VaryList(["kqf0", "kqd0"], step=1e-8, tag="quad")],
        targets=[xt.TargetSet(qx=qx, qy=qy, tol=1e-6, tag="tune")],
    )

    ensure_tidp(line)
    line.discard_tracker()
    inserted_monitor_name: str | None = None
    if observation_mode == "monitor":
        monitor_at_s = float(line.get_s_position(at_elements=monitor_element, mode="upstream"))
        line.insert_element(
            name=monitor_name,
            element=xt.ParticlesMonitor(
                start_at_turn=0,
                stop_at_turn=num_turns + 1,
                num_particles=num_particles,
            ),
            at_s=monitor_at_s,
        )
        inserted_monitor_name = monitor_name
    else:
        if not no_cycle:
            line.cycle(name_first_element=monitor_element, inplace=True)

    if omp_threads in {"auto", "openmp"}:
        context = xo.ContextCpu(omp_num_threads=0)
    else:
        context = xo.ContextCpu(omp_num_threads=int(omp_threads))

    print(f"[sextupoles_off] Building tracker for observation at {monitor_element} ({observation_mode})")
    line.build_tracker(_context=context)
    return line, inserted_monitor_name


def configure_line_basic(
    *,
    line_path: str,
    qx: float,
    qy: float,
    omp_threads: str,
    error_variant_name: str,
) -> xt.Line:
    print(f"[sextupoles_off] Loading lattice from {line_path}")
    line = xt.load(line_path)

    cavity_elements, cavity_names = line.get_elements_of_type(xt.Cavity)
    for name in cavity_names:
        line[name].frequency = 200e6
        line[name].lag = 180
        line[name].voltage = 0
    line["actcse.31632"].voltage = 3.0e6

    apply_error_configuration(line, error_variant_name=error_variant_name)
    zero_sextupoles(line)

    print("[sextupoles_off] Matching tunes with quadrupoles only")
    line.match(
        method="6d",
        vary=[xt.VaryList(["kqf0", "kqd0"], step=1e-8, tag="quad")],
        targets=[xt.TargetSet(qx=qx, qy=qy, tol=1e-6, tag="tune")],
    )

    ensure_tidp(line)
    line.discard_tracker()
    if omp_threads in {"auto", "openmp"}:
        context = xo.ContextCpu(omp_num_threads=0)
    else:
        context = xo.ContextCpu(omp_num_threads=int(omp_threads))

    print("[sextupoles_off] Building tracker for plain loss-only tracking")
    line.build_tracker(_context=context)
    return line


def build_tune_map_for_scenario(
    *,
    args: argparse.Namespace,
    batch_dir: Path,
) -> tuple[TuneMap | None, Path | None]:
    tune_map_path = SHARED_TUNE_MAP_PATH

    if not tune_map_path.exists() and not args.force_rebuild_tune_map:
        candidate_maps = sorted(
            (Path(__file__).resolve().parent / args.output_base).glob("*/tune_map_sextupoles_off.npz"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidate_maps:
            print(f"[sextupoles_off] Promoting existing tune map to shared cache: {candidate_maps[0]}")
            shutil.copy2(candidate_maps[0], tune_map_path)
            candidate_png = candidate_maps[0].with_suffix(".png")
            if candidate_png.exists():
                shutil.copy2(candidate_png, SHARED_TUNE_MAP_FIGURE_PATH)

    if tune_map_path.exists() and (args.reuse_tune_map or not args.force_rebuild_tune_map):
        print(f"[sextupoles_off] Reusing shared tune map: {tune_map_path}")
        return TuneMap.load(str(tune_map_path)), tune_map_path

    delta_extent = abs(df_to_delta(args.total_sweep_hz))
    delta_range = (-delta_extent, delta_extent)
    print(
        "[sextupoles_off] Building local tune map for sextupoles-off scenario "
        f"over delta range {delta_range}"
    )

    try:
        line = xt.load(args.line_path)
        cavity_elements, cavity_names = line.get_elements_of_type(xt.Cavity)
        for name in cavity_names:
            line[name].frequency = 200e6
            line[name].lag = 180
            line[name].voltage = 0
        line["actcse.31632"].voltage = 3.0e6
        apply_error_configuration(line, error_variant_name=args.error_variant)
        zero_sextupoles(line)
        line.match(
            method="6d",
            vary=[xt.VaryList(["kqf0", "kqd0"], step=1e-8, tag="quad")],
            targets=[xt.TargetSet(qx=args.qx, qy=args.qy, tol=1e-6, tag="tune")],
        )
        sweep = SweepTrajectory.from_twiss_scan(
            line,
            delta_range=delta_range,
            n_points=301,
            verbose=False,
        )
        tune_map = sweep.build_map()
    except Exception as exc:
        print(f"[sextupoles_off] Failed to build local tune map: {exc}")
        return None, None

    tune_map.save(str(tune_map_path))

    fig, _ = tune_map.plot_map()
    fig.suptitle("Sextupoles-off tune map")
    fig.savefig(SHARED_TUNE_MAP_FIGURE_PATH, dpi=180)
    fig.savefig(batch_dir / "tune_map_sextupoles_off.png", dpi=180)
    plt.close(fig)
    return tune_map, tune_map_path


def get_monitor_dispersion(line: xt.Line, monitor_name: str) -> dict[str, float]:
    tw = line.twiss()
    row = tw.rows[monitor_name]
    return {
        "dx": float(np.asarray(row.dx)[0]),
        "dpx": float(np.asarray(row.dpx)[0]),
        "dy": float(np.asarray(row.dy)[0]),
        "dpy": float(np.asarray(row.dpy)[0]),
    }


def get_line_start_dispersion(line: xt.Line) -> dict[str, float]:
    tw = line.twiss()
    return {
        "dx": float(np.asarray(tw.dx)[0]),
        "dpx": float(np.asarray(tw.dpx)[0]),
        "dy": float(np.asarray(tw.dy)[0]),
        "dpy": float(np.asarray(tw.dpy)[0]),
    }


def build_closed_orbit_x_map(
    *,
    line_path: str,
    qx: float,
    qy: float,
    error_variant_name: str,
    element_name: str,
    delta_range: tuple[float, float],
    n_points: int = 301,
) -> tuple[np.ndarray, np.ndarray]:
    line = xt.load(line_path)
    cavity_elements, cavity_names = line.get_elements_of_type(xt.Cavity)
    for name in cavity_names:
        line[name].frequency = 200e6
        line[name].lag = 180
        line[name].voltage = 0
    line["actcse.31632"].voltage = 3.0e6

    apply_error_configuration(line, error_variant_name=error_variant_name)
    zero_sextupoles(line)
    line.match(
        method="6d",
        vary=[xt.VaryList(["kqf0", "kqd0"], step=1e-8, tag="quad")],
        targets=[xt.TargetSet(qx=qx, qy=qy, tol=1e-6, tag="tune")],
    )

    deltas = np.linspace(delta_range[0], delta_range[1], n_points)
    x_values = np.full_like(deltas, np.nan, dtype=float)
    index_zero = int(np.argmin(np.abs(deltas)))

    tw0 = line.twiss4d(delta0=0.0)
    x_values[index_zero] = float(np.asarray(tw0.rows[element_name].x)[0])

    co_prev = tw0.particle_on_co
    for ii in range(index_zero + 1, len(deltas)):
        delta = float(deltas[ii])
        tw = line.twiss4d(delta0=delta, co_guess=co_prev)
        x_values[ii] = float(np.asarray(tw.rows[element_name].x)[0])
        co_prev = tw.particle_on_co

    co_prev = tw0.particle_on_co
    for ii in range(index_zero - 1, -1, -1):
        delta = float(deltas[ii])
        tw = line.twiss4d(delta0=delta, co_guess=co_prev)
        x_values[ii] = float(np.asarray(tw.rows[element_name].x)[0])
        co_prev = tw.particle_on_co

    return deltas, x_values


def monitor_to_frames(
    monitor: xt.ParticlesMonitor,
    *,
    num_turns: int,
    sweep_per_turn_hz: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    turns = np.arange(num_turns + 1, dtype=int)
    state = np.asarray(monitor.state)
    particle_id = np.asarray(monitor.particle_id)
    at_turn = np.asarray(monitor.at_turn)

    summary_rows: list[dict[str, float | int]] = []
    for turn_index, turn in enumerate(turns):
        alive_mask = state[:, turn_index] > 0
        row: dict[str, float | int] = {
            "turn": int(turn),
            "alive_count": int(np.sum(alive_mask)),
            "delta_from_sweep": float(df_to_delta(sweep_per_turn_hz * turn)),
        }
        for coord in COORDINATES:
            values = np.asarray(getattr(monitor, coord))[:, turn_index]
            values = values[alive_mask].astype(float, copy=False)
            mean, std, moment3, skewness = particle_stats(values)
            row[f"{coord}_mean"] = mean
            row[f"{coord}_std"] = std
            row[f"{coord}_moment3"] = moment3
            row[f"{coord}_skewness"] = skewness
            if coord == "delta":
                row["delta_min"] = float(np.min(values)) if values.size else np.nan
                row["delta_max"] = float(np.max(values)) if values.size else np.nan
        summary_rows.append(row)

    long_columns: dict[str, np.ndarray] = {
        "particle_id": particle_id.reshape(-1).astype(int, copy=False),
        "turn": at_turn.reshape(-1).astype(int, copy=False),
        "state": state.reshape(-1).astype(int, copy=False),
    }
    for coord in COORDINATES:
        long_columns[coord] = np.asarray(getattr(monitor, coord)).reshape(-1).astype(float, copy=False)
    long_columns["delta_from_sweep"] = df_to_delta(sweep_per_turn_hz * long_columns["turn"])

    return pd.DataFrame(summary_rows), pd.DataFrame(long_columns)


def compute_loss_curve_from_particles(
    particles,
    *,
    num_particles: int,
    num_turns: int,
    sweep_per_turn_hz: float,
) -> pd.DataFrame:
    state = np.asarray(particles.state)
    at_turn = np.asarray(particles.at_turn).astype(int, copy=False)
    dead_turns = at_turn[state <= 0]

    turn_axis = np.arange(num_turns + 1, dtype=int)
    lost_count = np.zeros_like(turn_axis)
    if dead_turns.size > 0:
        clipped_turns = np.clip(dead_turns, 0, num_turns)
        unique_turns, counts = np.unique(clipped_turns, return_counts=True)
        lost_count[unique_turns] = counts

    surviving_fraction = 1.0 - np.cumsum(lost_count) / float(num_particles)
    return pd.DataFrame(
        {
            "turn": turn_axis,
            "delta": df_to_delta(sweep_per_turn_hz * turn_axis),
            "lost_count": lost_count.astype(int),
            "surviving_fraction": surviving_fraction,
        }
    )


def plot_loss_only_delta_distributions(case_dir: Path, particles) -> None:
    state = np.asarray(particles.state)
    delta = np.asarray(particles.delta).astype(float, copy=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    survivor_delta = delta[state > 0]
    dead_delta = delta[state <= 0]

    if survivor_delta.size:
        axes[0].hist(survivor_delta * 1e3, bins=50, color="tab:blue", alpha=0.8)
    axes[0].set_title("Surviving-particle final delta")
    axes[0].set_xlabel(r"$\delta$ [$10^{-3}$]")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.2)

    if dead_delta.size:
        axes[1].hist(dead_delta * 1e3, bins=50, color="tab:red", alpha=0.8)
    axes[1].set_title("Dead-particle final delta")
    axes[1].set_xlabel(r"$\delta$ [$10^{-3}$]")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.2)

    fig.savefig(case_dir / "delta_distributions.png", dpi=180)
    plt.close(fig)


def save_loss_only_outputs(
    *,
    case_dir: Path,
    particles,
    tune_map,
    num_particles: int,
    num_turns: int,
    sweep_per_turn_hz: float,
) -> None:
    loss_curve_frame = compute_loss_curve_from_particles(
        particles,
        num_particles=num_particles,
        num_turns=num_turns,
        sweep_per_turn_hz=sweep_per_turn_hz,
    )
    write_frame_outputs(case_dir / "intensity_loss", loss_curve_frame)
    plot_intensity_loss(
        {
            "delta": loss_curve_frame["delta"].to_numpy(),
            "surviving_fraction": loss_curve_frame["surviving_fraction"].to_numpy(),
        },
        case_dir / "intensity_loss_vs_delta.png",
    )
    plot_loss_only_delta_distributions(case_dir, particles)
    export_dead_particle_deltas(case_dir, particles, tune_map)
    with (case_dir / "death_turns.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "at_turn": np.asarray(particles.at_turn).tolist(),
                "final_state": np.asarray(particles.state).tolist(),
            },
            fh,
            indent=2,
        )


def save_monitor_outputs(
    *,
    case_dir: Path,
    summary_frame: pd.DataFrame,
    monitor_frame: pd.DataFrame | None,
    tune_map,
    fft_window: int,
    fft_step: int,
    naff_harmonics: int,
    num_particles: int,
    particles,
    observation_dispersion: dict[str, float],
    track_only: bool,
) -> None:
    summary_frame = add_dispersion_subtracted_columns(summary_frame, observation_dispersion)
    write_frame_outputs(case_dir / "turn_summary", summary_frame)
    if monitor_frame is not None:
        write_frame_outputs(case_dir / "monitor_particles", monitor_frame)

    loss_curve_frame = compute_loss_curve_from_summary(summary_frame, num_particles=num_particles)
    write_frame_outputs(case_dir / "intensity_loss", loss_curve_frame)
    export_dead_particle_deltas(case_dir, particles, tune_map)

    with (case_dir / "death_turns.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "observation_dispersion": observation_dispersion,
                "at_turn": np.asarray(particles.at_turn).tolist(),
                "final_state": np.asarray(particles.state).tolist(),
            },
            fh,
            indent=2,
        )

    np.savez_compressed(
        case_dir / "centroid_signals.npz",
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

    if track_only:
        return

    plot_moment_family(summary_frame, case_dir / "mean_evolution.png", "mean")
    plot_moment_family(summary_frame, case_dir / "std_evolution.png", "std")
    plot_delta_envelope(summary_frame, case_dir / "delta_envelope_vs_sweep_delta.png")
    plot_intensity_loss(
        {
            "delta": loss_curve_frame["delta"].to_numpy(),
            "surviving_fraction": loss_curve_frame["surviving_fraction"].to_numpy(),
        },
        case_dir / "intensity_loss_vs_delta.png",
    )

    summary = {column: summary_frame[column].to_numpy() for column in summary_frame.columns}
    beta_summary = {
        "x_mean": summary_frame["x_beta_mean"].to_numpy(),
        "px_mean": summary_frame["px_beta_mean"].to_numpy(),
        "y_mean": summary_frame["y_beta_mean"].to_numpy(),
        "py_mean": summary_frame["py_beta_mean"].to_numpy(),
    }

    plot_spectrogram(summary, case_dir / "centroid_spectrogram.png", window_size=fft_window, step=fft_step)
    plot_spectrogram(beta_summary, case_dir / "centroid_spectrogram_beta.png", window_size=fft_window, step=fft_step)

    save_sliding_naff(
        summary,
        case_dir / "sliding_naff_global",
        window_size=fft_window,
        step=fft_step,
        num_harmonics=naff_harmonics,
    )
    save_sliding_naff(
        beta_summary,
        case_dir / "sliding_naff_beta",
        window_size=fft_window,
        step=fft_step,
        num_harmonics=naff_harmonics,
    )
    plot_naff_tracks(
        summary,
        case_dir / "sliding_naff_global.png",
        window_size=fft_window,
        step=fft_step,
        num_harmonics=naff_harmonics,
    )
    plot_naff_tracks(
        beta_summary,
        case_dir / "sliding_naff_beta.png",
        window_size=fft_window,
        step=fft_step,
        num_harmonics=naff_harmonics,
    )

    horizontal_beta_path = case_dir / "sliding_naff_beta_horizontal.parquet"
    vertical_beta_path = case_dir / "sliding_naff_beta_vertical.parquet"
    if horizontal_beta_path.exists() and vertical_beta_path.exists():
        horizontal_beta = pd.read_parquet(horizontal_beta_path)
        vertical_beta = pd.read_parquet(vertical_beta_path)
        plot_naff_harmonics_positive_vs_sweep_map(
            horizontal_beta,
            vertical_beta,
            summary_frame,
            tune_map,
            case_dir / "naff_harmonics_positive_vs_sweep_map.png",
        )

    xonly_harmonics = 3
    global_position_only = {
        "x_mean": summary_frame["x_mean"].to_numpy(),
        "y_mean": summary_frame["y_mean"].to_numpy(),
    }
    beta_position_only = {
        "x_mean": summary_frame["x_beta_mean"].to_numpy(),
        "y_mean": summary_frame["y_beta_mean"].to_numpy(),
    }
    save_sliding_naff(
        global_position_only,
        case_dir / "sliding_naff_global_xonly_h3",
        window_size=fft_window,
        step=fft_step,
        num_harmonics=xonly_harmonics,
    )
    save_sliding_naff(
        beta_position_only,
        case_dir / "sliding_naff_beta_xonly_h3",
        window_size=fft_window,
        step=fft_step,
        num_harmonics=xonly_harmonics,
    )
    plot_naff_tracks(
        global_position_only,
        case_dir / "sliding_naff_global_xonly_h3.png",
        window_size=fft_window,
        step=fft_step,
        num_harmonics=xonly_harmonics,
    )
    plot_naff_tracks(
        beta_position_only,
        case_dir / "sliding_naff_beta_xonly_h3.png",
        window_size=fft_window,
        step=fft_step,
        num_harmonics=xonly_harmonics,
    )
    horizontal_beta_xonly_path = case_dir / "sliding_naff_beta_xonly_h3_horizontal.parquet"
    vertical_beta_xonly_path = case_dir / "sliding_naff_beta_xonly_h3_vertical.parquet"
    if horizontal_beta_xonly_path.exists() and vertical_beta_xonly_path.exists():
        horizontal_beta_xonly = pd.read_parquet(horizontal_beta_xonly_path)
        vertical_beta_xonly = pd.read_parquet(vertical_beta_xonly_path)
        plot_naff_harmonics_positive_vs_sweep_map(
            horizontal_beta_xonly,
            vertical_beta_xonly,
            summary_frame,
            tune_map,
            case_dir / "naff_harmonics_positive_vs_sweep_map_xonly_h3.png",
        )

    tune_estimate = build_tune_estimate_from_naff(
        summary_frame,
        signal_columns={"horizontal": "x_beta_mean", "vertical": "y_beta_mean"},
        window_size=fft_window,
        step=fft_step,
        num_harmonics=naff_harmonics,
    )
    if tune_estimate is not None:
        tune_estimate = add_sweep_map_tunes_to_estimate(tune_estimate, tune_map)
        write_frame_outputs(case_dir / "tune_estimate", tune_estimate)
        plot_tune_estimate(
            tune_estimate,
            case_dir / "tune_estimate_vs_turn.png",
            x_key="window_center",
            x_label="Turn",
        )
        plot_tune_estimate(
            tune_estimate,
            case_dir / "tune_estimate_vs_delta.png",
            x_key="delta_center",
            x_label=r"$\delta$",
        )
        plot_tune_estimate_vs_sweep_map(tune_estimate, case_dir / "tune_estimate_vs_sweep_map.png")
        plot_tune_estimate_abs_vs_sweep_map(
            tune_estimate,
            case_dir / "tune_estimate_abs_vs_sweep_map.png",
        )
        plot_naff_tune_diagram(tune_estimate, tune_map, case_dir / "naff_tune_diagram.png")
        plot_naff_abs_tune_diagram(tune_estimate, tune_map, case_dir / "naff_abs_tune_diagram.png")

    tune_estimate_xonly_h3 = build_tune_estimate_position_only(
        summary_frame,
        signal_columns={"horizontal": "x_beta_mean", "vertical": "y_beta_mean"},
        window_size=fft_window,
        step=fft_step,
        num_harmonics=xonly_harmonics,
    )
    if tune_estimate_xonly_h3 is not None:
        tune_estimate_xonly_h3 = add_sweep_map_tunes_to_estimate(tune_estimate_xonly_h3, tune_map)
        write_frame_outputs(case_dir / "tune_estimate_xonly_h3", tune_estimate_xonly_h3)
        plot_tune_estimate(
            tune_estimate_xonly_h3,
            case_dir / "tune_estimate_vs_turn_xonly_h3.png",
            x_key="window_center",
            x_label="Turn",
        )
        plot_tune_estimate(
            tune_estimate_xonly_h3,
            case_dir / "tune_estimate_vs_delta_xonly_h3.png",
            x_key="delta_center",
            x_label=r"$\delta$",
        )
        plot_tune_estimate_vs_sweep_map(
            tune_estimate_xonly_h3,
            case_dir / "tune_estimate_vs_sweep_map_xonly_h3.png",
        )
        plot_tune_estimate_abs_vs_sweep_map(
            tune_estimate_xonly_h3,
            case_dir / "tune_estimate_abs_vs_sweep_map_xonly_h3.png",
        )


def _add_tune_shift_columns(loss: pd.DataFrame, tune_map: TuneMap | None) -> pd.DataFrame:
    frame = loss.copy()
    if tune_map is None or frame.empty:
        return frame

    clipped = np.clip(frame["delta"].to_numpy(dtype=float), tune_map.delta_min, tune_map.delta_max)
    qx_map, qy_map = tune_map(clipped)
    qx0, qy0 = tune_map(0.0, extrapolate=True)
    frame["qx_map"] = np.asarray(qx_map, dtype=float)
    frame["qy_map"] = np.asarray(qy_map, dtype=float)
    frame["dqx_map"] = frame["qx_map"] - float(qx0)
    frame["dqy_map"] = frame["qy_map"] - float(qy0)
    return frame


def plot_survival_vs_tune(
    batch_dir: Path,
    case_dirs: dict[str, Path],
    tune_map: TuneMap | None,
) -> None:
    if tune_map is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, qcol, title in zip(
        axes,
        ("qx_map", "qy_map"),
        ("Surviving fraction vs $Q_x$", "Surviving fraction vs $Q_y$"),
    ):
        for plane, color in (("DPpos", "tab:blue"), ("DPneg", "tab:orange")):
            loss = pd.read_parquet(case_dirs[plane] / "intensity_loss.parquet")
            loss = _add_tune_shift_columns(loss, tune_map)
            ax.plot(loss[qcol], loss["surviving_fraction"], label=plane, color=color, linewidth=2.0)
        ax.set_xlabel(r"$Q_x$" if qcol == "qx_map" else r"$Q_y$")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Surviving fraction")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(batch_dir / "comparison_intensity_vs_q.png", dpi=200)
    plt.close(fig)


def add_closed_orbit_x_column(
    loss: pd.DataFrame,
    *,
    delta_grid: np.ndarray,
    x_grid: np.ndarray,
) -> pd.DataFrame:
    frame = loss.copy()
    if frame.empty:
        return frame

    interpolator = CubicSpline(delta_grid, x_grid)
    clipped = np.clip(frame["delta"].to_numpy(dtype=float), float(delta_grid[0]), float(delta_grid[-1]))
    frame["x_co"] = interpolator(clipped)
    return frame


def plot_batch_comparison(
    batch_dir: Path,
    case_dirs: dict[str, Path],
    tune_map: TuneMap | None,
    *,
    closed_orbit_delta_grid: np.ndarray | None = None,
    closed_orbit_x_grid: np.ndarray | None = None,
    closed_orbit_element: str | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=False, sharey=False)

    intensity_ax = axes[0]
    for plane, color in (("DPpos", "tab:blue"), ("DPneg", "tab:orange")):
        loss = pd.read_parquet(case_dirs[plane] / "intensity_loss.parquet")
        intensity_ax.plot(loss["delta"] * 1e3, loss["surviving_fraction"], label=plane, color=color)
    intensity_ax.set_xlabel(r"$\delta$ [$10^{-3}$]")
    intensity_ax.set_ylabel("Surviving fraction")
    intensity_ax.set_title("Intensity loss vs delta")
    intensity_ax.grid(True, alpha=0.25)
    intensity_ax.legend()

    centroid_ax = axes[1]
    have_summary = all((case_dirs[plane] / "turn_summary.parquet").exists() for plane in ("DPpos", "DPneg"))
    if have_summary:
        for plane, color in (("DPpos", "tab:blue"), ("DPneg", "tab:orange")):
            summary = pd.read_parquet(case_dirs[plane] / "turn_summary.parquet")
            centroid_ax.plot(summary["turn"], summary["x_mean"], label=f"{plane} x", color=color, linestyle="-")
            centroid_ax.plot(summary["turn"], summary["y_mean"], label=f"{plane} y", color=color, linestyle="--")
        centroid_ax.set_xlabel("Turn")
        centroid_ax.set_ylabel("Centroid [m or rad]")
        centroid_ax.set_title("Monitor centroids")
        centroid_ax.grid(True, alpha=0.25)
        centroid_ax.legend()
    else:
        centroid_ax.text(0.5, 0.5, "No turn summary in loss-only mode", ha="center", va="center")
        centroid_ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(batch_dir / "comparison_overview.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for plane, color in (("DPpos", "tab:blue"), ("DPneg", "tab:orange")):
        loss = pd.read_parquet(case_dirs[plane] / "intensity_loss.parquet")
        ax.plot(loss["delta"] * 1e3, loss["surviving_fraction"], label=plane, color=color, linewidth=2.0)
    ax.set_xlabel(r"$\delta$ [$10^{-3}$]")
    ax.set_ylabel("Surviving fraction")
    ax.set_title("DPpos and DPneg vs delta")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(batch_dir / "comparison_intensity_vs_delta.png", dpi=200)
    plt.close(fig)

    if tune_map is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for ax, dcol, title in zip(
            axes,
            ("dqx_map", "dqy_map"),
            (r"Surviving fraction vs $\Delta Q_x$", r"Surviving fraction vs $\Delta Q_y$"),
        ):
            for plane, color in (("DPpos", "tab:blue"), ("DPneg", "tab:orange")):
                loss = pd.read_parquet(case_dirs[plane] / "intensity_loss.parquet")
                loss = _add_tune_shift_columns(loss, tune_map)
                ax.plot(loss[dcol], loss["surviving_fraction"], label=plane, color=color, linewidth=2.0)
            ax.set_xlabel(r"$\Delta Q_x$" if dcol == "dqx_map" else r"$\Delta Q_y$")
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
        axes[0].set_ylabel("Surviving fraction")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(batch_dir / "comparison_intensity_vs_dq.png", dpi=200)
        plt.close(fig)
        plot_survival_vs_tune(batch_dir, case_dirs, tune_map)

    if (
        closed_orbit_delta_grid is not None
        and closed_orbit_x_grid is not None
        and closed_orbit_element is not None
    ):
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for plane, color in (("DPpos", "tab:blue"), ("DPneg", "tab:orange")):
            loss = pd.read_parquet(case_dirs[plane] / "intensity_loss.parquet")
            loss = add_closed_orbit_x_column(
                loss,
                delta_grid=closed_orbit_delta_grid,
                x_grid=closed_orbit_x_grid,
            )
            ax.plot(loss["x_co"] * 1e3, loss["surviving_fraction"], label=plane, color=color, linewidth=2.0)
        ax.set_xlabel(rf"$x_{{co}}$ at {closed_orbit_element} [mm]")
        ax.set_ylabel("Surviving fraction")
        ax.set_title(f"Intensity loss vs closed orbit at {closed_orbit_element}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(batch_dir / f"comparison_intensity_vs_xco_{closed_orbit_element.replace('.', '_')}.png", dpi=200)
        plt.close(fig)


def run_case(
    args: argparse.Namespace,
    *,
    plane: str,
    batch_dir: Path,
    tune_map: TuneMap | None,
    tune_map_path: Path | None,
) -> Path:
    tune_map_case = args.tune_map_case or "LocalSextupolesOff"
    case_dir = build_case_dir(batch_dir, plane)
    signed_total_sweep_hz = repo_convention_signed_sweep(args.total_sweep_hz, plane)
    sweep_per_turn_hz = signed_total_sweep_hz / args.num_turns
    monitor_name = f"monitor_{args.monitor_element.replace('.', '_')}"

    config = CaseConfig(
        line_path=args.line_path,
        qx=args.qx,
        qy=args.qy,
        xi_x=args.xi_x,
        xi_y=args.xi_y,
        error_variant=args.error_variant,
        tune_map_case=tune_map_case,
        monitor_element=args.monitor_element,
        plane=plane,
        total_sweep_hz=signed_total_sweep_hz,
        sweep_per_turn_hz=sweep_per_turn_hz,
        num_turns=args.num_turns,
        num_particles=args.num_particles,
        nemitt_x=args.nemitt_x,
        nemitt_y=args.nemitt_y,
        sigma_z=args.sigma_z,
        naff_harmonics=args.naff_harmonics,
        fft_window=args.fft_window,
        fft_step=args.fft_step,
        omp_threads=args.omp_threads,
        sextupoles_off=True,
        output_dir=str(case_dir),
        tune_map_path=str(tune_map_path) if tune_map_path is not None else None,
        note=args.note,
    )
    with (case_dir / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(config), fh, indent=2)

    print(
        f"[sextupoles_off] Starting {plane}: total_sweep={signed_total_sweep_hz} Hz, "
        f"sweep_per_turn={sweep_per_turn_hz} Hz/turn, output={case_dir}"
    )
    if args.loss_only:
        line = configure_line_basic(
            line_path=args.line_path,
            qx=args.qx,
            qy=args.qy,
            omp_threads=args.omp_threads,
            error_variant_name=args.error_variant,
        )
        monitor_name = None
        observation_dispersion = get_line_start_dispersion(line)
    else:
        line, monitor_name = configure_line_for_observation(
            line_path=args.line_path,
            qx=args.qx,
            qy=args.qy,
            omp_threads=args.omp_threads,
            monitor_element=args.monitor_element,
            monitor_name=monitor_name,
            num_turns=args.num_turns,
            num_particles=args.num_particles,
            error_variant_name=args.error_variant,
            observation_mode=args.observation_mode,
            no_cycle=args.no_cycle,
        )
        if monitor_name is not None:
            observation_dispersion = get_monitor_dispersion(line, monitor_name)
        else:
            observation_dispersion = get_line_start_dispersion(line)
        print(
            "[sextupoles_off] Observation dispersion: "
            f"Dx={observation_dispersion['dx']}, Dpx={observation_dispersion['dpx']}, "
            f"Dy={observation_dispersion['dy']}, Dpy={observation_dispersion['dpy']}"
        )

    line.collimators.assign_optics(nemitt_x=args.nemitt_x, nemitt_y=args.nemitt_y)
    particles = xp.generate_matched_gaussian_bunch(
        nemitt_x=args.nemitt_x,
        nemitt_y=args.nemitt_y,
        sigma_z=args.sigma_z,
        num_particles=args.num_particles,
        line=line,
    )

    rf_sweep = xc.RFSweep(line)
    rf_sweep.prepare(sweep_per_turn=sweep_per_turn_hz)

    if args.loss_only:
        line.scattering.enable()
        line.track(particles=particles, num_turns=args.num_turns, time=True, with_progress=5)
        line.scattering.disable()
        save_loss_only_outputs(
            case_dir=case_dir,
            particles=particles,
            tune_map=tune_map,
            num_particles=args.num_particles,
            num_turns=args.num_turns,
            sweep_per_turn_hz=sweep_per_turn_hz,
        )
        print(f"[sextupoles_off] Finished {plane} (loss-only)")
        return case_dir

    if monitor_name is not None:
        line.scattering.enable()
        line.track(particles=particles, num_turns=args.num_turns, time=True, with_progress=5)
        line.scattering.disable()

        summary_frame, monitor_frame = monitor_to_frames(
            line[monitor_name],
            num_turns=args.num_turns,
            sweep_per_turn_hz=sweep_per_turn_hz,
        )
    else:
        summary_rows: list[dict[str, float | int]] = []
        summary_rows.append(collect_turn_row(particles, turn=0, sweep_per_turn_hz=sweep_per_turn_hz))
        line.scattering.enable()
        for turn in range(1, args.num_turns + 1):
            line.track(particles=particles, num_turns=1)
            summary_rows.append(collect_turn_row(particles, turn=turn, sweep_per_turn_hz=sweep_per_turn_hz))
        line.scattering.disable()
        summary_frame = pd.DataFrame(summary_rows)
        monitor_frame = None
    save_monitor_outputs(
        case_dir=case_dir,
        summary_frame=summary_frame,
        monitor_frame=monitor_frame,
        tune_map=tune_map,
        fft_window=args.fft_window,
        fft_step=args.fft_step,
        naff_harmonics=args.naff_harmonics,
        num_particles=args.num_particles,
        particles=particles,
        observation_dispersion=observation_dispersion,
        track_only=args.track_only,
    )
    print(f"[sextupoles_off] Finished {plane}")
    return case_dir


def main() -> None:
    args = parse_args()
    batch_dir = Path(__file__).resolve().parent / args.output_base / args.batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)
    tune_map, tune_map_path = build_tune_map_for_scenario(args=args, batch_dir=batch_dir)
    closed_orbit_delta_grid = None
    closed_orbit_x_grid = None
    if tune_map is not None:
        try:
            closed_orbit_delta_grid, closed_orbit_x_grid = build_closed_orbit_x_map(
                line_path=args.line_path,
                qx=args.qx,
                qy=args.qy,
                error_variant_name=args.error_variant,
                element_name=args.monitor_element,
                delta_range=(tune_map.delta_min, tune_map.delta_max),
            )
        except Exception as exc:
            print(f"[sextupoles_off] Failed to build closed-orbit x map at {args.monitor_element}: {exc}")

    case_dirs: dict[str, Path] = {}
    for plane in args.planes:
        case_dirs[plane] = run_case(
            args,
            plane=plane,
            batch_dir=batch_dir,
            tune_map=tune_map,
            tune_map_path=tune_map_path,
        )

    with (batch_dir / "batch_index.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "batch_dir": str(batch_dir),
                "cases": {plane: str(path) for plane, path in case_dirs.items()},
            },
            fh,
            indent=2,
        )

    if set(case_dirs) == set(DEFAULT_PLANES):
        plot_batch_comparison(
            batch_dir,
            case_dirs,
            tune_map,
            closed_orbit_delta_grid=closed_orbit_delta_grid,
            closed_orbit_x_grid=closed_orbit_x_grid,
            closed_orbit_element=args.monitor_element,
        )

    print(f"[sextupoles_off] All requested planes completed. Batch directory: {batch_dir}")


if __name__ == "__main__":
    main()
