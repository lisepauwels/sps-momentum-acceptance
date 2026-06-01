from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import xobjects as xo
import xtrack as xt


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_DIR = REPO_ROOT / "helper_functions"
sys.path.insert(0, str(HELPER_DIR))

from tune_diagram import TuneDiagram, TuneMap


DEFAULT_LINE_PATH = "/Users/lisepauwels/phd/code/sps-xsuite-model/sps_with_aperture_inj_q20_beam_sagitta4.json"
DEFAULT_OUTPUT_BASE = "dual_plane_dead_particle_diagnostics"
DEFAULT_PLANES = ("DPpos", "DPneg")
COORDS = ("x", "px", "y", "py", "zeta", "delta", "state", "particle_id", "at_turn")


ERROR_VARIANTS = {
    "none": [0, 0, 0, 0, 0, 0],
    "dipole_b3": [0, 0, 1, 0, 0, 0],
    "dipole_b5": [0, 0, 0, 0, 1, 0],
    "dipole_b3b5": [0, 0, 1, 0, 1, 0],
    "quadrupole_b4": [0, 0, 0, 1, 0, 0],
    "quadrupole_b6": [0, 0, 0, 0, 0, 1],
    "quadrupole_b4b6": [0, 0, 0, 1, 0, 1],
    "dipole_b3_quadrupole_b4": [0, 0, 1, 1, 0, 0],
    "all": [0, 0, 1, 1, 1, 1],
}


@dataclass
class PlaneRunConfig:
    line_path: str
    qx: float
    qy: float
    xi_x: float
    xi_y: float
    error_variant: str
    plane: str
    total_sweep_hz: float
    sweep_per_turn_hz: float
    num_turns: int
    num_particles: int
    nemitt_x: float
    nemitt_y: float
    sigma_z: float
    tune_map_path: str
    dx_monitor: float
    dpx_monitor: float
    dy_monitor: float
    dpy_monitor: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a notebook-style turn-by-turn RF sweep for DPpos and DPneg, "
            "save full monitor/final-particle arrays, overlay both dead-particle "
            "delta clouds on one tune diagram, and diagnose late-turn blow-up."
        )
    )
    parser.add_argument("--line-path", default=DEFAULT_LINE_PATH)
    parser.add_argument("--qx", type=float, default=20.13)
    parser.add_argument("--qy", type=float, default=20.18)
    parser.add_argument("--xi-x", type=float, default=0.5)
    parser.add_argument("--xi-y", type=float, default=0.5)
    parser.add_argument("--error-variant", choices=sorted(ERROR_VARIANTS), default="none")
    parser.add_argument("--planes", nargs="+", choices=DEFAULT_PLANES, default=list(DEFAULT_PLANES))
    parser.add_argument("--total-sweep-hz", type=float, default=3000.0)
    parser.add_argument("--num-turns", type=int, default=6000)
    parser.add_argument("--num-particles", type=int, default=100)
    parser.add_argument("--nemitt-x", type=float, default=2e-6)
    parser.add_argument("--nemitt-y", type=float, default=2e-6)
    parser.add_argument("--sigma-z", type=float, default=0.224)
    parser.add_argument("--omp-threads", default="0")
    parser.add_argument("--tune-map-path", default=None)
    parser.add_argument("--output-base", default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--batch-name", default="q20_xix0p5_xiy0p5")
    parser.add_argument("--replot-batch-dir", default=None)
    parser.add_argument("--tail-turns", type=int, default=100)
    parser.add_argument(
        "--tail-particles",
        type=int,
        default=0,
        help="Number of latest-dead particles to include; use 0 or a negative value for all dead particles.",
    )
    parser.add_argument(
        "--cohort-particles",
        type=int,
        default=1,
        help="Number of similar dead particles to show in detailed motion diagnostics.",
    )
    parser.add_argument("--particle-id-dppos", type=int, default=None)
    parser.add_argument("--particle-id-dpneg", type=int, default=None)
    return parser.parse_args()


def apply_errors(line: xt.Line, error_variant_name: str) -> None:
    env = line.env
    b1, b2, b3, b4, b5, b6 = ERROR_VARIANTS[error_variant_name]
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


def configure_line(
    *,
    line_path: str,
    qx: float,
    qy: float,
    xi_x: float,
    xi_y: float,
    error_variant: str,
    num_particles: int,
    num_turns: int,
    omp_threads: str,
) -> xt.Line:
    line = xt.load(line_path)

    _, cavity_names = line.get_elements_of_type(xt.Cavity)
    for name in cavity_names:
        line[name].frequency = 200e6
        line[name].lag = 180
        line[name].voltage = 0
    line["actcse.31632"].voltage = 3.0e6

    apply_errors(line, error_variant)

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

    line.discard_tracker()
    if omp_threads in {"auto", "openmp"}:
        context = xo.ContextCpu(omp_num_threads=0)
    else:
        context = xo.ContextCpu(omp_num_threads=int(omp_threads))
    line.build_tracker(_context=context)
    return line


def signed_total_sweep_hz(total_sweep_hz: float, plane: str) -> float:
    return -abs(total_sweep_hz) if plane == "DPpos" else abs(total_sweep_hz)


def save_npz_dict(path: Path, arrays: dict[str, np.ndarray]) -> None:
    serializable = {key: np.asarray(value) for key, value in arrays.items()}
    np.savez_compressed(path, **serializable)


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def monitor_to_arrays(monitor) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for coord in COORDS:
        arrays[coord] = np.asarray(getattr(monitor, coord))
    return arrays


def particles_to_arrays(particles) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for coord in ("x", "px", "y", "py", "zeta", "delta", "at_turn", "at_element", "state", "particle_id"):
        arrays[coord] = np.asarray(getattr(particles, coord))
    return arrays


def get_line_start_dispersion(line: xt.Line) -> dict[str, float]:
    tw = line.twiss()
    return {
        "dx": float(np.asarray(tw.dx)[0]),
        "dpx": float(np.asarray(tw.dpx)[0]),
        "dy": float(np.asarray(tw.dy)[0]),
        "dpy": float(np.asarray(tw.dpy)[0]),
    }


def run_plane(
    *,
    args: argparse.Namespace,
    plane: str,
    plane_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    import xcoll as xc
    import xpart as xp

    line = configure_line(
        line_path=args.line_path,
        qx=args.qx,
        qy=args.qy,
        xi_x=args.xi_x,
        xi_y=args.xi_y,
        error_variant=args.error_variant,
        num_particles=args.num_particles,
        num_turns=args.num_turns,
        omp_threads=args.omp_threads,
    )
    monitor_dispersion = get_line_start_dispersion(line)

    particles = xp.generate_matched_gaussian_bunch(
        nemitt_x=args.nemitt_x,
        nemitt_y=args.nemitt_y,
        sigma_z=args.sigma_z,
        num_particles=args.num_particles,
        line=line,
    )

    total_sweep = signed_total_sweep_hz(args.total_sweep_hz, plane)
    sweep_per_turn = total_sweep / float(args.num_turns)
    rf_sweep = xc.RFSweep(line)
    rf_sweep.prepare(sweep_per_turn=sweep_per_turn)

    if hasattr(line, "scattering"):
        line.scattering.enable()
    line.track(
        particles=particles,
        num_turns=args.num_turns,
        with_progress=5,
        turn_by_turn_monitor=True,
        time=True,
    )
    if hasattr(line, "scattering"):
        line.scattering.disable()

    monitor_arrays = monitor_to_arrays(line.record_last_track)
    final_arrays = particles_to_arrays(particles)

    save_npz_dict(plane_dir / "monitor_arrays.npz", monitor_arrays)
    save_npz_dict(plane_dir / "final_particles.npz", final_arrays)

    run_cfg = PlaneRunConfig(
        line_path=args.line_path,
        qx=args.qx,
        qy=args.qy,
        xi_x=args.xi_x,
        xi_y=args.xi_y,
        error_variant=args.error_variant,
        plane=plane,
        total_sweep_hz=total_sweep,
        sweep_per_turn_hz=sweep_per_turn,
        num_turns=args.num_turns,
        num_particles=args.num_particles,
        nemitt_x=args.nemitt_x,
        nemitt_y=args.nemitt_y,
        sigma_z=args.sigma_z,
        tune_map_path=str(Path(args.tune_map_path).resolve()),
        dx_monitor=monitor_dispersion["dx"],
        dpx_monitor=monitor_dispersion["dpx"],
        dy_monitor=monitor_dispersion["dy"],
        dpy_monitor=monitor_dispersion["dpy"],
    )
    with (plane_dir / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(run_cfg), fh, indent=2)

    return monitor_arrays, final_arrays, monitor_dispersion


def estimate_dispersion_from_monitor_arrays(
    monitor_arrays: dict[str, np.ndarray],
    *,
    max_turns: int = 256,
) -> dict[str, float]:
    delta_grid = np.asarray(monitor_arrays["delta"]).astype(float, copy=False)
    state_grid = np.asarray(monitor_arrays["state"]).astype(int, copy=False)
    n_turns = min(delta_grid.shape[1], max_turns)
    alive_mask = state_grid[:, :n_turns] > 0
    delta = delta_grid[:, :n_turns][alive_mask]

    if delta.size < 8:
        return {"dx": 0.0, "dpx": 0.0, "dy": 0.0, "dpy": 0.0}

    delta_centered = delta - float(np.mean(delta))
    delta_var = float(np.dot(delta_centered, delta_centered))
    if delta_var <= 1e-20:
        return {"dx": 0.0, "dpx": 0.0, "dy": 0.0, "dpy": 0.0}

    slopes: dict[str, float] = {}
    for coord in ("x", "px", "y", "py"):
        coord_grid = np.asarray(monitor_arrays[coord]).astype(float, copy=False)
        values = coord_grid[:, :n_turns][alive_mask]
        values_centered = values - float(np.mean(values))
        slopes[f"d{coord}"] = float(np.dot(delta_centered, values_centered) / delta_var)
    return slopes


def dispersion_from_run_config(
    run_config: dict[str, object],
    monitor_arrays: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    if all(key in run_config for key in ("dx_monitor", "dpx_monitor", "dy_monitor", "dpy_monitor")):
        return {
            "dx": float(run_config["dx_monitor"]),
            "dpx": float(run_config["dpx_monitor"]),
            "dy": float(run_config["dy_monitor"]),
            "dpy": float(run_config["dpy_monitor"]),
        }

    if monitor_arrays is not None:
        return estimate_dispersion_from_monitor_arrays(monitor_arrays)

    return {"dx": 0.0, "dpx": 0.0, "dy": 0.0, "dpy": 0.0}


def plot_combined_dead_particle_tune_diagram(
    *,
    batch_dir: Path,
    tune_map: TuneMap,
    final_by_plane: dict[str, dict[str, np.ndarray]],
) -> None:
    plane_specs = {
        "DPpos": {"marker": "o", "label": "DPpos dead particles"},
        "DPneg": {"marker": "s", "label": "DPneg dead particles"},
    }

    delta_arrays: list[np.ndarray] = []
    for plane in ("DPpos", "DPneg"):
        final_arrays = final_by_plane[plane]
        dead_mask = np.asarray(final_arrays["state"]) <= 0
        delta_arrays.append(np.asarray(final_arrays["delta"])[dead_mask].astype(float, copy=False))
    all_delta = np.concatenate([arr for arr in delta_arrays if arr.size > 0])
    if all_delta.size == 0:
        return

    d_map, qx_map, qy_map = tune_map.sample(500)
    qx0 = float(qx_map[np.argmin(np.abs(d_map))])
    qy0 = float(qy_map[np.argmin(np.abs(d_map))])

    td = TuneDiagram(qx0=qx0, qy0=qy0, half_range=0.4, max_order=3, skew=True)
    fig, ax = td.plot(figsize=(8.5, 7.5), show_working_point=True)
    ax.plot(qx_map, qy_map, color="0.65", linewidth=1.8, label="Sweep trajectory")

    scatter = None
    abs_all_delta = np.abs(all_delta)
    delta_abs_min = float(np.min(abs_all_delta))
    delta_abs_max = max(float(np.max(abs_all_delta)), delta_abs_min + 1e-12)
    for plane in ("DPpos", "DPneg"):
        final_arrays = final_by_plane[plane]
        dead_mask = np.asarray(final_arrays["state"]) <= 0
        dead_delta = np.asarray(final_arrays["delta"])[dead_mask].astype(float, copy=False)
        if dead_delta.size == 0:
            continue
        dead_delta_clip = np.clip(dead_delta, tune_map.delta_min, tune_map.delta_max)
        qx_dead, qy_dead = tune_map(dead_delta_clip)
        scatter = ax.scatter(
            qx_dead,
            qy_dead,
            c=np.abs(dead_delta),
            cmap="cool",
            vmin=delta_abs_min,
            vmax=delta_abs_max,
            s=18,
            alpha=0.8,
            marker=plane_specs[plane]["marker"],
            label=plane_specs[plane]["label"],
            zorder=6,
        )

    td.finalize(ax, extra_handles=None)
    ax.legend(loc="best", frameon=True)
    if scatter is not None:
        fig.colorbar(scatter, ax=ax, pad=0.02, label=r"Dead-particle $|\delta|$")
    fig.suptitle("Dead particles from DPpos and DPneg on one tune diagram")
    fig.savefig(batch_dir / "dead_particle_tune_diagram_both_planes.png", dpi=180)
    plt.close(fig)


def late_dead_particle_ids(final_arrays: dict[str, np.ndarray], max_particles: int) -> np.ndarray:
    state = np.asarray(final_arrays["state"])
    at_turn = np.asarray(final_arrays["at_turn"]).astype(int, copy=False)
    particle_id = np.asarray(final_arrays["particle_id"]).astype(int, copy=False)
    dead_mask = state <= 0
    if not np.any(dead_mask):
        return np.array([], dtype=int)
    order = np.argsort(at_turn[dead_mask])[::-1]
    dead_ids = particle_id[dead_mask][order]
    if max_particles <= 0:
        return dead_ids
    return dead_ids[:max_particles]


def similar_dead_particle_ids(
    monitor_arrays: dict[str, np.ndarray],
    final_arrays: dict[str, np.ndarray],
    *,
    coord: str,
    mom: str | None,
    dispersion: dict[str, float],
    cohort_particles: int,
) -> np.ndarray:
    dead_ids = late_dead_particle_ids(final_arrays, 0)
    if dead_ids.size == 0 or cohort_particles <= 0:
        return np.array([], dtype=int)
    if dead_ids.size <= cohort_particles:
        return dead_ids

    row_by_particle = particle_row_lookup(monitor_arrays)
    state_grid = np.asarray(monitor_arrays["state"]).astype(int, copy=False)
    records: list[tuple[int, float]] = []

    if mom is None:
        coord_grid = corrected_position_grid(monitor_arrays, coord=coord, dispersion=dispersion[f"d{coord}"])
        for pid in dead_ids:
            rr = row_by_particle.get(int(pid))
            if rr is None:
                continue
            scale = estimate_initial_position_scale(coord_grid[rr], state_grid[rr])
            if np.isfinite(scale):
                records.append((int(pid), float(scale)))
    else:
        coord_grid, mom_grid = corrected_phase_space_grid(
            monitor_arrays,
            coord=coord,
            mom=mom,
            dispersion=dispersion,
        )
        for pid in dead_ids:
            rr = row_by_particle.get(int(pid))
            if rr is None:
                continue
            scale = estimate_initial_phase_amplitude_scale(coord_grid[rr], mom_grid[rr], state_grid[rr])
            if np.isfinite(scale):
                records.append((int(pid), float(scale)))

    if len(records) <= cohort_particles:
        return np.array([pid for pid, _ in records], dtype=int)

    records.sort(key=lambda item: item[0])
    best_window: list[tuple[int, float]] | None = None
    best_score: tuple[float, float] | None = None
    for start in range(0, len(records) - cohort_particles + 1):
        window = records[start:start + cohort_particles]
        ids = np.array([pid for pid, _ in window], dtype=float)
        scales = np.array([scale for _, scale in window], dtype=float)
        amp_spread = float(np.ptp(scales) / max(np.median(scales), 1e-16))
        id_spread = float(np.ptp(ids))
        score = (amp_spread, id_spread)
        if best_score is None or score < best_score:
            best_score = score
            best_window = window

    if best_window is None:
        return np.array([], dtype=int)
    return np.array([pid for pid, _ in best_window], dtype=int)


def particle_row_lookup(monitor_arrays: dict[str, np.ndarray]) -> dict[int, int]:
    particle_id = np.asarray(monitor_arrays["particle_id"])
    if particle_id.ndim == 2:
        particle_id = particle_id[:, 0]
    return {int(pid): idx for idx, pid in enumerate(particle_id.astype(int, copy=False))}


def final_turn_lookup(final_arrays: dict[str, np.ndarray]) -> dict[int, int]:
    particle_id = np.asarray(final_arrays["particle_id"]).astype(int, copy=False)
    at_turn = np.asarray(final_arrays["at_turn"]).astype(int, copy=False)
    return {int(pid): int(turn) for pid, turn in zip(particle_id, at_turn)}


def corrected_position_grid(
    monitor_arrays: dict[str, np.ndarray],
    *,
    coord: str,
    dispersion: float,
) -> np.ndarray:
    coord_grid = np.asarray(monitor_arrays[coord]).astype(float, copy=False)
    delta_grid = np.asarray(monitor_arrays["delta"]).astype(float, copy=False)
    return coord_grid - dispersion * delta_grid


def corrected_phase_space_grid(
    monitor_arrays: dict[str, np.ndarray],
    *,
    coord: str,
    mom: str,
    dispersion: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    delta_grid = np.asarray(monitor_arrays["delta"]).astype(float, copy=False)
    coord_grid = np.asarray(monitor_arrays[coord]).astype(float, copy=False) - dispersion[f"d{coord}"] * delta_grid
    mom_grid = np.asarray(monitor_arrays[mom]).astype(float, copy=False) - dispersion[f"d{mom}"] * delta_grid
    return coord_grid, mom_grid


def phase_amplitude(coord: np.ndarray, mom: np.ndarray) -> np.ndarray:
    return np.sqrt(np.asarray(coord, dtype=float) ** 2 + np.asarray(mom, dtype=float) ** 2)


def estimate_initial_position_scale(
    position_trace: np.ndarray,
    state_trace: np.ndarray,
    *,
    reference_turns: int = 128,
) -> float:
    valid = (state_trace > 0) & np.isfinite(position_trace)
    if not np.any(valid):
        return np.nan
    first_valid = np.flatnonzero(valid)
    if first_valid.size == 0:
        return np.nan
    start = int(first_valid[0])
    stop = min(position_trace.size, start + reference_turns)
    window = position_trace[start:stop]
    state_window = state_trace[start:stop]
    window = window[(state_window > 0) & np.isfinite(window)]
    if window.size < 4:
        if window.size == 0:
            return np.nan
        return float(np.max(np.abs(window)))

    centered = window - float(np.mean(window))
    scale = float(np.sqrt(2.0) * np.std(centered))
    if not np.isfinite(scale) or scale <= 1e-16:
        scale = float(np.max(np.abs(centered)))
    return scale if np.isfinite(scale) and scale > 1e-16 else np.nan


def plot_tail_particle_traces(
    *,
    batch_dir: Path,
    monitor_by_plane: dict[str, dict[str, np.ndarray]],
    final_by_plane: dict[str, dict[str, np.ndarray]],
    dispersion_by_plane: dict[str, dict[str, float]],
    tail_turns: int,
    tail_particles: int,
    cohort_particles: int,
    particle_id_overrides: dict[str, int | None],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, constrained_layout=True)
    plane_order = ("DPpos", "DPneg")
    coord_order = ("x", "y")

    for row, plane in enumerate(plane_order):
        monitor_arrays = monitor_by_plane[plane]
        final_arrays = final_by_plane[plane]
        plane_dispersion = dispersion_by_plane[plane]
        row_by_particle = particle_row_lookup(monitor_arrays)
        death_turn_by_particle = final_turn_lookup(final_arrays)
        state_grid = np.asarray(monitor_arrays["state"]).astype(int, copy=False)
        chosen_id = particle_id_overrides.get(plane)
        if chosen_id is None:
            chosen_ids = similar_dead_particle_ids(
                monitor_arrays,
                final_arrays,
                coord="x",
                mom=None,
                dispersion=plane_dispersion,
                cohort_particles=cohort_particles,
            )
            chosen_id = int(chosen_ids[0]) if chosen_ids.size > 0 else None

        for col, coord in enumerate(coord_order):
            ax = axes[row, col]
            if chosen_id is None:
                ax.set_title(f"{plane} {coord} no dead particle selected")
                ax.set_axis_off()
                continue
            rr = row_by_particle.get(chosen_id)
            death_turn = death_turn_by_particle.get(chosen_id)
            if rr is None:
                ax.set_title(f"{plane} {coord} particle unavailable")
                ax.set_axis_off()
                continue
            coord_grid = np.asarray(monitor_arrays[coord]).astype(float, copy=False)
            turn_axis = np.arange(coord_grid.shape[1], dtype=int)
            alive_mask = state_grid[rr] > 0
            values = coord_grid[rr].copy()
            values[~alive_mask] = np.nan
            color = "tab:blue" if coord == "x" else "tab:red"
            ax.plot(turn_axis, values, linewidth=1.4, color=color, label=f"id {chosen_id}")
            if death_turn is not None:
                ax.axvline(int(death_turn), color="0.5", linewidth=0.9, linestyle="--", label="death turn")
            ax.set_title(f"{plane} {coord} position over all turns")
            ax.set_xlabel("Turn")
            ax.set_ylabel(f"{coord} [m]")
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=8, frameon=True)

    fig.savefig(batch_dir / "single_particle_positions_all_turns.png", dpi=180)
    plt.close(fig)


def alive_std_evolution(monitor_arrays: dict[str, np.ndarray], coord: str) -> np.ndarray:
    coord_grid = np.asarray(monitor_arrays[coord]).astype(float, copy=False)
    state_grid = np.asarray(monitor_arrays["state"]).astype(int, copy=False)
    stds = np.full(coord_grid.shape[1], np.nan, dtype=float)
    for jj in range(coord_grid.shape[1]):
        alive_mask = state_grid[:, jj] > 0
        values = coord_grid[alive_mask, jj]
        if values.size > 1:
            stds[jj] = float(np.std(values))
    return stds


def plot_std_explosion(
    *,
    batch_dir: Path,
    monitor_by_plane: dict[str, dict[str, np.ndarray]],
    tail_turns: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {"x": "tab:blue", "y": "tab:red"}

    for row, plane in enumerate(("DPpos", "DPneg")):
        monitor_arrays = monitor_by_plane[plane]
        turn_axis = np.arange(np.asarray(monitor_arrays["x"]).shape[1], dtype=int)
        x_std = alive_std_evolution(monitor_arrays, "x")
        y_std = alive_std_evolution(monitor_arrays, "y")

        ax_full = axes[row, 0]
        ax_tail = axes[row, 1]
        for ax in (ax_full, ax_tail):
            ax.plot(turn_axis, x_std, color=colors["x"], linewidth=1.8, label="x std")
            ax.plot(turn_axis, y_std, color=colors["y"], linewidth=1.8, label="y std")
            ax.grid(True, alpha=0.2)
            ax.set_ylabel("Std [m]")
        ax_full.set_title(f"{plane} std evolution")
        ax_full.set_xlabel("Turn")
        ax_tail.set_title(f"{plane} std evolution, last {tail_turns} turns")
        ax_tail.set_xlabel("Turn")
        ax_tail.set_xlim(max(0, turn_axis[-1] - tail_turns), turn_axis[-1])
        if row == 0:
            ax_full.legend(frameon=True)

    fig.savefig(batch_dir / "std_explosion_evolution.png", dpi=180)
    plt.close(fig)


def estimate_initial_phase_amplitude_scale(
    coord_trace: np.ndarray,
    mom_trace: np.ndarray,
    state_trace: np.ndarray,
    *,
    reference_turns: int = 128,
) -> float:
    amplitude = phase_amplitude(coord_trace, mom_trace)
    valid = (state_trace > 0) & np.isfinite(amplitude)
    if not np.any(valid):
        return np.nan
    first_valid = np.flatnonzero(valid)
    if first_valid.size == 0:
        return np.nan
    start = int(first_valid[0])
    stop = min(amplitude.size, start + reference_turns)
    window = amplitude[start:stop]
    state_window = state_trace[start:stop]
    window = window[(state_window > 0) & np.isfinite(window)]
    if window.size == 0:
        return np.nan
    scale = float(np.nanmedian(window))
    if not np.isfinite(scale) or scale <= 1e-16:
        scale = float(np.nanmax(window))
    return scale if np.isfinite(scale) and scale > 1e-16 else np.nan


def death_aligned_normalized_betatron_amplitude_matrix(
    monitor_arrays: dict[str, np.ndarray],
    final_arrays: dict[str, np.ndarray],
    *,
    coord: str,
    mom: str,
    dispersion: dict[str, float],
    tail_turns: int,
    chosen_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    row_by_particle = particle_row_lookup(monitor_arrays)
    death_turn_by_particle = final_turn_lookup(final_arrays)
    rel_turns = np.arange(-tail_turns, 1, dtype=int)
    matrix = np.full((chosen_ids.size, rel_turns.size), np.nan, dtype=float)

    coord_grid, mom_grid = corrected_phase_space_grid(
        monitor_arrays,
        coord=coord,
        mom=mom,
        dispersion=dispersion,
    )
    state_grid = np.asarray(monitor_arrays["state"]).astype(int, copy=False)

    for ii, pid in enumerate(chosen_ids):
        rr = row_by_particle.get(int(pid))
        death_turn = death_turn_by_particle.get(int(pid))
        if rr is None or death_turn is None:
            continue
        start = death_turn - tail_turns
        if start < 0:
            continue
        turn_slice = slice(start, death_turn + 1)
        scale = estimate_initial_phase_amplitude_scale(coord_grid[rr], mom_grid[rr], state_grid[rr])
        if not np.isfinite(scale):
            continue
        alive_mask = state_grid[rr, turn_slice] > 0
        values = phase_amplitude(coord_grid[rr, turn_slice], mom_grid[rr, turn_slice]) / scale
        values[~alive_mask] = np.nan
        if values.size == rel_turns.size:
            matrix[ii, :] = values

    valid_rows = np.any(np.isfinite(matrix), axis=1)
    return rel_turns, matrix[valid_rows]


def plot_death_aligned_amplitude_growth(
    *,
    batch_dir: Path,
    monitor_by_plane: dict[str, dict[str, np.ndarray]],
    final_by_plane: dict[str, dict[str, np.ndarray]],
    dispersion_by_plane: dict[str, dict[str, float]],
    tail_turns: int,
    tail_particles: int,
    cohort_particles: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for row, plane in enumerate(("DPpos", "DPneg")):
        monitor_arrays = monitor_by_plane[plane]
        final_arrays = final_by_plane[plane]
        plane_dispersion = dispersion_by_plane[plane]
        for col, coord in enumerate(("x", "y")):
            ax = axes[row, col]
            chosen_ids = similar_dead_particle_ids(
                monitor_arrays,
                final_arrays,
                coord=coord,
                mom=None,
                dispersion=plane_dispersion,
                cohort_particles=cohort_particles,
            )
            if chosen_ids.size == 0:
                ax.set_title(f"{plane} {coord} no usable late-dead particles")
                ax.set_axis_off()
                continue
            pid = int(chosen_ids[0])
            row_by_particle = particle_row_lookup(monitor_arrays)
            death_turn_by_particle = final_turn_lookup(final_arrays)
            rr = row_by_particle.get(pid)
            death_turn = death_turn_by_particle.get(pid)
            if rr is None or death_turn is None:
                ax.set_title(f"{plane} {coord} particle unavailable")
                ax.set_axis_off()
                continue

            coord_grid = corrected_position_grid(
                monitor_arrays,
                coord=coord,
                dispersion=plane_dispersion[f"d{coord}"],
            )
            state_grid = np.asarray(monitor_arrays["state"]).astype(int, copy=False)
            start = death_turn - tail_turns
            if start < 0:
                ax.set_title(f"{plane} {coord} particle dies too early")
                ax.set_axis_off()
                continue
            turn_slice = slice(start, death_turn + 1)
            rel_turns = np.arange(-tail_turns, 1, dtype=int)
            scale = estimate_initial_position_scale(coord_grid[rr], state_grid[rr])
            if not np.isfinite(scale):
                ax.set_title(f"{plane} {coord} no usable normalization")
                ax.set_axis_off()
                continue
            values = coord_grid[rr, turn_slice] / scale
            alive_mask = state_grid[rr, turn_slice] > 0
            values = np.where(alive_mask, values, np.nan)

            color = "tab:blue" if coord == "x" else "tab:red"
            ax.plot(rel_turns, values, color=color, linewidth=2.0, label=f"id {pid}")
            ax.axvline(0, color="0.5", linewidth=0.9, linestyle="--")
            ax.set_xlabel("Turns before death")
            ax.set_ylabel(f"({coord} - D{coord} delta) / A0")
            ax.set_title(f"{plane} death-aligned normalized {coord}")
            ax.grid(True, alpha=0.2)
            ax.legend(loc="upper left", fontsize=8, frameon=True)

    fig.savefig(batch_dir / "death_aligned_normalized_positions.png", dpi=180)
    plt.close(fig)


def replot_saved_batch(
    *,
    batch_dir: Path,
    tune_map_path: str | None,
    tail_turns: int,
    tail_particles: int,
    cohort_particles: int,
    particle_id_overrides: dict[str, int | None],
) -> None:
    monitor_by_plane: dict[str, dict[str, np.ndarray]] = {}
    final_by_plane: dict[str, dict[str, np.ndarray]] = {}
    dispersion_by_plane: dict[str, dict[str, float]] = {}
    saved_planes: list[str] = []
    inferred_tune_map_path = tune_map_path

    for plane_dir in sorted(path for path in batch_dir.iterdir() if path.is_dir() and path.name in DEFAULT_PLANES):
        saved_planes.append(plane_dir.name)
        monitor_by_plane[plane_dir.name] = load_npz_dict(plane_dir / "monitor_arrays.npz")
        final_by_plane[plane_dir.name] = load_npz_dict(plane_dir / "final_particles.npz")
        run_config = json.loads((plane_dir / "run_config.json").read_text(encoding="utf-8"))
        dispersion_by_plane[plane_dir.name] = dispersion_from_run_config(
            run_config,
            monitor_arrays=monitor_by_plane[plane_dir.name],
        )
        if inferred_tune_map_path is None:
            inferred_tune_map_path = str(run_config["tune_map_path"])

    if not saved_planes:
        raise FileNotFoundError(f"No saved plane folders found under {batch_dir}")
    if inferred_tune_map_path is None:
        raise ValueError("No tune map path provided and none could be inferred from saved run_config.json")

    tune_map = TuneMap.load(str(Path(inferred_tune_map_path).resolve()))
    plot_combined_dead_particle_tune_diagram(
        batch_dir=batch_dir,
        tune_map=tune_map,
        final_by_plane=final_by_plane,
    )
    plot_tail_particle_traces(
        batch_dir=batch_dir,
        monitor_by_plane=monitor_by_plane,
        final_by_plane=final_by_plane,
        dispersion_by_plane=dispersion_by_plane,
        tail_turns=tail_turns,
        tail_particles=tail_particles,
        cohort_particles=cohort_particles,
        particle_id_overrides=particle_id_overrides,
    )
    plot_std_explosion(
        batch_dir=batch_dir,
        monitor_by_plane=monitor_by_plane,
        tail_turns=tail_turns,
    )
    plot_death_aligned_amplitude_growth(
        batch_dir=batch_dir,
        monitor_by_plane=monitor_by_plane,
        final_by_plane=final_by_plane,
        dispersion_by_plane=dispersion_by_plane,
        tail_turns=tail_turns,
        tail_particles=tail_particles,
        cohort_particles=cohort_particles,
    )

    manifest_path = batch_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "batch_dir": str(batch_dir),
            "planes": saved_planes,
            "tune_map_path": str(Path(inferred_tune_map_path).resolve()),
            "tail_turns": tail_turns,
            "tail_particles": tail_particles,
            "cohort_particles": cohort_particles,
            "saved_files": [
                "DPpos/monitor_arrays.npz",
                "DPpos/final_particles.npz",
                "DPneg/monitor_arrays.npz",
                "DPneg/final_particles.npz",
                "dead_particle_tune_diagram_both_planes.png",
                "tail_particle_traces_normalized.png",
                "std_explosion_evolution.png",
                "death_aligned_normalized_positions.png",
            ],
        }
    )
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def main() -> None:
    args = parse_args()

    if args.replot_batch_dir is not None:
        replot_saved_batch(
            batch_dir=Path(args.replot_batch_dir).resolve(),
            tune_map_path=args.tune_map_path,
            tail_turns=args.tail_turns,
            tail_particles=args.tail_particles,
            cohort_particles=args.cohort_particles,
            particle_id_overrides={
                "DPpos": args.particle_id_dppos,
                "DPneg": args.particle_id_dpneg,
            },
        )
        print(f"[dual_plane_dead_particle_diagnostics] Replotted outputs in {Path(args.replot_batch_dir).resolve()}")
        return

    batch_dir = Path(__file__).resolve().parent / args.output_base / args.batch_name
    if batch_dir.exists():
        raise FileExistsError(f"Output directory already exists: {batch_dir}")
    batch_dir.mkdir(parents=True, exist_ok=False)

    if args.tune_map_path is None:
        raise ValueError("--tune-map-path is required unless --replot-batch-dir is used")

    tune_map = TuneMap.load(str(Path(args.tune_map_path).resolve()))

    monitor_by_plane: dict[str, dict[str, np.ndarray]] = {}
    final_by_plane: dict[str, dict[str, np.ndarray]] = {}
    dispersion_by_plane: dict[str, dict[str, float]] = {}

    for plane in args.planes:
        plane_dir = batch_dir / plane
        plane_dir.mkdir(parents=True, exist_ok=False)
        monitor_arrays, final_arrays, monitor_dispersion = run_plane(args=args, plane=plane, plane_dir=plane_dir)
        monitor_by_plane[plane] = monitor_arrays
        final_by_plane[plane] = final_arrays
        dispersion_by_plane[plane] = monitor_dispersion

    plot_combined_dead_particle_tune_diagram(
        batch_dir=batch_dir,
        tune_map=tune_map,
        final_by_plane=final_by_plane,
    )
    plot_tail_particle_traces(
        batch_dir=batch_dir,
        monitor_by_plane=monitor_by_plane,
        final_by_plane=final_by_plane,
        dispersion_by_plane=dispersion_by_plane,
        tail_turns=args.tail_turns,
        tail_particles=args.tail_particles,
        cohort_particles=args.cohort_particles,
        particle_id_overrides={
            "DPpos": args.particle_id_dppos,
            "DPneg": args.particle_id_dpneg,
        },
    )
    plot_std_explosion(
        batch_dir=batch_dir,
        monitor_by_plane=monitor_by_plane,
        tail_turns=args.tail_turns,
    )
    plot_death_aligned_amplitude_growth(
        batch_dir=batch_dir,
        monitor_by_plane=monitor_by_plane,
        final_by_plane=final_by_plane,
        dispersion_by_plane=dispersion_by_plane,
        tail_turns=args.tail_turns,
        tail_particles=args.tail_particles,
        cohort_particles=args.cohort_particles,
    )

    manifest = {
        "batch_dir": str(batch_dir),
        "planes": args.planes,
        "tune_map_path": str(Path(args.tune_map_path).resolve()),
        "tail_turns": args.tail_turns,
        "tail_particles": args.tail_particles,
        "cohort_particles": args.cohort_particles,
        "saved_files": [
            "DPpos/monitor_arrays.npz",
            "DPpos/final_particles.npz",
            "DPneg/monitor_arrays.npz",
            "DPneg/final_particles.npz",
            "dead_particle_tune_diagram_both_planes.png",
            "tail_particle_traces_normalized.png",
            "std_explosion_evolution.png",
            "death_aligned_normalized_positions.png",
        ],
    }
    with (batch_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[dual_plane_dead_particle_diagnostics] Wrote outputs to {batch_dir}")


if __name__ == "__main__":
    main()
