from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xcoll as xc
import xpart as xp


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_sextupoles_off_monitor_scan import (
    DEFAULT_LINE_PATH,
    add_dispersion_subtracted_columns,
    build_tune_map_for_scenario,
    build_tune_estimate_from_naff,
    build_tune_estimate_position_only,
    collect_turn_row,
    configure_line_basic,
    export_dead_particle_deltas,
    get_line_start_dispersion,
    add_sweep_map_tunes_to_estimate,
    compute_loss_curve_from_summary,
    write_frame_outputs,
)
from rf_sweep_speed_scan import (
    alive_particle_arrays,
    plot_delta_envelope,
    plot_intensity_loss,
    plot_moment_family,
    plot_naff_abs_tune_diagram,
    plot_naff_harmonics_positive_vs_sweep_map,
    plot_naff_tracks,
    plot_naff_tune_diagram,
    plot_phase_space_evolution,
    plot_phase_space_evolution_beta,
    plot_phase_space_turn_colored,
    plot_phase_space_turn_colored_beta,
    plot_spectrogram,
    plot_tune_estimate,
    plot_tune_estimate_abs_vs_sweep_map,
    plot_tune_estimate_vs_sweep_map,
    plot_violin_evolution,
    plot_violin_evolution_beta,
    save_sliding_naff,
)


@dataclass
class NoSweepConfig:
    line_path: str
    qx: float
    qy: float
    error_variant: str
    num_turns: int
    num_particles: int
    nemitt_x: float
    nemitt_y: float
    sigma_z: float
    snapshot_every: int
    naff_harmonics: int
    fft_window: int
    fft_step: int
    omp_threads: str
    output_dir: str
    tune_map_path: str | None
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track a sextupoles-off bunch without RF sweep and save line-start diagnostics."
    )
    parser.add_argument("--line-path", default=DEFAULT_LINE_PATH)
    parser.add_argument("--qx", type=float, default=20.13)
    parser.add_argument("--qy", type=float, default=20.18)
    parser.add_argument("--error-variant", default="none")
    parser.add_argument("--num-turns", type=int, default=6000)
    parser.add_argument("--num-particles", type=int, default=100)
    parser.add_argument("--nemitt-x", type=float, default=2e-6)
    parser.add_argument("--nemitt-y", type=float, default=2e-6)
    parser.add_argument("--sigma-z", type=float, default=0.224)
    parser.add_argument("--snapshot-every", type=int, default=500)
    parser.add_argument("--naff-harmonics", type=int, default=5)
    parser.add_argument("--fft-window", type=int, default=256)
    parser.add_argument("--fft-step", type=int, default=64)
    parser.add_argument("--omp-threads", default="0")
    parser.add_argument("--output-base", default="sextupoles_off_monitor_outputs")
    parser.add_argument(
        "--batch-name",
        default=f"sextupoles_off_no_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--reuse-tune-map", action="store_true")
    parser.add_argument("--force-rebuild-tune-map", action="store_true")
    parser.add_argument(
        "--note",
        default=(
            "Plain sextupoles-off tracking without RF sweep. "
            "Line start is used as the observation point."
        ),
    )
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    batch_dir = HERE / args.output_base / args.batch_name
    case_dir = batch_dir / "no_sweep"
    case_dir.mkdir(parents=True, exist_ok=False)

    # Reuse the shared sextupoles-off tune map infrastructure; the no-sweep run itself does not sweep.
    tune_map, tune_map_path = build_tune_map_for_scenario(args=args, batch_dir=batch_dir)

    config = NoSweepConfig(
        line_path=args.line_path,
        qx=args.qx,
        qy=args.qy,
        error_variant=args.error_variant,
        num_turns=args.num_turns,
        num_particles=args.num_particles,
        nemitt_x=args.nemitt_x,
        nemitt_y=args.nemitt_y,
        sigma_z=args.sigma_z,
        snapshot_every=args.snapshot_every,
        naff_harmonics=args.naff_harmonics,
        fft_window=args.fft_window,
        fft_step=args.fft_step,
        omp_threads=args.omp_threads,
        output_dir=str(case_dir),
        tune_map_path=str(tune_map_path) if tune_map_path is not None else None,
        note=args.note,
    )
    with (case_dir / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(config), fh, indent=2)

    line = configure_line_basic(
        line_path=args.line_path,
        qx=args.qx,
        qy=args.qy,
        omp_threads=args.omp_threads,
        error_variant_name=args.error_variant,
    )
    observation_dispersion = get_line_start_dispersion(line)
    line.collimators.assign_optics(nemitt_x=args.nemitt_x, nemitt_y=args.nemitt_y)
    particles = xp.generate_matched_gaussian_bunch(
        nemitt_x=args.nemitt_x,
        nemitt_y=args.nemitt_y,
        sigma_z=args.sigma_z,
        num_particles=args.num_particles,
        line=line,
    )
    initial_delta = np.asarray(particles.delta).astype(float, copy=True)

    snapshot_dir = case_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    violin_dir = case_dir / "violin_plots"
    violin_dir.mkdir(exist_ok=True)
    phase_dir = case_dir / "phase_space_plots"
    phase_dir.mkdir(exist_ok=True)

    summary_rows: list[dict[str, float | int]] = [collect_turn_row(particles, turn=0, sweep_per_turn_hz=0.0)]
    snapshot_records: list[tuple[int, dict[str, np.ndarray]]] = []
    if 0 % args.snapshot_every == 0:
        arrays0 = save_snapshot_with_initial_delta(snapshot_dir, particles, turn=0, initial_delta_by_id=initial_delta)
        snapshot_records.append((0, arrays0))
    print(
        "[sextupoles_off_no_sweep] Starting tracking: "
        f"particles={args.num_particles}, turns={args.num_turns}"
    )
    track_start = time.time()
    if hasattr(line, "scattering"):
        line.scattering.enable()
    for turn in range(1, args.num_turns + 1):
        line.track(particles=particles, num_turns=1)
        summary_rows.append(collect_turn_row(particles, turn=turn, sweep_per_turn_hz=0.0))
        if turn % args.snapshot_every == 0 or turn == args.num_turns:
            arrays = save_snapshot_with_initial_delta(snapshot_dir, particles, turn=turn, initial_delta_by_id=initial_delta)
            snapshot_records.append((turn, arrays))
        if turn % 100 == 0 or turn == args.num_turns:
            elapsed = time.time() - track_start
            print(
                "[sextupoles_off_no_sweep] Progress: "
                f"{turn}/{args.num_turns} turns in {elapsed:.1f}s"
            )
    if hasattr(line, "scattering"):
        line.scattering.disable()
    print(
        "[sextupoles_off_no_sweep] Tracking finished in "
        f"{time.time() - track_start:.1f}s"
    )

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame = add_dispersion_subtracted_columns(summary_frame, observation_dispersion)
    write_frame_outputs(case_dir / "turn_summary", summary_frame)

    plot_moment_family(summary_frame, case_dir / "mean_evolution.png", "mean")
    plot_moment_family(summary_frame, case_dir / "std_evolution.png", "std")
    plot_delta_envelope(summary_frame, case_dir / "delta_envelope_vs_time.png")
    plot_violin_evolution(violin_dir, snapshot_records)
    plot_violin_evolution_beta(violin_dir, snapshot_records, observation_dispersion)
    plot_phase_space_evolution(phase_dir, snapshot_records)
    plot_phase_space_evolution_beta(phase_dir, snapshot_records, observation_dispersion)
    plot_phase_space_turn_colored(phase_dir, snapshot_records)
    plot_phase_space_turn_colored_beta(phase_dir, snapshot_records, observation_dispersion)
    plot_phase_space_initial_delta_overlay(phase_dir, snapshot_records)

    loss_curve_frame = compute_loss_curve_from_summary(summary_frame, num_particles=args.num_particles)
    write_frame_outputs(case_dir / "intensity_loss", loss_curve_frame)
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

    plot_spectrogram(summary, case_dir / "centroid_spectrogram.png", window_size=args.fft_window, step=args.fft_step)
    plot_spectrogram(
        beta_summary,
        case_dir / "centroid_spectrogram_beta.png",
        window_size=args.fft_window,
        step=args.fft_step,
    )

    save_sliding_naff(
        summary,
        case_dir / "sliding_naff_global",
        window_size=args.fft_window,
        step=args.fft_step,
        num_harmonics=args.naff_harmonics,
    )
    save_sliding_naff(
        beta_summary,
        case_dir / "sliding_naff_beta",
        window_size=args.fft_window,
        step=args.fft_step,
        num_harmonics=args.naff_harmonics,
    )
    plot_naff_tracks(
        summary,
        case_dir / "sliding_naff_global.png",
        window_size=args.fft_window,
        step=args.fft_step,
        num_harmonics=args.naff_harmonics,
    )
    plot_naff_tracks(
        beta_summary,
        case_dir / "sliding_naff_beta.png",
        window_size=args.fft_window,
        step=args.fft_step,
        num_harmonics=args.naff_harmonics,
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

    tune_estimate = build_tune_estimate_from_naff(
        summary_frame,
        signal_columns={"horizontal": "x_beta_mean", "vertical": "y_beta_mean"},
        window_size=args.fft_window,
        step=args.fft_step,
        num_harmonics=args.naff_harmonics,
    )
    if tune_estimate is not None:
        tune_estimate = add_sweep_map_tunes_to_estimate(tune_estimate, tune_map)
        write_frame_outputs(case_dir / "tune_estimate", tune_estimate)
        plot_tune_estimate(tune_estimate, case_dir / "tune_estimate_vs_turn.png", x_key="window_center", x_label="Turn")
        plot_tune_estimate(tune_estimate, case_dir / "tune_estimate_vs_delta.png", x_key="delta_center", x_label=r"$\delta$")
        plot_tune_estimate_vs_sweep_map(tune_estimate, case_dir / "tune_estimate_vs_sweep_map.png")
        plot_tune_estimate_abs_vs_sweep_map(tune_estimate, case_dir / "tune_estimate_abs_vs_sweep_map.png")
        plot_naff_tune_diagram(tune_estimate, tune_map, case_dir / "naff_tune_diagram.png")
        plot_naff_abs_tune_diagram(tune_estimate, tune_map, case_dir / "naff_abs_tune_diagram.png")

    tune_estimate_xonly_h3 = build_tune_estimate_position_only(
        summary_frame,
        signal_columns={"horizontal": "x_beta_mean", "vertical": "y_beta_mean"},
        window_size=args.fft_window,
        step=args.fft_step,
        num_harmonics=3,
    )
    if tune_estimate_xonly_h3 is not None:
        tune_estimate_xonly_h3 = add_sweep_map_tunes_to_estimate(tune_estimate_xonly_h3, tune_map)
        write_frame_outputs(case_dir / "tune_estimate_xonly_h3", tune_estimate_xonly_h3)

    export_dead_particle_deltas(case_dir, particles, tune_map)
    with (case_dir / "death_turns.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "at_turn": np.asarray(particles.at_turn).tolist(),
                "final_state": np.asarray(particles.state).tolist(),
                "observation_dispersion": observation_dispersion,
            },
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
