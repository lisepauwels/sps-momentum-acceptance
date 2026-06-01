from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_DIR = REPO_ROOT / "helper_functions"
sys.path.insert(0, str(HELPER_DIR))

from tune_diagram import TuneDiagram, TuneMap


DEFAULT_BATCH_DIR = (
    Path(__file__).resolve().parent
    / "dual_plane_dead_particle_diagnostics"
    / "q20_xix0p5_xiy0p5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run sliding-window NAFF on one saved particle per plane from an existing "
            "dual-plane diagnostics batch and save tune-evolution plots."
        )
    )
    parser.add_argument("--batch-dir", default=str(DEFAULT_BATCH_DIR))
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--num-harmonics", type=int, default=1)
    parser.add_argument("--particle-id-dppos", type=int, default=None)
    parser.add_argument("--particle-id-dpneg", type=int, default=None)
    parser.add_argument("--output-dir-name", default="particle_median_naff")
    return parser.parse_args()


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def particle_row_lookup(monitor_arrays: dict[str, np.ndarray]) -> dict[int, int]:
    particle_id = np.asarray(monitor_arrays["particle_id"])
    if particle_id.ndim == 2:
        particle_id = particle_id[:, 0]
    return {int(pid): idx for idx, pid in enumerate(particle_id.astype(int, copy=False))}


def final_turn_lookup(final_arrays: dict[str, np.ndarray]) -> dict[int, int]:
    particle_id = np.asarray(final_arrays["particle_id"]).astype(int, copy=False)
    at_turn = np.asarray(final_arrays["at_turn"]).astype(int, copy=False)
    return {int(pid): int(turn) for pid, turn in zip(particle_id, at_turn)}


def latest_dead_particle_id(final_arrays: dict[str, np.ndarray]) -> int:
    state = np.asarray(final_arrays["state"])
    at_turn = np.asarray(final_arrays["at_turn"]).astype(int, copy=False)
    particle_id = np.asarray(final_arrays["particle_id"]).astype(int, copy=False)
    dead_mask = state <= 0
    if not np.any(dead_mask):
        raise ValueError("No dead particles found in final arrays.")
    order = np.argsort(at_turn[dead_mask])[::-1]
    return int(particle_id[dead_mask][order][0])


def all_particle_ids(monitor_arrays: dict[str, np.ndarray]) -> np.ndarray:
    particle_id = np.asarray(monitor_arrays["particle_id"])
    if particle_id.ndim == 2:
        particle_id = particle_id[:, 0]
    return particle_id.astype(int, copy=False)


def sliding_naff_1d(
    signal: np.ndarray,
    *,
    window_size: int,
    step: int,
    num_harmonics: int,
) -> pd.DataFrame:
    import nafflib

    signal = np.asarray(signal, dtype=float)
    rows: list[dict[str, float | int]] = []
    for start in range(0, len(signal) - window_size + 1, step):
        stop = start + window_size
        segment = signal[start:stop]
        if not np.all(np.isfinite(segment)):
            continue
        try:
            amplitudes, frequencies = nafflib.harmonics(
                segment,
                num_harmonics=num_harmonics,
                window_order=2,
                window_type="hann",
            )
        except Exception:
            continue
        if len(frequencies) == 0:
            continue
        rows.append(
            {
                "window_start": start,
                "window_stop": stop,
                "window_center": start + window_size // 2,
                "frequency": float(frequencies[0]),
                "amplitude_abs": float(np.abs(amplitudes[0])),
            }
        )

    if not rows:
        raise RuntimeError("NAFF produced no windows.")
    return pd.DataFrame(rows)


def plot_tune_vs_turn(track: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    axes[0].plot(track["turn_center"], track["qx_plot"], marker="o", markersize=2.5, linewidth=1.0)
    axes[0].set_ylabel(r"$Q_x$")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(track["turn_center"], track["qy_plot"], marker="o", markersize=2.5, linewidth=1.0)
    axes[1].set_ylabel(r"$Q_y$")
    axes[1].set_xlabel("Turn")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_tune_vs_turn_with_spread(track: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)

    axes[0].plot(track["turn_center"], track["qx_plot"], linewidth=1.2, color="tab:blue", label="median")
    if {"qx_q25", "qx_q75"}.issubset(track.columns):
        axes[0].fill_between(track["turn_center"], track["qx_q25"], track["qx_q75"], color="tab:blue", alpha=0.2, label="IQR")
    axes[0].set_ylabel(r"$Q_x$")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=True, fontsize=8)

    axes[1].plot(track["turn_center"], track["qy_plot"], linewidth=1.2, color="tab:red", label="median")
    if {"qy_q25", "qy_q75"}.issubset(track.columns):
        axes[1].fill_between(track["turn_center"], track["qy_q25"], track["qy_q75"], color="tab:red", alpha=0.2, label="IQR")
    axes[1].set_ylabel(r"$Q_y$")
    axes[1].set_xlabel("Turn")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=True, fontsize=8)

    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_tune_diagram_overlay(
    track: pd.DataFrame,
    tune_map: TuneMap,
    output_path: Path,
    title: str,
) -> None:
    d_map, qx_map, qy_map = tune_map.sample(500)
    qx0 = float(qx_map[np.argmin(np.abs(d_map))])
    qy0 = float(qy_map[np.argmin(np.abs(d_map))])

    td = TuneDiagram(qx0=qx0, qy0=qy0, half_range=0.4, max_order=3, skew=True)
    fig, ax = td.plot(figsize=(8.5, 7.5), show_working_point=True)
    ax.plot(qx_map, qy_map, color="0.65", linewidth=1.8, label="Sweep trajectory")
    scatter = ax.scatter(
        track["qx_full"],
        track["qy_full"],
        c=track["turn_center"],
        cmap="plasma",
        s=18,
        alpha=0.85,
        zorder=6,
        label="Particle NAFF track",
    )
    td.finalize(ax, extra_handles=None)
    ax.legend(loc="best", frameon=True)
    fig.colorbar(scatter, ax=ax, pad=0.02, label="Turn")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_particle_track(
    *,
    monitor_arrays: dict[str, np.ndarray],
    final_arrays: dict[str, np.ndarray],
    run_config: dict[str, object],
    particle_id: int,
    window_size: int,
    step: int,
    num_harmonics: int,
) -> pd.DataFrame:
    row_by_particle = particle_row_lookup(monitor_arrays)
    death_turn_by_particle = final_turn_lookup(final_arrays)
    row = row_by_particle[particle_id]
    death_turn = death_turn_by_particle.get(particle_id, np.asarray(monitor_arrays["x"]).shape[1] - 1)
    stop = min(int(death_turn) + 1, np.asarray(monitor_arrays["x"]).shape[1])

    state = np.asarray(monitor_arrays["state"][row, :stop]).astype(int, copy=False)
    alive_mask = state > 0
    turns = np.arange(stop, dtype=int)[alive_mask]
    delta = np.asarray(monitor_arrays["delta"][row, :stop]).astype(float, copy=False)[alive_mask]

    dx = float(run_config.get("dx_monitor", 0.0))
    dy = float(run_config.get("dy_monitor", 0.0))
    x = np.asarray(monitor_arrays["x"][row, :stop]).astype(float, copy=False)[alive_mask] - dx * delta
    y = np.asarray(monitor_arrays["y"][row, :stop]).astype(float, copy=False)[alive_mask] - dy * delta

    qx_track = sliding_naff_1d(x, window_size=window_size, step=step, num_harmonics=num_harmonics)
    qy_track = sliding_naff_1d(y, window_size=window_size, step=step, num_harmonics=num_harmonics)

    merged = qx_track[["window_center", "frequency", "amplitude_abs"]].rename(
        columns={"frequency": "qx_estimate", "amplitude_abs": "qx_amplitude"}
    )
    merged = merged.merge(
        qy_track[["window_center", "frequency", "amplitude_abs"]].rename(
            columns={"frequency": "qy_estimate", "amplitude_abs": "qy_amplitude"}
        ),
        on="window_center",
        how="inner",
    )

    centers = merged["window_center"].to_numpy(dtype=int)
    merged["turn_center"] = turns[centers]
    merged["delta_center"] = delta[centers]
    merged["qx_plot"] = np.abs(merged["qx_estimate"].to_numpy(dtype=float))
    merged["qy_plot"] = np.abs(merged["qy_estimate"].to_numpy(dtype=float))
    return merged


def add_tune_map_lift(track: pd.DataFrame, tune_map: TuneMap) -> pd.DataFrame:
    frame = track.copy()
    delta_center = frame["delta_center"].to_numpy(dtype=float)
    clipped = np.clip(delta_center, tune_map.delta_min, tune_map.delta_max)
    qx_map, qy_map = tune_map(clipped)
    qx_map = np.asarray(qx_map, dtype=float)
    qy_map = np.asarray(qy_map, dtype=float)
    frame["qx_map"] = qx_map
    frame["qy_map"] = qy_map
    frame["qx_full"] = np.floor(qx_map) + frame["qx_plot"].to_numpy(dtype=float)
    frame["qy_full"] = np.floor(qy_map) + frame["qy_plot"].to_numpy(dtype=float)
    return frame


def process_plane(
    *,
    batch_dir: Path,
    plane: str,
    particle_id_override: int | None,
    window_size: int,
    step: int,
    num_harmonics: int,
    output_dir_name: str,
) -> tuple[Path, int, int]:
    plane_dir = batch_dir / plane
    monitor_arrays = load_npz_dict(plane_dir / "monitor_arrays.npz")
    final_arrays = load_npz_dict(plane_dir / "final_particles.npz")
    run_config = json.loads((plane_dir / "run_config.json").read_text(encoding="utf-8"))
    particle_id = particle_id_override if particle_id_override is not None else latest_dead_particle_id(final_arrays)
    particle_ids = all_particle_ids(monitor_arrays)

    tune_map = TuneMap.load(str(Path(run_config["tune_map_path"]).resolve()))
    track = build_particle_track(
        monitor_arrays=monitor_arrays,
        final_arrays=final_arrays,
        run_config=run_config,
        particle_id=particle_id,
        window_size=window_size,
        step=step,
        num_harmonics=num_harmonics,
    )
    track = add_tune_map_lift(track, tune_map)

    all_tracks: list[pd.DataFrame] = []
    skipped = 0
    for pid in particle_ids:
        try:
            pid_track = build_particle_track(
                monitor_arrays=monitor_arrays,
                final_arrays=final_arrays,
                run_config=run_config,
                particle_id=int(pid),
                window_size=window_size,
                step=step,
                num_harmonics=num_harmonics,
            )
            pid_track["particle_id"] = int(pid)
            all_tracks.append(pid_track)
        except Exception:
            skipped += 1

    if not all_tracks:
        raise RuntimeError(f"No particle NAFF tracks could be built for {plane}.")

    all_track_frame = pd.concat(all_tracks, ignore_index=True)
    median_track = (
        all_track_frame.groupby("window_center", as_index=False)
        .agg(
            turn_center=("turn_center", "median"),
            delta_center=("delta_center", "median"),
            qx_plot=("qx_plot", "median"),
            qy_plot=("qy_plot", "median"),
            qx_q25=("qx_plot", lambda s: float(np.nanpercentile(s, 25))),
            qx_q75=("qx_plot", lambda s: float(np.nanpercentile(s, 75))),
            qy_q25=("qy_plot", lambda s: float(np.nanpercentile(s, 25))),
            qy_q75=("qy_plot", lambda s: float(np.nanpercentile(s, 75))),
            n_particles=("particle_id", "nunique"),
        )
    )
    median_track = add_tune_map_lift(median_track, tune_map)

    outdir = batch_dir / output_dir_name / f"{plane}_median_all_particles"
    outdir.mkdir(parents=True, exist_ok=True)
    track.to_parquet(outdir / "particle_tune_track.parquet", index=False)
    all_track_frame.to_parquet(outdir / "all_particle_tune_tracks.parquet", index=False)
    median_track.to_parquet(outdir / "median_tune_track.parquet", index=False)
    plot_tune_vs_turn(
        track,
        outdir / "particle_tune_vs_turn.png",
        title=f"{plane} particle {particle_id} sliding-window NAFF",
    )
    plot_tune_diagram_overlay(
        track,
        tune_map,
        outdir / "particle_naff_tune_diagram.png",
        title=f"{plane} particle {particle_id} NAFF track on tune diagram",
    )
    plot_tune_vs_turn_with_spread(
        median_track,
        outdir / "median_tune_vs_turn.png",
        title=f"{plane} median sliding-window NAFF over all particles",
    )
    plot_tune_diagram_overlay(
        median_track,
        tune_map,
        outdir / "median_naff_tune_diagram.png",
        title=f"{plane} median NAFF track on tune diagram",
    )
    return outdir, particle_id, int(len(particle_ids) - skipped)


def main() -> None:
    args = parse_args()
    batch_dir = Path(args.batch_dir).resolve()
    batch_outdir = batch_dir / args.output_dir_name
    batch_outdir.mkdir(parents=True, exist_ok=True)

    results = {}
    for plane, override in (
        ("DPpos", args.particle_id_dppos),
        ("DPneg", args.particle_id_dpneg),
    ):
        outdir, particle_id, num_success = process_plane(
            batch_dir=batch_dir,
            plane=plane,
            particle_id_override=override,
            window_size=args.window_size,
            step=args.step,
            num_harmonics=args.num_harmonics,
            output_dir_name=args.output_dir_name,
        )
        results[plane] = {
            "reference_particle_id": particle_id,
            "num_particles_used": num_success,
            "output_dir": str(outdir),
        }

    with (batch_outdir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "batch_dir": str(batch_dir),
                "window_size": args.window_size,
                "step": args.step,
                "num_harmonics": args.num_harmonics,
                "results": results,
            },
            fh,
            indent=2,
        )

    print(f"[saved_single_particle_naff] Wrote outputs to {batch_outdir}")


if __name__ == "__main__":
    main()
