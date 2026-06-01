from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DEFAULT_BATCH_DIR = (
    Path(__file__).resolve().parent
    / "dual_plane_dead_particle_diagnostics"
    / "q20_xix0p5_xiy0p5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build violin-plot GIFs from saved dual-plane monitor arrays."
    )
    parser.add_argument("--batch-dir", default=str(DEFAULT_BATCH_DIR))
    parser.add_argument("--pre-loss-turns", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--coarse-step", type=int, default=50)
    parser.add_argument("--preloss-step", type=int, default=5)
    parser.add_argument("--dense-preloss-span", type=int, default=200)
    parser.add_argument("--loss-step", type=int, default=1)
    parser.add_argument("--duration-ms", type=int, default=250)
    parser.add_argument("--output-dir-name", default="violin_gifs")
    return parser.parse_args()


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def loss_turn_window(final_arrays: dict[str, np.ndarray], pre_loss_turns: int) -> tuple[int, int]:
    state = np.asarray(final_arrays["state"])
    at_turn = np.asarray(final_arrays["at_turn"]).astype(int, copy=False)
    dead_mask = state <= 0
    if not np.any(dead_mask):
        raise ValueError("No dead particles found; cannot build loss-window GIF.")
    dead_turns = np.sort(at_turn[dead_mask])
    start_turn = max(0, int(dead_turns[0]) - pre_loss_turns)
    stop_turn = int(dead_turns[-1])
    return start_turn, stop_turn


def build_sampled_turns(
    *,
    n_turns: int,
    first_loss_turn: int,
    last_loss_turn: int,
    coarse_step: int,
    preloss_step: int,
    dense_preloss_span: int,
    loss_step: int,
    max_frames: int,
) -> np.ndarray:
    preloss_dense_start = max(0, first_loss_turn - dense_preloss_span)

    early = np.arange(0, preloss_dense_start, max(coarse_step, 1), dtype=int)
    preloss = np.arange(preloss_dense_start, first_loss_turn, max(preloss_step, 1), dtype=int)
    loss = np.arange(first_loss_turn, min(last_loss_turn + 1, n_turns), max(loss_step, 1), dtype=int)

    turns = np.unique(np.concatenate([early, preloss, loss]))
    turns = turns[(turns >= 0) & (turns < n_turns)]

    if turns.size > max_frames:
        sample_idx = np.linspace(0, turns.size - 1, max_frames, dtype=int)
        turns = turns[sample_idx]
    return turns


def build_violin_frames(
    *,
    monitor_arrays: dict[str, np.ndarray],
    run_config: dict[str, object],
    plane: str,
    output_dir: Path,
    turns_to_plot: np.ndarray,
) -> list[Path]:
    state = np.asarray(monitor_arrays["state"]).astype(int, copy=False)
    x = np.asarray(monitor_arrays["x"]).astype(float, copy=False)
    y = np.asarray(monitor_arrays["y"]).astype(float, copy=False)
    delta = np.asarray(monitor_arrays["delta"]).astype(float, copy=False)
    dx = float(run_config.get("dx_monitor", 0.0))
    dy = float(run_config.get("dy_monitor", 0.0))
    x = x - dx * delta
    y = y - dy * delta
    x_all = x[np.isfinite(x)]
    y_all = y[np.isfinite(y)]
    x_lim = (float(np.nanmin(x_all)), float(np.nanmax(x_all)))
    y_lim = (float(np.nanmin(y_all)), float(np.nanmax(y_all)))

    frame_paths: list[Path] = []
    for turn in turns_to_plot:
        alive_mask = state[:, turn] > 0
        x_turn = x[alive_mask, turn]
        y_turn = y[alive_mask, turn]

        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.8), constrained_layout=True)
        for axis, values, coord, color, ylim in (
            (axes[0], x_turn, "x", "tab:blue", x_lim),
            (axes[1], y_turn, "y", "tab:red", y_lim),
        ):
            if values.size > 0:
                violin = axis.violinplot([values], positions=[1], showmeans=True, showextrema=True, widths=0.8)
                for body in violin["bodies"]:
                    body.set_facecolor(color)
                    body.set_edgecolor("black")
                    body.set_alpha(0.65)
            axis.set_xticks([1])
            axis.set_xticklabels([coord])
            axis.set_ylabel(f"{coord} [m]")
            axis.set_ylim(*ylim)
            axis.grid(True, alpha=0.2)

        fig.suptitle(f"{plane} turn {turn}  |  alive particles: {int(np.sum(alive_mask))}")
        frame_path = output_dir / f"frame_{turn:05d}.png"
        fig.savefig(frame_path, dpi=160)
        plt.close(fig)
        frame_paths.append(frame_path)

    return frame_paths


def save_gif(frame_paths: list[Path], output_path: Path, duration_ms: int) -> None:
    if not frame_paths:
        return
    images = [Image.open(path) for path in frame_paths]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    for image in images:
        image.close()


def main() -> None:
    args = parse_args()
    batch_dir = Path(args.batch_dir).resolve()
    outdir = batch_dir / args.output_dir_name
    outdir.mkdir(parents=True, exist_ok=True)

    for plane in ("DPpos", "DPneg"):
        plane_dir = batch_dir / plane
        monitor_arrays = load_npz_dict(plane_dir / "monitor_arrays.npz")
        final_arrays = load_npz_dict(plane_dir / "final_particles.npz")
        run_config = json.loads((plane_dir / "run_config.json").read_text(encoding="utf-8"))
        start_turn, stop_turn = loss_turn_window(final_arrays, args.pre_loss_turns)
        n_turns = int(np.asarray(monitor_arrays["x"]).shape[1])
        first_loss_turn = start_turn + args.pre_loss_turns
        turns_to_plot = build_sampled_turns(
            n_turns=n_turns,
            first_loss_turn=first_loss_turn,
            last_loss_turn=stop_turn,
            coarse_step=args.coarse_step,
            preloss_step=args.preloss_step,
            dense_preloss_span=args.dense_preloss_span,
            loss_step=args.loss_step,
            max_frames=args.max_frames,
        )
        frame_dir = outdir / f"{plane}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = build_violin_frames(
            monitor_arrays=monitor_arrays,
            run_config=run_config,
            plane=plane,
            output_dir=frame_dir,
            turns_to_plot=turns_to_plot,
        )
        save_gif(
            frame_paths,
            outdir / f"{plane}_violin_loss_window.gif",
            duration_ms=args.duration_ms,
        )

    print(f"[saved_violin_gif] Wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
