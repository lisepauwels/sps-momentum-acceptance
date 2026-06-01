from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import xtrack as xt


DEFAULT_BATCH_DIR = (
    Path(__file__).resolve().parent
    / "dual_plane_dead_particle_diagnostics"
    / "q20_xix0p5_xiy0p5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build action-distribution GIFs from saved dual-plane monitor arrays."
    )
    parser.add_argument("--batch-dir", default=str(DEFAULT_BATCH_DIR))
    parser.add_argument("--pre-loss-turns", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--coarse-step", type=int, default=50)
    parser.add_argument("--preloss-step", type=int, default=5)
    parser.add_argument("--dense-preloss-span", type=int, default=200)
    parser.add_argument("--loss-step", type=int, default=1)
    parser.add_argument("--duration-ms", type=int, default=250)
    parser.add_argument("--output-dir-name", default="action_gifs")
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


def load_optics(run_config: dict[str, object]) -> dict[str, float]:
    line = xt.load(str(run_config["line_path"]))
    _, cavity_names = line.get_elements_of_type(xt.Cavity)
    for name in cavity_names:
        line[name].frequency = 200e6
        line[name].lag = 180
        line[name].voltage = 0
    line["actcse.31632"].voltage = 3.0e6
    line.match(
        method="6d",
        vary=[
            xt.VaryList(["kqf0", "kqd0"], step=1e-8, tag="quad"),
            xt.VaryList(["qph_setvalue", "qpv_setvalue"], step=1e-4, tag="sext"),
        ],
        targets=[
            xt.TargetSet(qx=float(run_config["qx"]), qy=float(run_config["qy"]), tol=1e-6, tag="tune"),
            xt.TargetSet(
                dqx=float(run_config["xi_x"]) * float(run_config["qx"]),
                dqy=float(run_config["xi_y"]) * float(run_config["qy"]),
                tol=1e-2,
                tag="chrom",
            ),
        ],
    )
    tw = line.twiss()
    return {
        "betx": float(np.asarray(tw.betx)[0]),
        "alfx": float(np.asarray(tw.alfx)[0]),
        "bety": float(np.asarray(tw.bety)[0]),
        "alfy": float(np.asarray(tw.alfy)[0]),
        "dx": float(run_config.get("dx_monitor", 0.0)),
        "dpx": float(run_config.get("dpx_monitor", 0.0)),
        "dy": float(run_config.get("dy_monitor", 0.0)),
        "dpy": float(run_config.get("dpy_monitor", 0.0)),
    }


def action(coord: np.ndarray, mom: np.ndarray, beta: float, alpha: float) -> np.ndarray:
    gamma = (1.0 + alpha**2) / beta
    return 0.5 * (gamma * coord**2 + 2.0 * alpha * coord * mom + beta * mom**2)


def build_action_frames(
    *,
    monitor_arrays: dict[str, np.ndarray],
    optics: dict[str, float],
    plane: str,
    output_dir: Path,
    turns_to_plot: np.ndarray,
) -> list[Path]:
    state = np.asarray(monitor_arrays["state"]).astype(int, copy=False)
    delta = np.asarray(monitor_arrays["delta"]).astype(float, copy=False)
    x = np.asarray(monitor_arrays["x"]).astype(float, copy=False) - optics["dx"] * delta
    px = np.asarray(monitor_arrays["px"]).astype(float, copy=False) - optics["dpx"] * delta
    y = np.asarray(monitor_arrays["y"]).astype(float, copy=False) - optics["dy"] * delta
    py = np.asarray(monitor_arrays["py"]).astype(float, copy=False) - optics["dpy"] * delta

    jx_all = action(x, px, optics["betx"], optics["alfx"])
    jy_all = action(y, py, optics["bety"], optics["alfy"])
    jx_lim = (float(np.nanmin(jx_all)), float(np.nanmax(jx_all)))
    jy_lim = (float(np.nanmin(jy_all)), float(np.nanmax(jy_all)))

    frame_paths: list[Path] = []
    for turn in turns_to_plot:
        alive_mask = state[:, turn] > 0
        jx_turn = jx_all[alive_mask, turn]
        jy_turn = jy_all[alive_mask, turn]

        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.8), constrained_layout=True)
        for axis, values, label, color, ylim in (
            (axes[0], jx_turn, "Jx", "tab:blue", jx_lim),
            (axes[1], jy_turn, "Jy", "tab:red", jy_lim),
        ):
            if values.size > 0:
                violin = axis.violinplot([values], positions=[1], showmeans=True, showextrema=True, widths=0.8)
                for body in violin["bodies"]:
                    body.set_facecolor(color)
                    body.set_edgecolor("black")
                    body.set_alpha(0.65)
            axis.set_xticks([1])
            axis.set_xticklabels([label])
            axis.set_ylabel(label)
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
        optics = load_optics(run_config)
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
        frame_paths = build_action_frames(
            monitor_arrays=monitor_arrays,
            optics=optics,
            plane=plane,
            output_dir=frame_dir,
            turns_to_plot=turns_to_plot,
        )
        save_gif(frame_paths, outdir / f"{plane}_action_loss_window.gif", args.duration_ms)

    print(f"[saved_action_gif] Wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
