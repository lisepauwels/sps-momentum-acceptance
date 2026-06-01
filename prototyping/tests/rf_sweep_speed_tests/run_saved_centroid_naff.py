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
        description="Run centroid-based sliding-window NAFF from saved dual-plane monitor arrays."
    )
    parser.add_argument("--batch-dir", default=str(DEFAULT_BATCH_DIR))
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--step", type=int, default=64)
    parser.add_argument("--num-harmonics", type=int, default=6)
    parser.add_argument("--output-dir-name", default="centroid_naff")
    return parser.parse_args()


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def build_summary_frame(
    monitor_arrays: dict[str, np.ndarray],
    run_config: dict[str, object],
) -> pd.DataFrame:
    state = np.asarray(monitor_arrays["state"]).astype(int, copy=False)
    n_turns = state.shape[1]
    rows: list[dict[str, float | int]] = []
    dx = float(run_config.get("dx_monitor", 0.0))
    dpx = float(run_config.get("dpx_monitor", 0.0))
    dy = float(run_config.get("dy_monitor", 0.0))
    dpy = float(run_config.get("dpy_monitor", 0.0))
    sweep_per_turn_hz = float(run_config["sweep_per_turn_hz"])

    for turn in range(n_turns):
        alive_mask = state[:, turn] > 0
        row: dict[str, float | int] = {
            "turn": turn,
            "alive_count": int(np.sum(alive_mask)),
            "delta_from_sweep": sweep_per_turn_hz * turn,
        }
        for coord in ("x", "px", "y", "py", "delta"):
            values = np.asarray(monitor_arrays[coord])[:, turn].astype(float, copy=False)
            values = values[alive_mask]
            row[f"{coord}_mean"] = float(np.mean(values)) if values.size else np.nan
        row["x_beta_mean"] = row["x_mean"] - dx * row["delta_mean"]
        row["px_beta_mean"] = row["px_mean"] - dpx * row["delta_mean"]
        row["y_beta_mean"] = row["y_mean"] - dy * row["delta_mean"]
        row["py_beta_mean"] = row["py_mean"] - dpy * row["delta_mean"]
        rows.append(row)

    return pd.DataFrame(rows)


def sliding_naff(
    signal: np.ndarray,
    signal_px: np.ndarray | None,
    *,
    window_size: int,
    step: int,
    num_harmonics: int,
) -> pd.DataFrame:
    import nafflib

    signal = np.asarray(signal, dtype=float)
    signal = np.nan_to_num(signal, nan=0.0)
    signal_px_arr = None
    if signal_px is not None:
        signal_px_arr = np.asarray(signal_px, dtype=float)
        signal_px_arr = np.nan_to_num(signal_px_arr, nan=0.0)
    rows: list[dict[str, float | int]] = []
    for start in range(0, len(signal) - window_size + 1, step):
        stop = start + window_size
        segment = signal[start:stop] - np.mean(signal[start:stop])
        segment_px = None
        if signal_px_arr is not None:
            segment_px = signal_px_arr[start:stop] - np.mean(signal_px_arr[start:stop])
        if not np.any(np.abs(segment) > 0):
            continue
        try:
            if segment_px is None:
                amps, freqs = nafflib.harmonics(
                    segment,
                    num_harmonics=num_harmonics,
                    window_order=2,
                    window_type="hann",
                )
            else:
                amps, freqs = nafflib.harmonics(
                    segment,
                    segment_px,
                    num_harmonics=num_harmonics,
                    window_order=2,
                    window_type="hann",
                )
        except Exception:
            continue
        for harmonic, (amp, freq) in enumerate(zip(amps, freqs)):
            rows.append(
                {
                    "window_center": start + window_size // 2,
                    "harmonic": harmonic,
                    "frequency": float(freq),
                    "amplitude_abs": float(np.abs(amp)),
                }
            )
    if not rows:
        raise RuntimeError("NAFF produced no windows.")
    return pd.DataFrame(rows)


def select_harmonic_near_tune_map(
    frame: pd.DataFrame,
    *,
    summary_frame: pd.DataFrame,
    tune_map: TuneMap,
    plane: str,
) -> pd.DataFrame:
    out_rows: list[dict[str, float | int]] = []
    turn = summary_frame["turn"].to_numpy(dtype=float)
    delta = summary_frame["delta_mean"].to_numpy(dtype=float)

    for window_center, sub in frame.groupby("window_center"):
        delta_center = float(np.interp(float(window_center), turn, delta))
        clipped = float(np.clip(delta_center, tune_map.delta_min, tune_map.delta_max))
        qx_map, qy_map = tune_map(np.array([clipped]))
        target = float(np.mod(qx_map[0] if plane == "horizontal" else qy_map[0], 1.0))

        candidates = sub[sub["frequency"] > 0].copy()
        if candidates.empty:
            continue
        candidates["distance_to_map"] = np.abs(candidates["frequency"].to_numpy(dtype=float) - target)
        best = candidates.sort_values(["distance_to_map", "harmonic", "amplitude_abs"], ascending=[True, True, False]).iloc[0]
        out_rows.append(
            {
                "window_center": int(window_center),
                "frequency": float(best["frequency"]),
                "amplitude_abs": float(best["amplitude_abs"]),
                "harmonic": int(best["harmonic"]),
                "target_fractional": target,
                "delta_center": delta_center,
            }
        )

    if not out_rows:
        raise RuntimeError(f"No positive harmonics could be matched to the tune map for {plane}.")
    return pd.DataFrame(out_rows)


def build_tune_estimate_from_centroid_summary(
    summary_frame: pd.DataFrame,
    *,
    tune_map: TuneMap,
    window_size: int,
    step: int,
    num_harmonics: int,
) -> pd.DataFrame:
    qx_all = sliding_naff(
        summary_frame["x_beta_mean"].to_numpy(dtype=float),
        summary_frame["px_beta_mean"].to_numpy(dtype=float),
        window_size=window_size,
        step=step,
        num_harmonics=num_harmonics,
    )
    qy_all = sliding_naff(
        summary_frame["y_beta_mean"].to_numpy(dtype=float),
        summary_frame["py_beta_mean"].to_numpy(dtype=float),
        window_size=window_size,
        step=step,
        num_harmonics=num_harmonics,
    )
    qx = select_harmonic_near_tune_map(qx_all, summary_frame=summary_frame, tune_map=tune_map, plane="horizontal").rename(
        columns={"frequency": "qx_estimate", "amplitude_abs": "qx_amplitude", "harmonic": "qx_harmonic"}
    )
    qy = select_harmonic_near_tune_map(qy_all, summary_frame=summary_frame, tune_map=tune_map, plane="vertical").rename(
        columns={"frequency": "qy_estimate", "amplitude_abs": "qy_amplitude", "harmonic": "qy_harmonic"}
    )
    merged = qx.merge(qy, on="window_center", how="inner")
    merged["turn_center"] = merged["window_center"].to_numpy(dtype=float)
    if "delta_center_x" in merged.columns and "delta_center_y" in merged.columns:
        merged["delta_center"] = 0.5 * (
            merged["delta_center_x"].to_numpy(dtype=float) + merged["delta_center_y"].to_numpy(dtype=float)
        )
        merged = merged.drop(columns=["delta_center_x", "delta_center_y"])
    elif "delta_center" not in merged.columns:
        raise RuntimeError("Merged tune estimate is missing delta_center.")
    merged["qx_plot"] = np.abs(merged["qx_estimate"].to_numpy(dtype=float))
    merged["qy_plot"] = np.abs(merged["qy_estimate"].to_numpy(dtype=float))
    return merged


def add_tune_map_lift(track: pd.DataFrame, tune_map: TuneMap) -> pd.DataFrame:
    frame = track.copy()
    clipped = np.clip(frame["delta_center"].to_numpy(dtype=float), tune_map.delta_min, tune_map.delta_max)
    qx_map, qy_map = tune_map(clipped)
    qx_map = np.asarray(qx_map, dtype=float)
    qy_map = np.asarray(qy_map, dtype=float)
    frame["qx_map"] = qx_map
    frame["qy_map"] = qy_map
    frame["qx_full"] = np.floor(qx_map) + frame["qx_plot"].to_numpy(dtype=float)
    frame["qy_full"] = np.floor(qy_map) + frame["qy_plot"].to_numpy(dtype=float)
    frame["qx_residual"] = frame["qx_plot"] - np.mod(qx_map, 1.0)
    frame["qy_residual"] = frame["qy_plot"] - np.mod(qy_map, 1.0)
    return frame


def plot_tune_vs_turn(track: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    axes[0].plot(track["turn_center"], track["qx_plot"], marker="o", markersize=3, linewidth=1.1)
    axes[0].set_ylabel(r"$Q_x$")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(track["turn_center"], track["qy_plot"], marker="o", markersize=3, linewidth=1.1)
    axes[1].set_ylabel(r"$Q_y$")
    axes[1].set_xlabel("Turn")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_tune_diagram_overlay(track: pd.DataFrame, tune_map: TuneMap, output_path: Path, title: str) -> None:
    d_map, qx_map, qy_map = tune_map.sample(500)
    qx0 = float(qx_map[np.argmin(np.abs(d_map))])
    qy0 = float(qy_map[np.argmin(np.abs(d_map))])
    td = TuneDiagram(qx0=qx0, qy0=qy0, half_range=0.4, max_order=3, skew=True)
    fig, ax = td.plot(figsize=(8.5, 7.5), show_working_point=True)
    ax.plot(qx_map, qy_map, color="0.65", linewidth=1.8, label="Sweep trajectory")
    sc = ax.scatter(
        track["qx_full"],
        track["qy_full"],
        c=track["turn_center"],
        cmap="plasma",
        s=22,
        alpha=0.85,
        zorder=6,
        label="Centroid NAFF track",
    )
    td.finalize(ax, extra_handles=None)
    ax.legend(loc="best", frameon=True)
    fig.colorbar(sc, ax=ax, pad=0.02, label="Turn")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    batch_dir = Path(args.batch_dir).resolve()
    outroot = batch_dir / args.output_dir_name
    outroot.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "batch_dir": str(batch_dir),
        "window_size": args.window_size,
        "step": args.step,
        "num_harmonics": args.num_harmonics,
        "planes": {},
    }

    for plane in ("DPpos", "DPneg"):
        plane_dir = batch_dir / plane
        outdir = outroot / plane
        outdir.mkdir(parents=True, exist_ok=True)
        monitor_arrays = load_npz_dict(plane_dir / "monitor_arrays.npz")
        run_config = json.loads((plane_dir / "run_config.json").read_text(encoding="utf-8"))
        tune_map = TuneMap.load(str(Path(run_config["tune_map_path"]).resolve()))
        summary = build_summary_frame(monitor_arrays, run_config)
        tune_estimate = build_tune_estimate_from_centroid_summary(
            summary,
            tune_map=tune_map,
            window_size=args.window_size,
            step=args.step,
            num_harmonics=args.num_harmonics,
        )
        tune_estimate = add_tune_map_lift(tune_estimate, tune_map)
        summary.to_csv(outdir / "centroid_summary.csv", index=False)
        tune_estimate.to_csv(outdir / "centroid_tune_estimate.csv", index=False)
        plot_tune_vs_turn(
            tune_estimate,
            outdir / "centroid_tune_vs_turn.png",
            title=f"{plane} centroid sliding-window NAFF",
        )
        plot_tune_diagram_overlay(
            tune_estimate,
            tune_map,
            outdir / "centroid_naff_tune_diagram.png",
            title=f"{plane} centroid NAFF track on tune diagram",
        )
        manifest["planes"][plane] = {"output_dir": str(outdir), "n_windows": int(len(tune_estimate))}

    with (outroot / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[saved_centroid_naff] Wrote outputs to {outroot}")


if __name__ == "__main__":
    main()
