from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "helper_functions"))
sys.path.insert(0, str(REPO_ROOT / "tune_scan_workflow"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rf_sweep_speed_scan import (
    add_dispersion_subtracted_columns,
    add_sweep_map_tunes_to_estimate,
    build_tune_estimate_from_naff,
    configure_line,
    df_to_delta,
    get_observation_dispersion,
    plot_naff_abs_tune_diagram,
    plot_naff_tune_diagram,
    plot_tune_estimate,
    plot_tune_estimate_abs_vs_sweep_map,
    plot_tune_estimate_vs_sweep_map,
)
from recompute_saved_tune_analysis import build_tune_estimate_position_only
from tune_diagram import TuneMap


DEFAULT_BATCH_DIR = (
    Path(__file__).resolve().parent
    / "dual_plane_dead_particle_diagnostics"
    / "q20_xix0p5_xiy0p5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce rf_sweep_speed_tests centroid NAFF outputs exactly from saved monitor arrays."
    )
    parser.add_argument("--batch-dir", default=str(DEFAULT_BATCH_DIR))
    parser.add_argument("--fft-window", type=int, default=256)
    parser.add_argument("--fft-step", type=int, default=64)
    parser.add_argument("--naff-harmonics", type=int, default=6)
    parser.add_argument("--output-dir-name", default="centroid_naff_exact")
    return parser.parse_args()


def load_npz_dict(path: Path) -> dict[str, object]:
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def build_summary_frame(monitor_arrays: dict[str, object], run_config: dict[str, object]) -> pd.DataFrame:
    import numpy as np

    state = np.asarray(monitor_arrays["state"]).astype(int, copy=False)
    n_turns = state.shape[1]
    rows: list[dict[str, float | int]] = []
    sweep_per_turn_hz = float(run_config["sweep_per_turn_hz"])

    for turn in range(n_turns):
        alive_mask = state[:, turn] > 0
        row: dict[str, float | int] = {
            "turn": int(turn),
            "alive_count": int(alive_mask.sum()),
            "delta_from_sweep": float(df_to_delta(sweep_per_turn_hz * turn)),
        }
        for coord in ("x", "px", "y", "py", "zeta", "delta"):
            values = np.asarray(monitor_arrays[coord])[:, turn].astype(float, copy=False)
            values = values[alive_mask]
            row[f"{coord}_mean"] = float(values.mean()) if values.size else float("nan")
            row[f"{coord}_std"] = float(values.std()) if values.size else float("nan")
        rows.append(row)

    summary_frame = pd.DataFrame(rows)
    if all(run_config.get(key) is not None for key in ("dx_monitor", "dpx_monitor", "dy_monitor", "dpy_monitor")):
        dispersion = {
            "dx": float(run_config["dx_monitor"]),
            "dpx": float(run_config["dpx_monitor"]),
            "dy": float(run_config["dy_monitor"]),
            "dpy": float(run_config["dpy_monitor"]),
        }
    else:
        line = configure_line(
            line_path=str(run_config["line_path"]),
            qx=float(run_config["qx"]),
            qy=float(run_config["qy"]),
            xi_x=float(run_config["xi_x"]),
            xi_y=float(run_config["xi_y"]),
            omp_threads="0",
            point_monitor_element=None,
            error_variant_name=str(run_config.get("error_variant", "none")),
        )
        dispersion = get_observation_dispersion(line)
    summary_frame = add_dispersion_subtracted_columns(summary_frame, dispersion)
    return summary_frame


def process_plane(
    *,
    batch_dir: Path,
    plane: str,
    fft_window: int,
    fft_step: int,
    naff_harmonics: int,
    output_dir_name: str,
) -> None:
    plane_dir = batch_dir / plane
    outdir = batch_dir / output_dir_name / plane
    outdir.mkdir(parents=True, exist_ok=True)

    monitor_arrays = load_npz_dict(plane_dir / "monitor_arrays.npz")
    run_config = json.loads((plane_dir / "run_config.json").read_text(encoding="utf-8"))
    summary_frame = build_summary_frame(monitor_arrays, run_config)

    tune_map = TuneMap.load(str(Path(run_config["tune_map_path"]).resolve()))

    tune_estimate = build_tune_estimate_from_naff(
        summary_frame,
        signal_columns={"horizontal": "x_beta_mean", "vertical": "y_beta_mean"},
        window_size=fft_window,
        step=fft_step,
        num_harmonics=naff_harmonics,
    )
    if tune_estimate is not None:
        tune_estimate = add_sweep_map_tunes_to_estimate(tune_estimate, tune_map)
        tune_estimate.to_csv(outdir / "tune_estimate.csv", index=False)
        plot_tune_estimate(tune_estimate, outdir / "tune_estimate_vs_turn.png", x_key="window_center", x_label="Turn")
        plot_tune_estimate(tune_estimate, outdir / "tune_estimate_vs_delta.png", x_key="delta_center", x_label=r"$\delta$")
        plot_tune_estimate_vs_sweep_map(tune_estimate, outdir / "tune_estimate_vs_sweep_map.png")
        plot_tune_estimate_abs_vs_sweep_map(tune_estimate, outdir / "tune_estimate_abs_vs_sweep_map.png")
        plot_naff_tune_diagram(tune_estimate, tune_map, outdir / "naff_tune_diagram.png")
        plot_naff_abs_tune_diagram(tune_estimate, tune_map, outdir / "naff_abs_tune_diagram.png")

    tune_estimate_xonly_h3 = build_tune_estimate_position_only(
        summary_frame,
        signal_columns={"horizontal": "x_beta_mean", "vertical": "y_beta_mean"},
        window_size=fft_window,
        step=fft_step,
        num_harmonics=3,
    )
    if tune_estimate_xonly_h3 is not None:
        tune_estimate_xonly_h3 = add_sweep_map_tunes_to_estimate(tune_estimate_xonly_h3, tune_map)
        tune_estimate_xonly_h3.to_csv(outdir / "tune_estimate_xonly_h3.csv", index=False)
        plot_tune_estimate(
            tune_estimate_xonly_h3,
            outdir / "tune_estimate_vs_turn_xonly_h3.png",
            x_key="window_center",
            x_label="Turn",
        )
        plot_naff_tune_diagram(tune_estimate_xonly_h3, tune_map, outdir / "naff_tune_diagram_xonly_h3.png")
        plot_naff_abs_tune_diagram(
            tune_estimate_xonly_h3,
            tune_map,
            outdir / "naff_abs_tune_diagram_xonly_h3.png",
        )

    summary_frame.to_csv(outdir / "summary_frame.csv", index=False)


def main() -> None:
    args = parse_args()
    batch_dir = Path(args.batch_dir).resolve()
    for plane in ("DPpos", "DPneg"):
        process_plane(
            batch_dir=batch_dir,
            plane=plane,
            fft_window=args.fft_window,
            fft_step=args.fft_step,
            naff_harmonics=args.naff_harmonics,
            output_dir_name=args.output_dir_name,
        )
    print(f"[saved_centroid_naff_exact] Wrote outputs to {batch_dir / args.output_dir_name}")


if __name__ == "__main__":
    main()
