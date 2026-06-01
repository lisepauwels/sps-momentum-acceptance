from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rf_sweep_speed_scan import (
    add_sweep_map_tunes_to_estimate,
    build_tune_estimate_from_naff,
    load_tune_map_case,
    plot_naff_abs_tune_diagram,
    plot_naff_harmonics_positive_vs_sweep_map,
    plot_naff_tracks,
    plot_naff_tune_diagram,
    plot_spectrogram,
    plot_tune_estimate_abs_vs_sweep_map,
    plot_tune_estimate,
    plot_tune_estimate_vs_sweep_map,
    save_sliding_naff,
    sliding_naff,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute FFT/NAFF/tune-estimate outputs from a saved run directory."
    )
    parser.add_argument("run_dir", nargs="+", help="One or more saved run directories.")
    parser.add_argument("--fft-window", type=int, required=True)
    parser.add_argument("--fft-step", type=int, required=True)
    parser.add_argument("--naff-harmonics", type=int, default=None)
    return parser.parse_args()


def build_tune_estimate_position_only(
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
            None,
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
    delta = summary_frame["delta_from_sweep"].to_numpy(dtype=float)
    merged["delta_center"] = np.interp(
        merged["window_center"].to_numpy(dtype=float),
        turn,
        delta,
    )
    return merged


def recompute_one(run_dir: Path, fft_window: int, fft_step: int, naff_harmonics: int | None) -> None:
    run_dir = run_dir.resolve()
    with (run_dir / "run_config.json").open("r", encoding="utf-8") as fh:
        run_config = json.load(fh)

    if naff_harmonics is None:
        naff_harmonics = int(run_config["naff_harmonics"])
    error_variant = str(run_config.get("error_variant", "none"))
    tune_map_case = run_config.get("tune_map_case")
    if tune_map_case is None:
        tune_map_case = "WithErrors" if error_variant != "none" else "WithoutErrors"
    tune_map = load_tune_map_case(
        float(run_config["qx"]),
        float(run_config["qy"]),
        float(run_config["xi_x"]),
        float(run_config["xi_y"]),
        str(tune_map_case),
    )

    summary_frame = pd.read_parquet(run_dir / "turn_summary.parquet")
    summary = {column: summary_frame[column].to_numpy() for column in summary_frame.columns}
    beta_summary = {
        "x_mean": summary_frame["x_beta_mean"].to_numpy(),
        "px_mean": summary_frame["px_beta_mean"].to_numpy(),
        "y_mean": summary_frame["y_beta_mean"].to_numpy(),
        "py_mean": summary_frame["py_beta_mean"].to_numpy(),
    }

    plot_spectrogram(summary, run_dir / "centroid_spectrogram.png", window_size=fft_window, step=fft_step)
    plot_spectrogram(beta_summary, run_dir / "centroid_spectrogram_beta.png", window_size=fft_window, step=fft_step)

    save_sliding_naff(
        summary,
        run_dir / "sliding_naff_global",
        window_size=fft_window,
        step=fft_step,
        num_harmonics=naff_harmonics,
    )
    save_sliding_naff(
        beta_summary,
        run_dir / "sliding_naff_beta",
        window_size=fft_window,
        step=fft_step,
        num_harmonics=naff_harmonics,
    )
    horizontal_beta = pd.read_parquet(run_dir / "sliding_naff_beta_horizontal.parquet")
    vertical_beta = pd.read_parquet(run_dir / "sliding_naff_beta_vertical.parquet")

    plot_naff_tracks(
        summary,
        run_dir / "sliding_naff_global.png",
        window_size=fft_window,
        step=fft_step,
        num_harmonics=naff_harmonics,
    )
    plot_naff_tracks(
        beta_summary,
        run_dir / "sliding_naff_beta.png",
        window_size=fft_window,
        step=fft_step,
        num_harmonics=naff_harmonics,
    )
    plot_naff_harmonics_positive_vs_sweep_map(
        horizontal_beta,
        vertical_beta,
        summary_frame,
        tune_map,
        run_dir / "naff_harmonics_positive_vs_sweep_map.png",
    )

    # Additional position-only NAFF variant: use x/y only, no px/py, with 3 harmonics.
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
        run_dir / "sliding_naff_global_xonly_h3",
        window_size=fft_window,
        step=fft_step,
        num_harmonics=xonly_harmonics,
    )
    save_sliding_naff(
        beta_position_only,
        run_dir / "sliding_naff_beta_xonly_h3",
        window_size=fft_window,
        step=fft_step,
        num_harmonics=xonly_harmonics,
    )
    horizontal_beta_xonly = pd.read_parquet(run_dir / "sliding_naff_beta_xonly_h3_horizontal.parquet")
    vertical_beta_xonly = pd.read_parquet(run_dir / "sliding_naff_beta_xonly_h3_vertical.parquet")
    plot_naff_tracks(
        global_position_only,
        run_dir / "sliding_naff_global_xonly_h3.png",
        window_size=fft_window,
        step=fft_step,
        num_harmonics=xonly_harmonics,
    )
    plot_naff_tracks(
        beta_position_only,
        run_dir / "sliding_naff_beta_xonly_h3.png",
        window_size=fft_window,
        step=fft_step,
        num_harmonics=xonly_harmonics,
    )
    plot_naff_harmonics_positive_vs_sweep_map(
        horizontal_beta_xonly,
        vertical_beta_xonly,
        summary_frame,
        tune_map,
        run_dir / "naff_harmonics_positive_vs_sweep_map_xonly_h3.png",
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
        tune_estimate.to_parquet(run_dir / "tune_estimate.parquet", index=False)
        plot_tune_estimate(
            tune_estimate,
            run_dir / "tune_estimate_vs_turn.png",
            x_key="window_center",
            x_label="Turn",
        )
        plot_tune_estimate(
            tune_estimate,
            run_dir / "tune_estimate_vs_delta.png",
            x_key="delta_center",
            x_label=r"$\delta$",
        )
        plot_tune_estimate_vs_sweep_map(
            tune_estimate,
            run_dir / "tune_estimate_vs_sweep_map.png",
        )
        plot_tune_estimate_abs_vs_sweep_map(
            tune_estimate,
            run_dir / "tune_estimate_abs_vs_sweep_map.png",
        )
        plot_naff_tune_diagram(
            tune_estimate,
            tune_map,
            run_dir / "naff_tune_diagram.png",
        )
        plot_naff_abs_tune_diagram(
            tune_estimate,
            tune_map,
            run_dir / "naff_abs_tune_diagram.png",
        )

    tune_estimate_xonly_h3 = build_tune_estimate_position_only(
        summary_frame,
        signal_columns={"horizontal": "x_beta_mean", "vertical": "y_beta_mean"},
        window_size=fft_window,
        step=fft_step,
        num_harmonics=xonly_harmonics,
    )
    if tune_estimate_xonly_h3 is not None:
        tune_estimate_xonly_h3 = add_sweep_map_tunes_to_estimate(tune_estimate_xonly_h3, tune_map)
        tune_estimate_xonly_h3.to_parquet(run_dir / "tune_estimate_xonly_h3.parquet", index=False)
        plot_tune_estimate(
            tune_estimate_xonly_h3,
            run_dir / "tune_estimate_vs_turn_xonly_h3.png",
            x_key="window_center",
            x_label="Turn",
        )
        plot_tune_estimate(
            tune_estimate_xonly_h3,
            run_dir / "tune_estimate_vs_delta_xonly_h3.png",
            x_key="delta_center",
            x_label=r"$\delta$",
        )
        plot_tune_estimate_vs_sweep_map(
            tune_estimate_xonly_h3,
            run_dir / "tune_estimate_vs_sweep_map_xonly_h3.png",
        )
        plot_tune_estimate_abs_vs_sweep_map(
            tune_estimate_xonly_h3,
            run_dir / "tune_estimate_abs_vs_sweep_map_xonly_h3.png",
        )

    run_config["fft_window"] = fft_window
    run_config["fft_step"] = fft_step
    run_config["naff_harmonics"] = naff_harmonics
    run_config["tune_map_case"] = tune_map_case
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2)

    print(f"[recompute_saved_tune_analysis] Updated {run_dir}")


def main() -> None:
    args = parse_args()
    for run_dir_str in args.run_dir:
        recompute_one(
            Path(run_dir_str),
            fft_window=args.fft_window,
            fft_step=args.fft_step,
            naff_harmonics=args.naff_harmonics,
        )


if __name__ == "__main__":
    main()
