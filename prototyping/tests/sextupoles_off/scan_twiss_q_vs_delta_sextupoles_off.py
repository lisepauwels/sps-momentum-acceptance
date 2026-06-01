from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xtrack as xt

from run_sextupoles_off_monitor_scan import DEFAULT_LINE_PATH, apply_error_configuration, zero_sextupoles


HERE = Path(__file__).resolve().parent


@dataclass
class ScanConfig:
    line_path: str
    qx: float
    qy: float
    error_variant: str
    delta_min: float
    delta_max: float
    num_points: int
    output_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a sextupoles-off twiss4d scan and plot Qx, Qy as a function of delta."
    )
    parser.add_argument("--line-path", default=DEFAULT_LINE_PATH)
    parser.add_argument("--qx", type=float, default=20.13)
    parser.add_argument("--qy", type=float, default=20.18)
    parser.add_argument("--error-variant", default="none")
    parser.add_argument("--delta-min", type=float, default=-1.0e-2)
    parser.add_argument("--delta-max", type=float, default=1.0e-2)
    parser.add_argument("--num-points", type=int, default=201)
    parser.add_argument("--output-base", default="twiss_delta_scans")
    parser.add_argument(
        "--batch-name",
        default=f"sextupoles_off_twiss_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    return parser.parse_args()


def configure_line(
    *,
    line_path: str,
    qx: float,
    qy: float,
    error_variant: str,
) -> xt.Line:
    line = xt.load(line_path)

    _, cavity_names = line.get_elements_of_type(xt.Cavity)
    for name in cavity_names:
        line[name].frequency = 200e6
        line[name].lag = 180
        line[name].voltage = 0
    line["actcse.31632"].voltage = 3.0e6

    apply_error_configuration(line, error_variant_name=error_variant)
    zero_sextupoles(line)
    line.match(
        method="6d",
        vary=[xt.VaryList(["kqf0", "kqd0"], step=1e-8, tag="quad")],
        targets=[xt.TargetSet(qx=qx, qy=qy, tol=1e-6, tag="tune")],
    )
    return line


def build_scan_frame(
    line: xt.Line,
    *,
    delta_min: float,
    delta_max: float,
    num_points: int,
) -> pd.DataFrame:
    deltas = np.linspace(delta_min, delta_max, num_points)
    qx_values = np.full_like(deltas, np.nan, dtype=float)
    qy_values = np.full_like(deltas, np.nan, dtype=float)

    index_zero = int(np.argmin(np.abs(deltas)))
    tw0 = line.twiss4d(delta0=0.0)
    qx_values[index_zero] = float(tw0.qx)
    qy_values[index_zero] = float(tw0.qy)

    co_prev = tw0.particle_on_co
    for ii in range(index_zero + 1, len(deltas)):
        tw = line.twiss4d(delta0=float(deltas[ii]), co_guess=co_prev)
        qx_values[ii] = float(tw.qx)
        qy_values[ii] = float(tw.qy)
        co_prev = tw.particle_on_co

    co_prev = tw0.particle_on_co
    for ii in range(index_zero - 1, -1, -1):
        tw = line.twiss4d(delta0=float(deltas[ii]), co_guess=co_prev)
        qx_values[ii] = float(tw.qx)
        qy_values[ii] = float(tw.qy)
        co_prev = tw.particle_on_co

    frame = pd.DataFrame({"delta": deltas, "qx": qx_values, "qy": qy_values})
    frame["dqx"] = frame["qx"] - float(tw0.qx)
    frame["dqy"] = frame["qy"] - float(tw0.qy)
    return frame


def plot_scan(frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True, constrained_layout=True)

    axes[0].plot(frame["delta"], frame["qx"], linewidth=1.8, label=r"$Q_x$")
    axes[0].plot(frame["delta"], frame["qy"], linewidth=1.8, label=r"$Q_y$")
    axes[0].set_ylabel("Tune")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(frame["delta"], frame["dqx"], linewidth=1.8, label=r"$\Delta Q_x$")
    axes[1].plot(frame["delta"], frame["dqy"], linewidth=1.8, label=r"$\Delta Q_y$")
    axes[1].set_xlabel(r"$\delta$")
    axes[1].set_ylabel(r"$\Delta Q$")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    fig.suptitle("Sextupoles-off twiss4d scan")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = HERE / args.output_base / args.batch_name
    outdir.mkdir(parents=True, exist_ok=False)

    config = ScanConfig(
        line_path=args.line_path,
        qx=args.qx,
        qy=args.qy,
        error_variant=args.error_variant,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        num_points=args.num_points,
        output_dir=str(outdir),
    )
    with (outdir / "scan_config.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(config), fh, indent=2)

    line = configure_line(
        line_path=args.line_path,
        qx=args.qx,
        qy=args.qy,
        error_variant=args.error_variant,
    )
    frame = build_scan_frame(
        line,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        num_points=args.num_points,
    )
    frame.to_parquet(outdir / "twiss_q_vs_delta.parquet", index=False)
    plot_scan(frame, outdir / "twiss_q_vs_delta.png")
    print(f"[scan_twiss_q_vs_delta_sextupoles_off] Wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
