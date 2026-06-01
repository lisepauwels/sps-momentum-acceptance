from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_sextupoles_off_monitor_scan import (
    TuneMap,
    build_closed_orbit_x_map,
    plot_batch_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild batch-level comparison plots, including intensity vs twiss closed orbit x, "
            "from an existing sextupoles_off batch directory."
        )
    )
    parser.add_argument("batch_dir", help="Path to an existing batch directory under sextupoles_off_monitor_outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_dir = Path(args.batch_dir).resolve()
    with (batch_dir / "batch_index.json").open("r", encoding="utf-8") as fh:
        batch_index = json.load(fh)

    case_dirs = {key: Path(value) for key, value in batch_index["cases"].items()}
    sample_case = next(iter(case_dirs.values()))
    with (sample_case / "run_config.json").open("r", encoding="utf-8") as fh:
        run_config = json.load(fh)

    tune_map_path = Path(run_config["tune_map_path"]) if run_config.get("tune_map_path") else None
    tune_map = TuneMap.load(str(tune_map_path)) if tune_map_path is not None and tune_map_path.exists() else None

    closed_orbit_delta_grid = None
    closed_orbit_x_grid = None
    if tune_map is not None:
        closed_orbit_delta_grid, closed_orbit_x_grid = build_closed_orbit_x_map(
            line_path=str(run_config["line_path"]),
            qx=float(run_config["qx"]),
            qy=float(run_config["qy"]),
            error_variant_name=str(run_config["error_variant"]),
            element_name=str(run_config["monitor_element"]),
            delta_range=(tune_map.delta_min, tune_map.delta_max),
        )

    plot_batch_comparison(
        batch_dir,
        case_dirs,
        tune_map,
        closed_orbit_delta_grid=closed_orbit_delta_grid,
        closed_orbit_x_grid=closed_orbit_x_grid,
        closed_orbit_element=str(run_config["monitor_element"]),
    )


if __name__ == "__main__":
    main()
