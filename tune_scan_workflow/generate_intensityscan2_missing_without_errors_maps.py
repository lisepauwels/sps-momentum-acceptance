from __future__ import annotations

from pathlib import Path
import runpy

TARGET = (
    Path(__file__).resolve().parents[1]
    / "studies"
    / "intensity_scan2"
    / "generate_intensityscan2_missing_without_errors_maps.py"
)

if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
