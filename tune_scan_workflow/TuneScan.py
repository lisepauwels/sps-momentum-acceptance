from __future__ import annotations

from pathlib import Path
import runpy

TARGET = (
    Path(__file__).resolve().parents[1]
    / "helper_functions"
    / "tune_diagram_helpers"
    / "TuneScan.py"
)

if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
