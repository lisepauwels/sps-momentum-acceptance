from __future__ import annotations

from pathlib import Path
import runpy

TARGET = (
    Path(__file__).resolve().parents[1]
    / "studies"
    / "intensity_scan2"
    / "plot_selected_coupled_chromascan.py"
)

if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
