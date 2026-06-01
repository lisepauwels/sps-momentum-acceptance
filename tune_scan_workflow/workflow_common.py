from __future__ import annotations

from pathlib import Path
import runpy

TARGET = (
    Path(__file__).resolve().parents[1]
    / "helper_functions"
    / "tune_diagram_helpers"
    / "workflow_common.py"
)

globals().update(runpy.run_path(str(TARGET)))
