from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "helper_functions"))

from load_paths import get_path

MAP_CASES = ["WithErrors", "WithoutErrors", "Simplified"]
HALF_RANGE = 0.4
MAX_ORDER = 3
OUTPUT_ROOT = get_path(
    "sweep_trajectory_maps_root",
    default=str(REPO_ROOT / "SweepTrajectoryMaps"),
)
CHROMA_MAP_ROOT = get_path(
    "sps_chromaticity_maps_root",
    default=str(REPO_ROOT / "sps-chromaticity-maps"),
)
CASE_DIRECTORY_MAP = {
    "WithErrors": "with_errors",
    "WithoutErrors": "without_errors",
    "Simplified": "simplified",
    "WithErrors_quadrupole_b6": "with_errors/quadrupole_b6",
}

MACHINE_XI_X_OFFSET = 0.005
MACHINE_XI_Y_OFFSET = -0.2


QX_SCAN = {
    "label": "QxScan",
    "type": "QxScan",
    "tunes": [
        20.07, 20.075, 20.08, 20.085, 20.09, 20.095,
        20.10, 20.105, 20.11, 20.115, 20.12, 20.125,
        20.13, 20.135, 20.14, 20.145, 20.15, 20.155,
        20.16, 20.165, 20.17, 20.175,
    ],
    "fixed_qy": 20.18,
    "xi_x": 0.5,
    "xi_y": 0.5,
    "include_chroma_in_filename": True,
    "plot_suffix": "xix0.500_xiy0.500",
}

QY_SCAN = {
    "label": "QyScan",
    "type": "QyScan",
    "tunes": [
        20.125, 20.13, 20.135, 20.14, 20.145, 20.15,
        20.155, 20.16, 20.165, 20.17, 20.175, 20.18,
        20.185, 20.19, 20.195, 20.20, 20.205, 20.21,
        20.215, 20.22, 20.225, 20.23, 20.235,
    ],
    "fixed_qx": 20.13,
    "xi_x": 0.5,
    "xi_y": 0.5,
    "include_chroma_in_filename": True,
    "plot_suffix": "xix0.500_xiy0.500",
}

_QX_CHROMA = 20.13
_QY_CHROMA = 20.18

CHROMA_SCAN_Y = {
    "label": "ChromaScanY",
    "type": "ChromaScanY",
    "xi_pairs": [(0.5, xi_y) for xi_y in [round(0.2 + i * 0.05, 2) for i in range(17)] + [1.15]],
    "fixed_qx": _QX_CHROMA,
    "fixed_qy": _QY_CHROMA,
}

CHROMA_SCAN_X = {
    "label": "ChromaScanX",
    "type": "ChromaScanX",
    "xi_pairs": [(xi_x, 0.5) for xi_x in [0.6, 0.7, 0.8]],
    "fixed_qx": _QX_CHROMA,
    "fixed_qy": _QY_CHROMA,
}


def _clone_scan(scan_cfg: dict, **updates) -> dict:
    cloned = deepcopy(scan_cfg)
    cloned.update(updates)
    return cloned


QX_SCAN_MACHINE = _clone_scan(
    QX_SCAN,
    xi_x=0.505,
    xi_y=0.300,
    plot_suffix="xix0.505_xiy0.300",
)

QY_SCAN_MACHINE = _clone_scan(
    QY_SCAN,
    xi_x=0.505,
    xi_y=0.300,
    plot_suffix="xix0.505_xiy0.300",
)

CHROMA_SCAN_Y_MACHINE = _clone_scan(
    CHROMA_SCAN_Y,
    xi_x_offset=MACHINE_XI_X_OFFSET,
    xi_y_offset=MACHINE_XI_Y_OFFSET,
    plot_suffix="machine_offset",
)

CHROMA_SCAN_X_MACHINE = _clone_scan(
    CHROMA_SCAN_X,
    xi_x_offset=MACHINE_XI_X_OFFSET,
    xi_y_offset=MACHINE_XI_Y_OFFSET,
    plot_suffix="machine_offset",
)

MACHINE_SCANS = [
    QX_SCAN_MACHINE,
    QY_SCAN_MACHINE,
    CHROMA_SCAN_Y_MACHINE,
    CHROMA_SCAN_X_MACHINE,
]


def format_chroma_suffix(xi_x: float, xi_y: float) -> str:
    return f"xix{xi_x:.3f}_xiy{xi_y:.3f}"


def tune_map_filename(
    qx: float,
    qy: float,
    xi_x: float | None = None,
    xi_y: float | None = None,
) -> str:
    base = f"tune_map_Qx{qx:.3f}_Qy{qy:.3f}"
    if xi_x is not None and xi_y is not None:
        return f"{base}_xix{xi_x:.3f}_xiy{xi_y:.3f}.npz"
    return f"{base}.npz"


def is_tune_scan(scan_cfg: dict) -> bool:
    return scan_cfg["type"] in ("QxScan", "QyScan")


def is_chroma_scan(scan_cfg: dict) -> bool:
    return scan_cfg["type"] in ("ChromaScanX", "ChromaScanY")


def is_chroma_scan_type(scan_type: str) -> bool:
    return scan_type in ("ChromaScanX", "ChromaScanY")


def canonical_case_directory(case_name: str) -> Path:
    return Path(CASE_DIRECTORY_MAP.get(case_name, case_name))


def map_case_root_for_scan_type(scan_type: str, case_name: str) -> Path:
    if is_chroma_scan_type(scan_type):
        return CHROMA_MAP_ROOT / canonical_case_directory(case_name)
    return OUTPUT_ROOT / scan_type / case_name


def plot_bucket_for_scan_type(scan_type: str) -> Path:
    bucket_map = {
        "QxScan": "qx_scan",
        "QyScan": "qy_scan",
        "ChromaScanX": "chroma_scan_x",
        "ChromaScanY": "chroma_scan_y",
        "SinglePoint": "single_point_machine",
        "SinglePointMachine": "single_point_machine",
    }
    return Path(bucket_map.get(scan_type, scan_type))


def plot_case_root_for_scan_type(scan_type: str, case_name: str) -> Path:
    return CHROMA_MAP_ROOT / "plots" / plot_bucket_for_scan_type(scan_type) / canonical_case_directory(case_name)


def scan_key_to_working_point(scan_cfg: dict, key) -> tuple[float, float]:
    scan_type = scan_cfg["type"]
    if scan_type == "QxScan":
        return key, scan_cfg["fixed_qy"]
    if scan_type == "QyScan":
        return scan_cfg["fixed_qx"], key
    if scan_type in ("ChromaScanX", "ChromaScanY"):
        return scan_cfg["fixed_qx"], scan_cfg["fixed_qy"]
    raise ValueError(f"Unsupported scan type: {scan_type}")


def scan_key_to_map_chroma(scan_cfg: dict, key) -> tuple[float | None, float | None]:
    scan_type = scan_cfg["type"]
    if scan_type in ("QxScan", "QyScan"):
        return scan_cfg["xi_x"], scan_cfg["xi_y"]
    if scan_type in ("ChromaScanX", "ChromaScanY"):
        xi_x, xi_y = key
        return (
            xi_x + scan_cfg.get("xi_x_offset", 0.0),
            xi_y + scan_cfg.get("xi_y_offset", 0.0),
        )
    raise ValueError(f"Unsupported scan type: {scan_type}")


def scan_key_to_measurement_chroma(scan_cfg: dict, key) -> tuple[float | None, float | None]:
    scan_type = scan_cfg["type"]
    if scan_type in ("QxScan", "QyScan"):
        return None, None
    if scan_type in ("ChromaScanX", "ChromaScanY"):
        return key
    raise ValueError(f"Unsupported scan type: {scan_type}")


def scan_key_to_chroma(scan_cfg: dict, key) -> tuple[float | None, float | None]:
    return scan_key_to_map_chroma(scan_cfg, key)


def scan_key_to_filename_chroma(scan_cfg: dict, key) -> tuple[float | None, float | None]:
    xi_x, xi_y = scan_key_to_map_chroma(scan_cfg, key)
    if is_tune_scan(scan_cfg) and not scan_cfg.get("include_chroma_in_filename", False):
        return None, None
    return xi_x, xi_y


def scan_param_value(scan_cfg: dict, key):
    scan_type = scan_cfg["type"]
    if scan_type in ("QxScan", "QyScan"):
        return key
    if scan_type == "ChromaScanY":
        return scan_key_to_map_chroma(scan_cfg, key)[1]
    if scan_type == "ChromaScanX":
        return scan_key_to_map_chroma(scan_cfg, key)[0]
    raise ValueError(f"Unsupported scan type: {scan_type}")


def scan_key_label(scan_cfg: dict, key) -> str:
    scan_type = scan_cfg["type"]
    if scan_type in ("QxScan", "QyScan"):
        qx, qy = scan_key_to_working_point(scan_cfg, key)
        return f"Qx={qx:.3f} Qy={qy:.2f}"
    if scan_type in ("ChromaScanX", "ChromaScanY"):
        xi_x, xi_y = scan_key_to_map_chroma(scan_cfg, key)
        return f"xi_x={xi_x:.3f} xi_y={xi_y:.3f}"
    raise ValueError(f"Unsupported scan type: {scan_type}")


def iter_scan_entries(scan_cfg: dict):
    scan_type = scan_cfg["type"]
    if scan_type == "QxScan":
        for qx in scan_cfg["tunes"]:
            xi_x, xi_y = scan_key_to_filename_chroma(scan_cfg, qx)
            yield qx, qx, scan_cfg["fixed_qy"], xi_x, xi_y
    elif scan_type == "QyScan":
        for qy in scan_cfg["tunes"]:
            xi_x, xi_y = scan_key_to_filename_chroma(scan_cfg, qy)
            yield qy, scan_cfg["fixed_qx"], qy, xi_x, xi_y
    elif scan_type in ("ChromaScanX", "ChromaScanY"):
        qx = scan_cfg["fixed_qx"]
        qy = scan_cfg["fixed_qy"]
        for key in scan_cfg["xi_pairs"]:
            xi_x, xi_y = scan_key_to_filename_chroma(scan_cfg, key)
            yield key, qx, qy, xi_x, xi_y
    else:
        raise ValueError(f"Unsupported scan type: {scan_type}")


def scan_keys_and_labels(scan_cfg: dict):
    scan_type = scan_cfg["type"]
    if scan_type == "QxScan":
        keys = scan_cfg["tunes"]
        return keys, keys, r"$Q_x$ (working point)"
    if scan_type == "QyScan":
        keys = scan_cfg["tunes"]
        return keys, keys, r"$Q_y$ (working point)"
    if scan_type == "ChromaScanY":
        keys = scan_cfg["xi_pairs"]
        return keys, [scan_key_to_map_chroma(scan_cfg, p)[1] for p in keys], r"$\xi_y$ (norm. chromaticity)"
    if scan_type == "ChromaScanX":
        keys = scan_cfg["xi_pairs"]
        return keys, [scan_key_to_map_chroma(scan_cfg, p)[0] for p in keys], r"$\xi_x$ (norm. chromaticity)"
    raise ValueError(f"Unsupported scan type: {scan_type}")


def scan_plot_suffix(scan_cfg: dict) -> str | None:
    if "plot_suffix" in scan_cfg:
        return scan_cfg["plot_suffix"]
    if is_tune_scan(scan_cfg) and scan_cfg.get("include_chroma_in_filename", False):
        return format_chroma_suffix(scan_cfg["xi_x"], scan_cfg["xi_y"])
    return None


def tune_diagram_spec(scan_cfg: dict):
    keys, scan_vals, cbar_label = scan_keys_and_labels(scan_cfg)
    scan_type = scan_cfg["type"]
    if scan_type == "QxScan":
        qx_centre = 0.5 * (scan_vals[0] + scan_vals[-1])
        qx_half = 0.5 * (scan_vals[-1] - scan_vals[0]) + HALF_RANGE
        return {
            "qx0": qx_centre,
            "qy0": scan_cfg["fixed_qy"],
            "half_range": (qx_half, HALF_RANGE),
            "scan_vals": scan_vals,
            "cbar_label": cbar_label,
        }
    if scan_type == "QyScan":
        qy_centre = 0.5 * (scan_vals[0] + scan_vals[-1])
        qy_half = 0.5 * (scan_vals[-1] - scan_vals[0]) + HALF_RANGE
        return {
            "qx0": scan_cfg["fixed_qx"],
            "qy0": qy_centre,
            "half_range": (HALF_RANGE, qy_half),
            "scan_vals": scan_vals,
            "cbar_label": cbar_label,
        }
    if scan_type in ("ChromaScanX", "ChromaScanY"):
        return {
            "qx0": scan_cfg["fixed_qx"],
            "qy0": scan_cfg["fixed_qy"],
            "half_range": HALF_RANGE,
            "scan_vals": scan_vals,
            "cbar_label": cbar_label,
        }
    raise ValueError(f"Unsupported scan type: {scan_type}")


def colorbar_inset_positions(n_cbars: int, x0: float = 1.04,
                             width: float = 0.03, gap: float = 0.08):
    return [[x0 + i * gap, 0.0, width, 1.0] for i in range(n_cbars)]


def finalize_colorbar_heights(fig, ax, caxes) -> None:
    """
    Resize colorbar axes to exactly match the height of the main axes.

    Must be called after all drawing is done.  Turns off constrained_layout
    so the corrected positions are not overwritten by savefig.
    """
    fig.canvas.draw()
    ax_pos = ax.get_position()
    for cax in caxes:
        p = cax.get_position()
        cax.set_position([p.x0, ax_pos.y0, p.width, ax_pos.height])
    fig.set_layout_engine(None)


def make_tune_diagram_figure(n_cbars: int = 1, figsize=(10, 9)):
    """
    Create a figure with a main axes + n dedicated colorbar axes using GridSpec.
    Each colorbar gets its own column so constrained_layout can properly
    allocate space for labels without any spillover.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    width_ratios = [20] + [1] * n_cbars
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = GridSpec(1, 1 + n_cbars, figure=fig,
                  width_ratios=width_ratios, wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    caxes = [fig.add_subplot(gs[0, i + 1]) for i in range(n_cbars)]
    return fig, ax, caxes
