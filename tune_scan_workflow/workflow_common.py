from __future__ import annotations


MAP_CASES = ["WithErrors", "WithoutErrors", "Simplified"]
HALF_RANGE = 0.4
MAX_ORDER = 3
OUTPUT_ROOT = "SweepTrajectoryMaps"


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


def scan_key_to_working_point(scan_cfg: dict, key) -> tuple[float, float]:
    scan_type = scan_cfg["type"]
    if scan_type == "QxScan":
        return key, scan_cfg["fixed_qy"]
    if scan_type == "QyScan":
        return scan_cfg["fixed_qx"], key
    if scan_type in ("ChromaScanX", "ChromaScanY"):
        return scan_cfg["fixed_qx"], scan_cfg["fixed_qy"]
    raise ValueError(f"Unsupported scan type: {scan_type}")


def scan_key_to_chroma(scan_cfg: dict, key) -> tuple[float | None, float | None]:
    scan_type = scan_cfg["type"]
    if scan_type in ("QxScan", "QyScan"):
        return None, None
    if scan_type in ("ChromaScanX", "ChromaScanY"):
        return key
    raise ValueError(f"Unsupported scan type: {scan_type}")


def scan_param_value(scan_cfg: dict, key):
    scan_type = scan_cfg["type"]
    if scan_type in ("QxScan", "QyScan"):
        return key
    if scan_type == "ChromaScanY":
        return key[1]
    if scan_type == "ChromaScanX":
        return key[0]
    raise ValueError(f"Unsupported scan type: {scan_type}")


def scan_key_label(scan_cfg: dict, key) -> str:
    scan_type = scan_cfg["type"]
    if scan_type in ("QxScan", "QyScan"):
        qx, qy = scan_key_to_working_point(scan_cfg, key)
        return f"Qx={qx:.3f} Qy={qy:.2f}"
    if scan_type in ("ChromaScanX", "ChromaScanY"):
        xi_x, xi_y = key
        return f"xi_x={xi_x:.3f} xi_y={xi_y:.3f}"
    raise ValueError(f"Unsupported scan type: {scan_type}")


def iter_scan_entries(scan_cfg: dict):
    scan_type = scan_cfg["type"]
    if scan_type == "QxScan":
        for qx in scan_cfg["tunes"]:
            yield qx, qx, scan_cfg["fixed_qy"], None, None
    elif scan_type == "QyScan":
        for qy in scan_cfg["tunes"]:
            yield qy, scan_cfg["fixed_qx"], qy, None, None
    elif scan_type in ("ChromaScanX", "ChromaScanY"):
        qx = scan_cfg["fixed_qx"]
        qy = scan_cfg["fixed_qy"]
        for xi_x, xi_y in scan_cfg["xi_pairs"]:
            yield (xi_x, xi_y), qx, qy, xi_x, xi_y
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
        return keys, [p[1] for p in keys], r"$\xi_y$ (norm. chromaticity)"
    if scan_type == "ChromaScanX":
        keys = scan_cfg["xi_pairs"]
        return keys, [p[0] for p in keys], r"$\xi_x$ (norm. chromaticity)"
    raise ValueError(f"Unsupported scan type: {scan_type}")


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
                             width: float = 0.03, gap: float = 0.13):
    return [[x0 + i * gap, 0.0, width, 1.0] for i in range(n_cbars)]
