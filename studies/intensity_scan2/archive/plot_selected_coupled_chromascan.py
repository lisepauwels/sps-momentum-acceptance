from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "helper_functions" / "intensity_helpers"))
sys.path.insert(0, str(REPO_ROOT / "helper_functions" / "tune_diagram_helpers"))

from tune_diagram import TuneDiagram, TuneMap
from workflow_common import map_case_root_for_scan_type


MAP_DIR = map_case_root_for_scan_type("ChromaScanX", "WithErrors")
OUT_PATH = MAP_DIR / "tune_diagram_WithErrors_selected_coupled_xi.pdf"
QX0 = 20.13
QY0 = 20.18
XI_VALUES = [-1.5, -0.5, 0.0, 0.5, 1.5]


def map_path_for_xi(xi: float) -> Path:
    return MAP_DIR / f"tune_map_Qx{QX0:.3f}_Qy{QY0:.3f}_xix{xi:.3f}_xiy{xi:.3f}.npz"


def main() -> None:
    maps: list[tuple[float, TuneMap]] = []
    for xi in XI_VALUES:
        path = map_path_for_xi(xi)
        if not path.exists():
            raise FileNotFoundError(f"Missing map for xi={xi:.3f}: {path}")
        maps.append((xi, TuneMap.load(str(path))))

    td = TuneDiagram(
        qx0=QX0,
        qy0=QY0,
        half_range=0.4,
        max_order=3,
        skew=True,
    )
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    td.plot(ax=ax, show_working_point=False)
    ax.set_aspect("equal")
    ax.scatter(QX0, QY0, color="k", s=55, zorder=7)

    cmap = cm.get_cmap("plasma")
    norm_positions = np.linspace(0.1, 0.9, len(maps))
    xi_handles: list[mlines.Line2D] = []

    for (xi, tm), cc in zip(maps, norm_positions):
        color = cmap(cc)
        d_arr, qx_arr, qy_arr = tm.sample(400)
        ax.plot(qx_arr, qy_arr, color=color, lw=2.0, alpha=0.95)

        neg_idx = int(np.argmin(d_arr))
        pos_idx = int(np.argmax(d_arr))
        neg_arrow_idx = min(8, len(qx_arr) - 1)
        pos_arrow_idx = max(len(qx_arr) - 9, 0)
        ax.annotate(
            "",
            xy=(qx_arr[neg_idx], qy_arr[neg_idx]),
            xytext=(qx_arr[neg_arrow_idx], qy_arr[neg_arrow_idx]),
            arrowprops=dict(arrowstyle="->", color="blue", lw=1.8),
            zorder=7,
        )
        ax.annotate(
            "",
            xy=(qx_arr[pos_idx], qy_arr[pos_idx]),
            xytext=(qx_arr[pos_arrow_idx], qy_arr[pos_arrow_idx]),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.8),
            zorder=7,
        )

        xi_handles.append(
            mlines.Line2D([], [], color=color, lw=2.0, label=rf"$\xi={xi:.1f}$")
        )

    xi_handles.extend(
        [
            mpatches.FancyArrowPatch((0, 0), (1, 0), arrowstyle="->", mutation_scale=12, color="red", label="Positive sweep"),
            mpatches.FancyArrowPatch((0, 0), (1, 0), arrowstyle="->", mutation_scale=12, color="blue", label="Negative sweep"),
        ]
    )

    resonance_legend = ax.legend(handles=td.legend_handles(), loc="upper left", frameon=True)
    ax.add_artist(resonance_legend)
    ax.legend(handles=xi_handles, loc="lower right", frameon=True)
    ax.set_xlabel(r"$Q_x$")
    ax.set_ylabel(r"$Q_y$")
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
