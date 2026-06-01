# sps-momentum-acceptance

Code and study structure for the SPS momentum-acceptance work.

**Data** (HTCondor simulation outputs) live in the companion GitLab repo `sps-simulations`.
**Reference files** (SPS line JSON, small static data) are committed here under `data/`.

## Repository layout

```
sps-momentum-acceptance/
  config/
    paths.example.yaml     path config template — copy to paths.yaml and fill in your local paths
    paths.yaml             your local absolute paths (not committed)
  data/
    sps_with_aperture_inj_q20_beam_sagitta4.json   SPS line with aperture (from sps-xsuite-model)
    sps_q20_inj.json                               SPS line without aperture
  helper_functions/
    load_paths.py          reads config/paths.yaml
    intensity_helpers/     midpoint extraction, intensity loss, acceptance, optics helpers
    tune_diagram_helpers/  tune-scan workflow: TuneScan, PlotTuneMaps, PlotMDIntensity,
                           workflow_common, tune_diagram
  studies/
    intensity_scan2/       chromaticity scan study
    offmom_bump_scans/     off-momentum bump scan study
    tune_diagram_variations/
    plots_md_intensity/    MD intensity overlay plots
  prototyping/
    new_model_sweeps/      new fitted error model sweep maps and plots
    tests/                 exploratory scripts
  sps-chromaticity-maps/   canonical chromaticity map store (.npz) and plot outputs
  SweepTrajectoryMaps/     non-chromaticity sweep maps
  tune_scan_workflow/      legacy location — scripts migrated to helper_functions/tune_diagram_helpers/
  markdown_legacy/         archived notes
```

## Quick start

Copy the path config and fill in your local paths:
```bash
cp config/paths.example.yaml config/paths.yaml
```

The minimum required key is `sps_simulations_data_root` (path to your local clone of the
`sps-simulations` GitLab repo). Everything else has a sensible default.

## Where to find things

### Tune-diagram workflow

Canonical location: `helper_functions/tune_diagram_helpers/`

| Script | Purpose |
|--------|---------|
| `TuneScan.py` | Generate tune maps (.npz) |
| `PlotTuneMaps.py` | Replot sweep figures from saved maps |
| `PlotMDIntensity.py` | Compare MD intensity data with saved maps |
| `workflow_common.py` | Scan definitions, map roots, naming rules |
| `tune_diagram.py` | TuneDiagram and TuneMap classes |
| `generate_single_machine_tune_map.py` | Single working-point machine map |

### Intensity helpers

`helper_functions/intensity_helpers/` — midpoint extraction, intensity loss, acceptance centres, beta analysis.

### IntensityScan2

`studies/intensity_scan2/` — chromaticity scan study comparing simulation with MD data.

| Script | Purpose |
|--------|---------|
| `generate_figures.py` | **Main entry point** — regenerates all figures |
| `figures.ipynb` | Interactive version of generate_figures.py |
| `generate_intensityscan2_missing_without_errors_maps.py` | Generate sweep maps for study chroma points |
| `plot_*.py` | Individual figure scripts (called by generate_figures.py) |
| `data/midpoints_MD.json` | Real machine δ₅₀ midpoints from 2025-06-16 MD |

Simulation results (gzip files) live in `sps-simulations/momentum-acceptance/intensity_scan2/study_results/`.

### New error model

`prototyping/new_model_sweeps/` — fitted error model sweeps.
Maps committed in `maps/WithErrors/`. Scripts: `generate_sweep_map.py`,
`plot_md_intensity_single_point.py`, `plot_qx_scan.py`.

## Map and plot storage

### Chromaticity maps (committed)

`sps-chromaticity-maps/` — canonical store, organised by model case:

```
simplified/
with_errors/
without_errors/
with_errors/quadrupole_b6/
```

Map filename convention:
```
tune_map_Qx{qx:.3f}_Qy{qy:.3f}_xix{xi_x:.3f}_xiy{xi_y:.3f}.npz
```

Plot outputs under `sps-chromaticity-maps/plots/`:
```
chroma_scan_x/   chroma_scan_y/   qx_scan/   qy_scan/   single_point_machine/
```

### Non-chromaticity maps

`SweepTrajectoryMaps/` — QxScan, QyScan, and single-point maps.

## paths.yaml keys

| Key | What it controls |
|-----|-----------------|
| `sps_simulations_data_root` | Path to `sps-simulations` GitLab clone (HTCondor results) |
| `sps_chromaticity_maps_root` | Chromaticity map store (default: `sps-chromaticity-maps/`) |
| `sweep_trajectory_maps_root` | Non-chroma sweep maps (default: `SweepTrajectoryMaps/`) |
| `line_with_ap_path` | SPS line with aperture JSON (default: `data/sps_with_aperture_inj_q20_beam_sagitta4.json`) |
| `frederik_offmom_scans_root` | `sps-offmom-scans` repo — `PlotMDIntensity.py` imports helpers from there |
| `offmom_scans_results_root` | BCT measurement JSON files (`sps-offmom-scans/results/`) |
| `legacy_workspace_root` | `~/sps_simulations` fallback (only needed if line not in `data/`) |

## SPS line

`data/` contains a committed copy of the SPS line JSON from `sps-xsuite-model`.
Update it by copying from `sps-xsuite-model` whenever the model is regenerated:
```bash
cp ~/phd/code/sps-xsuite-model/sps_with_aperture_inj_q20_beam_sagitta4.json data/
cp ~/phd/code/sps-xsuite-model/sps_q20_inj.json data/
```

## Legacy notes

Older markdown notes are archived in `markdown_legacy/` (not the current source of truth).
