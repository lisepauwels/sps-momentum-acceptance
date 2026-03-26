# Developer Reference

This document explains the internal structure of the workflow and the role of the main functions.

The three main scripts are:

- `TuneScan.py`
- `PlotTuneMaps.py`
- `PlotMDIntensity.py`

There is now also one shared support module:

- `workflow_common.py`

For a run-oriented overview, see [USER_GUIDE.md](/Users/lisepauwels/sps_simulations/Prototyping/20260317/USER_GUIDE.md).

## Core Design

The workflow has three stages:

```text
TuneScan.py
  -> writes TuneMap .npz files

PlotTuneMaps.py
  -> reads TuneMap .npz files only

PlotMDIntensity.py
  -> reads TuneMap .npz files
  -> reads MD measurement files
  -> combines both
```

This means the save/load contract must remain consistent across all three scripts.

The shared contract now lives primarily in `workflow_common.py`.

## Scan Model

The workflow supports four scan types:

- `QxScan`
- `QyScan`
- `ChromaScanX`
- `ChromaScanY`

### Key rule: scan-point key type

- tune scans use scalar keys
- chroma scans use tuple keys `(xi_x, xi_y)`

Any helper dealing with labels, filenames, colors, or ordering must respect this.

### Key rule: tune-diagram centering

For tune scans:

- center on the scanned tune range and the fixed opposite tune

For chroma scans:

- center on the fixed `(Qx, Qy)` working point

### Key rule: color meaning

The working-point color scale must represent only the scanned quantity:

- `QxScan` -> `Qx`
- `QyScan` -> `Qy`
- `ChromaScanX` -> `xi_x`
- `ChromaScanY` -> `xi_y`

## File and Folder Contract

### Output folders

```text
SweepTrajectoryMaps/
  <ScanType>/
    WithErrors/
    WithoutErrors/
    Simplified/
```

### Tune scan filename

```text
tune_map_Qx{qx:.3f}_Qy{qy:.3f}.npz
```

### Chroma scan filename

```text
tune_map_Qx{qx:.3f}_Qy{qy:.3f}_xix{xi_x:.3f}_xiy{xi_y:.3f}.npz
```

### Measurement folders expected by `PlotMDIntensity.py`

```text
results/TUNE_<qx>_<qy>_NEG/
results/TUNE_<qx>_<qy>_POS/
results/CHROM_<xi_x>_<xi_y>_NEG/
results/CHROM_<xi_x>_<xi_y>_POS/
```

## Shared Module: `workflow_common.py`

Purpose:

- hold shared scan definitions
- hold shared constants used by multiple scripts
- centralize scan-key and filename helpers

Important contents:

- `MAP_CASES`
- `HALF_RANGE`
- `MAX_ORDER`
- `OUTPUT_ROOT`
- `QX_SCAN`
- `QY_SCAN`
- `CHROMA_SCAN_X`
- `CHROMA_SCAN_Y`
- `tune_map_filename(...)`
- `scan_key_to_working_point(...)`
- `scan_key_to_chroma(...)`
- `scan_param_value(...)`
- `iter_scan_entries(...)`
- `scan_keys_and_labels(...)`
- `tune_diagram_spec(...)`
- `colorbar_inset_positions(...)`

## `TuneScan.py`

Purpose:

- produce simulation maps
- save them to disk
- optionally plot the summary tune diagram

### Important code locations

- local configuration and active scan selection
- error installation
- tune/chroma matching
- per-point map builder
- script-specific plotting
- tune-scan and chroma-scan drivers
- top-level entry point

### Function summary

#### `install_errors(line, error_variant_name)`

- applies the selected magnet error pattern to the line

#### `optimise_tune_chroma(line, xi_x, xi_y, qx, qy)`

- matches quadrupole and sextupole knobs to target tune and chromaticity

#### `_setup_cavities(line)`

- applies the cavity settings used by the workflow

#### `_tune_map_filename(qx, qy, xi_x=None, xi_y=None)`

- builds the `.npz` filename

#### `_case_dir(scan_type, case_name)`

- returns the output folder for one scan/case pair

#### `_setup_dirs(scan_type)`

- creates the required output folders for all cases

#### `_build_and_save(...)`

- builds one `TuneMap`
- saves it to disk
- returns the created `TuneMap`

Important internal behavior:

- installs errors only for `WithErrors`
- uses `SweepTrajectory.find_delta_limit(...)`
- uses either `SweepTrajectory.from_twiss_scan(...)` or `SweepTrajectory.from_chroma(...)`

#### `plot_tune_diagram(maps, scan_cfg, case_name, n_sample=300)`

- plots one summary tune diagram for a full scan/case

#### `run_tune_scan(scan_cfg, tt_aper)`

- drives either a `QxScan` or a `QyScan`

#### `run_chroma_scan(scan_cfg, tt_aper)`

- drives either a `ChromaScanX` or a `ChromaScanY`

#### `main()`

- dispatches active scan configs to the correct runner

## `PlotTuneMaps.py`

Purpose:

- reload saved maps from disk
- recreate the summary simulation tune diagram

### Important code locations

- active scan selection
- local path helpers
- map loading
- tune-diagram plotting
- top-level entry point

### Function summary

#### `_tune_map_filename(qx, qy, xi_x=None, xi_y=None)`

- reproduces the save-side naming convention

#### `_case_dir(scan_type, case_name)`

- returns the map folder

#### `_scan_output_pdf(scan_type, case_name)`

- returns the target PDF path

#### `_iter_scan_entries(scan_cfg)`

- central helper that unifies tune and chroma scan iteration
- yields `(key, qx, qy, xi_x_label, xi_y_label)`

#### `load_maps(scan_cfg, case_name)`

- loads every available map for one scan/case
- warns and skips if files are missing

#### `plot_tune_diagram(maps, scan_cfg, case_name, n_sample=N_SAMPLE)`

- recreates the summary tune-diagram figure
- applies scan-type-specific centering and color labeling

#### `main()`

- loops over active scans and all cases

## `PlotMDIntensity.py`

Purpose:

- load MD intensity data
- load the matching simulation map
- compute comparison metrics
- generate four analysis figures

### Important code locations

- local configuration and exclusions
- measurement folder lookup
- map loader
- repetition loading
- interpolation helpers
- plot 1 through plot 4
- top-level entry point

### Function summary

#### `_map_filename(qx, qy, xi_x=None, xi_y=None)`

- rebuilds the expected simulation map filename

#### `_folder_name_tune(qx, qy, side)`

- locates a measurement folder for a tune scan

#### `_folder_name_chroma(xi_x, xi_y, side)`

- locates a measurement folder for a chroma scan

#### `_is_tune_scan(scan_cfg)`

- distinguishes tune scans from chroma scans

#### `_scan_param_value(scan_cfg, key)`

- returns the scalar that controls viridis color mapping

#### `_scan_key_to_working_point(scan_cfg, key)`

- converts a scan key into `(Qx, Qy)`

#### `_scan_key_to_chroma(scan_cfg, key)`

- extracts `(xi_x, xi_y)` for chroma scans

#### `_scan_key_label(scan_cfg, key)`

- builds a readable label for logging and warnings

#### `load_tune_map(scan_cfg, key, map_case)`

- loads the saved simulation map corresponding to one scan point

#### `_smooth(y)`

- smooths noisy arrays using Savitzky-Golay filtering

#### `_is_valid_rep(intensity)`

- filters out clearly unusable measurement repetitions

#### `load_rep_data(scan_cfg, key)`

- loads NEG/POS repetitions
- merges them into one sorted `delta` axis
- computes smoothed `dI/dδ`

#### `_common_delta(reps, n=N_SAMPLE, tm=None)`

- builds a common interpolation grid
- uses full map range if `tm` is given
- otherwise uses measured overlap only

#### `_interp_stack(reps, delta_c, key, transform=None)`

- interpolates repetitions onto a common grid
- returns stack, mean, and standard deviation

#### `_td_and_fig(scan, scan_vals, n_cbars=1)`

- constructs the tune diagram and figure scaffolding

#### `_threshold_delta(delta, intensity, threshold)`

- finds threshold-crossing `delta` values

#### `plot1_sized_by_loss(...)`

- tune diagram with marker size representing `|dI/dδ|`

#### `plot2_coloured_by_intensity(...)`

- tune diagram with marker color representing mean intensity

#### `plot3_loss_vs_delta(...)`

- mean loss rate versus `delta`

#### `plot4_tunes_vs_delta(...)`

- `Qx(delta)` and `Qy(delta)` with peak-loss markers

#### `_scan_keys_and_labels(scan)`

- centralizes scan-key ordering and colorbar labels

#### `main()`

- loads everything and writes all output figures

## Invariants To Preserve

These should remain true after any refactor.

### 1. Tune scan behavior remains unchanged

- `QxScan` and `QyScan` use scalar keys
- tune-diagram centering follows the scanned tune range

### 2. Chroma scans use fixed working-point centering

- `ChromaScanX` and `ChromaScanY` center on `(fixed_qx, fixed_qy)`

### 3. Color meaning stays correct

- `ChromaScanX` colors by `xi_x`
- `ChromaScanY` colors by `xi_y`

### 4. Naming stays consistent

- save and load logic must keep matching exactly

### 5. Helpers must support both key types

No helper should silently assume all scan keys are scalar.

## Common Failure Modes

### Save/load mismatch

If one script changes filename or folder logic and the others do not, reload steps fail.

### Scalar-key assumptions leaking into chroma logic

This usually shows up in:

- wrong colorbars
- wrong labels
- wrong file lookup
- wrong tune-diagram centering

### Partial scans

`PlotTuneMaps.py` is tolerant of missing files, but `PlotMDIntensity.py` still needs both:

- a saved map
- usable measurement repetitions

## Recommended Future Cleanup

The biggest duplication has already been reduced by `workflow_common.py`, including:

- shared scan definitions
- shared filename generation
- shared scan-key conversion
- shared tune-diagram centering rules

The next likely cleanup target would be a small shared plotting helper for repeated figure/colorbar scaffolding.
