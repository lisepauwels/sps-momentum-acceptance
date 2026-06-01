# User Guide

This guide explains how to use the workflow in practice.

The three main scripts are:

- `TuneScan.py`
- `PlotTuneMaps.py`
- `PlotMDIntensity.py`

## What Each Script Does

### `TuneScan.py`

Use this when you want to generate new simulation maps.

It:

- computes `TuneMap` objects
- saves them as `.npz` files
- can save a summary tune-diagram plot for the scan

### `PlotTuneMaps.py`

Use this when the maps already exist and you only want to replot the simulation summary figure.

It:

- reloads saved `.npz` map files
- recreates the tune-diagram summary plot
- does not use measurement data

### `PlotMDIntensity.py`

Use this when you want to compare MD measurement data with the saved simulation maps.

It:

- loads MD measurement repetitions
- loads the corresponding saved `TuneMap`
- generates comparison and analysis plots

## Normal Workflow

Run the scripts in this order.

### 1. Generate simulation maps

Edit `ACTIVE_SCANS` in `TuneScan.py`, then run:

```bash
python TuneScan.py
```

Output:

- `.npz` maps in `SweepTrajectoryMaps/...`
- one summary tune-diagram PDF per scan/case

### 2. Replot simulation maps

Edit `ACTIVE_SCANS` in `PlotTuneMaps.py`, then run:

```bash
python PlotTuneMaps.py
```

Use this if:

- the maps already exist
- you do not want to rerun the simulation
- you only need the simulation-based summary figure

### 3. Compare with MD data

Edit `SCANS` in `PlotMDIntensity.py`, then run:

```bash
python PlotMDIntensity.py
```

Output:

- four PDF plots per scan/case in `plots_md_intensity/...`

## Supported Scan Types

The workflow supports:

- `QxScan`
- `QyScan`
- `ChromaScanX`
- `ChromaScanY`

## Scan Meaning

### `QxScan`

- scanned quantity: `Qx`
- fixed quantity: `Qy`

### `QyScan`

- scanned quantity: `Qy`
- fixed quantity: `Qx`

### `ChromaScanX`

- scanned quantity: `xi_x`
- fixed working point: `(Qx, Qy)`

### `ChromaScanY`

- scanned quantity: `xi_y`
- fixed working point: `(Qx, Qy)`

## Folder Structure

Saved maps live under:

```text
SweepTrajectoryMaps/
  QxScan/
  QyScan/
  ChromaScanX/
  ChromaScanY/
```

Each scan type contains:

```text
WithErrors/
WithoutErrors/
Simplified/
```

Measurement comparison plots are saved under:

```text
plots_md_intensity/
```

## File Naming

### Tune scans

```text
tune_map_Qx{qx:.3f}_Qy{qy:.3f}.npz
```

Examples:

- `tune_map_Qx20.130_Qy20.180.npz`
- `tune_map_Qx20.135_Qy20.180.npz`

### Chroma scans

```text
tune_map_Qx{qx:.3f}_Qy{qy:.3f}_xix{xi_x:.3f}_xiy{xi_y:.3f}.npz
```

Example:

- `tune_map_Qx20.130_Qy20.180_xix0.500_xiy1.150.npz`

## Typical Tasks

### Add a tune scan point

Update the scan lists in:

- `TuneScan.py`
- `PlotTuneMaps.py`
- `PlotMDIntensity.py`

Then rerun `TuneScan.py`.

### Add a chroma scan point

Update the `xi_pairs` lists in:

- `TuneScan.py`
- `PlotTuneMaps.py`
- `PlotMDIntensity.py`

Then rerun `TuneScan.py`.

### Recreate plots without rerunning simulation

- use `PlotTuneMaps.py` for simulation-only tune diagrams
- use `PlotMDIntensity.py` for MD comparison plots

## Outputs From `PlotMDIntensity.py`

It creates four plots per scan/case.

### Plot 1

- tune diagram
- marker size reflects loss rate `|dI/dδ|`
- marker color reflects `delta`

### Plot 2

- tune diagram
- marker color reflects normalized intensity
- threshold markers show 75%, 50%, and 25% intensity crossings

### Plot 3

- `|dI/dδ|` versus `delta`
- all working points on one figure

### Plot 4

- `Qx(delta)` and `Qy(delta)` from the `TuneMap`
- vertical dotted lines mark peak-loss `delta`

## Common Problems

### Missing map files

If `PlotTuneMaps.py` or `PlotMDIntensity.py` says a map is missing:

- check that `TuneScan.py` was run for that scan and case
- check that the filename convention still matches across scripts

### Missing MD folders

If `PlotMDIntensity.py` cannot find a measurement folder:

- check the `TUNE_*` or `CHROM_*` folder naming
- check the decimal formatting used in the folder names

### Wrong plot color meaning

If chroma plots seem colored by the wrong quantity:

- check whether the script is using `xi_x` for `ChromaScanX`
- check whether the script is using `xi_y` for `ChromaScanY`

## Related Documentation

For code-level details and function explanations, see [DEVELOPER_REFERENCE.md](/Users/lisepauwels/sps_simulations/Prototyping/20260317/DEVELOPER_REFERENCE.md).

## Internal Note

The workflow now uses a shared module, `workflow_common.py`, for:

- scan definitions
- map filename generation
- scan-key interpretation
- tune-diagram centering rules

That reduces the chance that one script is updated while the others still use older assumptions.
