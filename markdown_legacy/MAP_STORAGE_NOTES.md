# Map Storage Notes

## Current issue

Tune maps are currently stored by workflow/folder convention rather than by physics point.

That causes mismatches such as:
- coupled `xi_x = xi_y` studies being stored under `ChromaScanX`
- regenerated maps ending up in different `SweepTrajectoryMaps` roots depending on the working directory
- plotting scripts having to guess filenames and folders instead of querying by `(qx, qy, xi_x, xi_y, case)`

## Immediate commands

Generate IntensityScan2 coupled maps for all study `xi` values with the default full error model:

```bash
/Users/lisepauwels/miniforge3/envs/xcoll/bin/python \
tune_scan_workflow/generate_intensityscan2_missing_without_errors_maps.py \
  --study-root /Users/lisepauwels/sps_simulations/Studies/MomentumAcceptance/IntensityScan2 \
  --qx 20.13 \
  --qy 20.18 \
  --coupled-xi \
  --case WithErrors \
  --error-variant all \
  --with-diagram
```

Generate the same coupled maps but with only quadrupole `b6` errors:

```bash
/Users/lisepauwels/miniforge3/envs/xcoll/bin/python \
tune_scan_workflow/generate_intensityscan2_missing_without_errors_maps.py \
  --study-root /Users/lisepauwels/sps_simulations/Studies/MomentumAcceptance/IntensityScan2 \
  --qx 20.13 \
  --qy 20.18 \
  --coupled-xi \
  --case WithErrors \
  --error-variant quadrupole_b6 \
  --with-diagram
```

This writes into a separate subdirectory so it does not overwrite the existing full-error maps:

- `SweepTrajectoryMaps/ChromaScanX/WithErrors_quadrupole_b6/`

Rule now implemented in the generator:
- `--case WithErrors --error-variant all` -> `WithErrors/`
- `--case WithErrors --error-variant quadrupole_b6` -> `WithErrors_quadrupole_b6/`

## Code path

The relevant pieces are:
- `tune_scan_workflow/TuneScan.py`
  - `install_errors(...)`
  - `_build_and_save(...)`
- `tune_scan_workflow/generate_intensityscan2_missing_without_errors_maps.py`

The generator now accepts `--error-variant` and forwards it into the shared builder.

## Migration direction

Do not reorganize files first.

Recommended order:
1. Build a global map index keyed by:
   - `qx`
   - `qy`
   - `xi_x`
   - `xi_y`
   - `case`
   - `error_variant`
   - `path`
   - optional provenance fields
2. Make plotting and analysis scripts query the index instead of constructing paths by hand.
3. Only after that, consider physically reorganizing the map files.

The goal is to stop scripts from relying on folder names like:
- `ChromaScanX`
- `ChromaScanY`
- `WithErrors`
- `WithoutErrors`

when the real question is:
- "load the map for this physics point"

## Current state as of 2026-05-09

There are now multiple active map roots in use:

- repo-local root:
  - `phd/code/sps-momentum-acceptance/SweepTrajectoryMaps/`
- older prototyping root:
  - `/Users/lisepauwels/sps_simulations/Prototyping/20260317/SweepTrajectoryMaps/`
- workflow symlinked root:
  - `tune_scan_workflow/SweepTrajectoryMaps`

This already caused one concrete failure:
- new `WithErrors` coupled maps were generated into the repo-local root
- midpoint plotting was still looking in the prototyping root
- result: plotting reported missing maps even though the generation had succeeded

So for now, every plotting command must be checked against the exact map root it is using.

## IntensityScan2 coupled maps

Relevant generator:
- `tune_scan_workflow/generate_intensityscan2_missing_without_errors_maps.py`

It now supports:
- `--case WithErrors|WithoutErrors`
- `--error-variant ...`
- `--coupled-xi`
- `--with-diagram`

Important current behavior:
- output is written under the repo-local `SweepTrajectoryMaps/`
- not automatically into the old prototyping directory

Examples:

Full coupled `WithErrors`:

```bash
/Users/lisepauwels/miniforge3/envs/xcoll/bin/python \
tune_scan_workflow/generate_intensityscan2_missing_without_errors_maps.py \
  --study-root /Users/lisepauwels/sps_simulations/Studies/MomentumAcceptance/IntensityScan2 \
  --qx 20.13 \
  --qy 20.18 \
  --coupled-xi \
  --case WithErrors \
  --error-variant all \
  --with-diagram
```

Full coupled `WithoutErrors`:

```bash
/Users/lisepauwels/miniforge3/envs/xcoll/bin/python \
tune_scan_workflow/generate_intensityscan2_missing_without_errors_maps.py \
  --study-root /Users/lisepauwels/sps_simulations/Studies/MomentumAcceptance/IntensityScan2 \
  --qx 20.13 \
  --qy 20.18 \
  --coupled-xi \
  --case WithoutErrors \
  --with-diagram
```

Coupled `WithErrors` with only quadrupole `b6`:

```bash
/Users/lisepauwels/miniforge3/envs/xcoll/bin/python \
tune_scan_workflow/generate_intensityscan2_missing_without_errors_maps.py \
  --study-root /Users/lisepauwels/sps_simulations/Studies/MomentumAcceptance/IntensityScan2 \
  --qx 20.13 \
  --qy 20.18 \
  --coupled-xi \
  --case WithErrors \
  --error-variant quadrupole_b6 \
  --with-diagram
```

This last one now writes into:
- `SweepTrajectoryMaps/ChromaScanX/WithErrors_quadrupole_b6/`

so it does not overwrite:
- `SweepTrajectoryMaps/ChromaScanX/WithErrors/`

## Midpoint tune plots for IntensityScan2

Relevant script:
- `helper_functions/generate_intensityscan2_midpoint_qx_plots.py`

It now supports:
- separate `map_root` for `linear`
- separate `map_root_errors` for `errors`
- separate `md_map_root` for MD point conversion
- `Qy` output in addition to `Qx`

Important current behavior:
- simulation `linear` uses `map_root`
- simulation `errors` uses `map_root_errors`
- MD overlays use `md_map_root` if provided, otherwise `map_root`

This matters because earlier the MD points were always being projected through the no-error maps.

## Generated comparison plot folders

These IntensityScan2 midpoint comparison folders were generated:

- `Figures_midpoint_qx_linear_md_errormaps/`
- `Figures_midpoint_qx_both_md_errormaps/`
- `Figures_midpoint_qx_linear_nomd/`
- `Figures_midpoint_qx_both_nomd/`

Each contains:
- `qx50_vs_xi.png`
- `qy50_vs_xi.png`

## Restricted coupled trajectory plot

A dedicated plotter was added for the restricted coupled `WithErrors` trajectory subset:

- `tune_scan_workflow/plot_selected_coupled_chromascan.py`

Current output:
- `SweepTrajectoryMaps/ChromaScanX/WithErrors/tune_diagram_WithErrors_selected_coupled_xi.pdf`

using only:
- `xi = -1.5, -0.5, 0.0, 0.5, 1.5`

with:
- plasma trajectory colors
- no colorbars
- red marker for positive-sweep end
- blue marker for negative-sweep end

## Why migration is now urgent

The main problem is no longer only naming. It is that:
- generation and plotting are already reading/writing different roots
- variant-specific cases like `quadrupole_b6` need separate namespaces
- MD overlays need explicit map-family choices
- workflow names like `ChromaScanX` are no longer enough to identify the physics content

## Recommended next migration steps

1. Freeze one canonical map root.
   Example:
   - `phd/code/sps-momentum-acceptance/SweepTrajectoryMaps/`

2. Add one registry/index file in that root.
   Each record should contain at least:
   - `path`
   - `qx`
   - `qy`
   - `xi_x`
   - `xi_y`
   - `case`
   - `error_variant`
   - `generation_method`
   - `delta_min`
   - `delta_max`

3. Make plotting/analysis scripts query the registry instead of constructing paths by folder logic.

4. Only after that, move or merge the old prototyping maps into the canonical root.

## Scripts that should be adapted first

- `helper_functions/generate_intensityscan2_midpoint_qx_plots.py`
- `tune_scan_workflow/generate_intensityscan2_missing_without_errors_maps.py`
- `tune_scan_workflow/PlotTuneMaps.py`
- `tune_scan_workflow/PlotMDIntensity.py`
- `tune_scan_workflow/generate_single_machine_tune_map.py`

## Practical rule for now

Before running any plot command, check:
- which root contains the maps you just generated
- which root the plotting script is reading

If those are different, the result is unreliable even if both directories look valid.
