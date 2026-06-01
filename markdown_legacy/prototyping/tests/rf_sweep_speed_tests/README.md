# RF Sweep Speed Tests

This directory contains the current local workflow for SPS RF-sweep speed studies.

## Main Scripts

- `rf_sweep_speed_scan.py`
  - single-case tracking + diagnostics
- `run_rf_sweep_speed_cases.py`
  - batch launcher for `DPpos/DPneg` and `slow/nominal/fast`
- `recompute_saved_tune_analysis.py`
  - post-process saved runs to regenerate FFT / NAFF / tune-estimate outputs without rerunning tracking
- `tune_evolution_sweep.ipynb`
  - notebook for dynamic tune evolution checks during a sweep

## Output Location

By default, outputs are written under:

- `rf_sweep_speed_outputs/`

relative to this directory.

Typical structure:

- `rf_sweep_speed_outputs/<batch-name>/`
- `rf_sweep_speed_outputs/<batch-name>/<case-name>/`

## Current Tracking Workflow

Single-case script:

- one-turn tracking loop
- bunch sampled once per turn at the line start by default
- per-turn reduced bunch moments for `x, px, y, py, zeta, delta`
- sparse particle snapshots for violin and phase-space evolution
- denser snapshots after first losses
- early stop once `alive_count <= 1`

Main saved tables:

- `turn_summary.parquet`
- `intensity_loss.parquet`
- `dead_particles.parquet`
- `tune_estimate.parquet`

Main saved figures:

- `mean_evolution.png`
- `std_evolution.png`
- `delta_envelope_vs_sweep_delta.png`
- `centroid_spectrogram.png`
- `centroid_spectrogram_beta.png`
- `sliding_naff_global.png`
- `sliding_naff_beta.png`
- `tune_estimate_vs_turn.png`
- `tune_estimate_vs_delta.png`
- `tune_estimate_vs_sweep_map.png`
- `naff_harmonics_positive_vs_sweep_map.png`
- `dead_particle_tune_diagram.png`
- `naff_tune_diagram.png`

## Current NAFF Conventions

Primary current setting used for saved reprocessing:

- `window = 256`
- `step = 64`
- main NAFF harmonic count often regenerated with `6`

There are now several comparison variants:

- default phase-space NAFF
  - uses `(x, px)` and `(y, py)`
- positive-harmonic comparison
  - `naff_harmonics_positive_vs_sweep_map.png`
  - keeps only harmonics with positive NAFF frequency
- position-only variant
  - filenames suffixed with `_xonly_h3`
  - uses `x` / `y` only
  - uses `3` harmonics

## Tune-Map Convention

Dead-particle and NAFF comparisons use the tune map from:

- `tune_scan_workflow/SweepTrajectoryMaps/ChromaScanY/<case>/`

where `<case>` is typically:

- `WithoutErrors` for `error_variant = none`
- `WithErrors` for `error_variant != none`

## Resonance Plotting Rule

The tune-diagram resonance styling was updated:

- first order is always non-skew
- for order `>= 2`, resonances with odd vertical coefficient are treated as skew
- resonances with even vertical coefficient are treated as normal

This was introduced to better match the RDT-style rule based on
`(p-q) Qx + (r-t) Qy = n`.

## Notes

- Some older saved `run_config.json` and `batch_index.json` files still contain historical absolute paths from before this directory move.
- Current scripts write new outputs under the moved `rf_sweep_speed_tests` directory.
- If a saved case already exists, the tracking script creates a timestamped fallback directory instead of overwriting it.

## Relation To `tests/sextupoles_off`

The sextupoles-off study now reuses several ideas from this workflow, but the two modes should be kept distinct.

Important clarification from recent debugging:

- `rf_sweep_speed_tests` samples bunch moments once per turn in a Python one-turn loop
- that is useful for centroid / NAFF diagnostics
- it is not the right pattern for a pure high-statistics loss-only scan

For large-statistics sextupoles-off loss studies, the correct approach is now:

- no monitor
- no one-turn summary loop
- one plain `line.track(num_turns=...)`
- derive survival/loss, dead-particle `delta`, tune-map estimates, and closed-orbit coordinate plots from final loss data

Another important debugging result:

- broad notebook shutdowns using families such as `kl.*`, `ks.*`, `kls.*` are not equivalent to "sextupoles off only"
- those shutdowns can also remove `mdh/mdv`-type correction variables or other multipole families
- therefore they can change the asymmetry for reasons unrelated to pure sextupole removal

The line-start sextupoles-off diagnostics launcher is:

```bash
python tests/sextupoles_off/run_sextupoles_off_line_start_diagnostics.py --reuse-tune-map
```

The large-statistics sextupoles-off loss-only launcher is:

```bash
python tests/sextupoles_off/run_sextupoles_off_monitor_scan.py \
  --loss-only \
  --reuse-tune-map \
  --num-particles 1000 \
  --batch-name sextupoles_off_1000p_lossonly
```

## Common Commands

Activate the environment first:

```bash
mamba activate xcoll
```

Single-case run, no errors:

```bash
python tests/rf_sweep_speed_tests/rf_sweep_speed_scan.py \
  --case-name test_nominal_dpneg \
  --plane DPneg \
  --sweep-per-turn-hz 1.0 \
  --num-particles 200 \
  --num-turns 6000
```

Single-case run, all errors:

```bash
python tests/rf_sweep_speed_tests/rf_sweep_speed_scan.py \
  --case-name test_nominal_dpneg_errors_all \
  --plane DPneg \
  --sweep-per-turn-hz 1.0 \
  --num-particles 200 \
  --num-turns 6000 \
  --error-variant all \
  --tune-map-case WithErrors
```

Full 6-case batch, no errors:

```bash
python tests/rf_sweep_speed_tests/run_rf_sweep_speed_cases.py \
  --python /Users/lisepauwels/miniforge3/envs/xcoll/bin/python \
  --num-particles 200 \
  --num-turns 6000 \
  --batch-name three_speed_scan_20260414
```

Full 6-case batch, all errors:

```bash
python tests/rf_sweep_speed_tests/run_rf_sweep_speed_cases.py \
  --python /Users/lisepauwels/miniforge3/envs/xcoll/bin/python \
  --num-particles 200 \
  --num-turns 6000 \
  --error-variant all \
  --tune-map-case WithErrors \
  --batch-name three_speed_scan_errors_all_20260415
```

Recompute saved tune-analysis products without rerunning tracking:

```bash
python tests/rf_sweep_speed_tests/recompute_saved_tune_analysis.py \
  --fft-window 256 \
  --fft-step 64 \
  --naff-harmonics 6 \
  tests/rf_sweep_speed_tests/rf_sweep_speed_outputs/three_speed_scan_20260414/DPneg_nominal
```

Recompute for several saved cases at once:

```bash
python tests/rf_sweep_speed_tests/recompute_saved_tune_analysis.py \
  --fft-window 256 \
  --fft-step 64 \
  --naff-harmonics 6 \
  tests/rf_sweep_speed_tests/rf_sweep_speed_outputs/three_speed_scan_20260414/DPneg_fast \
  tests/rf_sweep_speed_tests/rf_sweep_speed_outputs/three_speed_scan_20260414/DPneg_nominal \
  tests/rf_sweep_speed_tests/rf_sweep_speed_outputs/three_speed_scan_20260414/DPneg_slow
```
