# Tests Area

This directory is for focused, local studies that are still exploratory and not yet part of the main scan workflows.

## What Is Being Tested Now

### 1. Tune Evolution During a Negative Sweep

Current notebook:

- `tune_evolution_sweep.ipynb`

Goal:

- track one particle during a negative RF sweep
- log turn-by-turn optics / Twiss-derived quantities
- compare the dynamic tune evolution seen during tracking with the static tune evolution predicted from off-momentum optics

Main question:

- does the dynamically tracked tune path differ in a meaningful way from the static off-momentum prediction?

This matters because the acceptance asymmetry may depend on the actual path followed during the sweep, not only on the static tune-versus-`delta` curve.

### 2. What Tune Does the Machine Actually "See"?

Working idea:

- track a small bunch rather than one particle
- record turn-by-turn centroid motion, especially mean `x` and mean `y`
- compute spectra from the centroid signals to estimate the tune seen by a BPM-like observable during the sweep

Interpretation:

- this does make sense as a first practical observable
- it is closer to what a machine measurement sees than a single-particle tune
- it should be treated as a time-dependent centroid spectrum, not as one unique tune for the full sweep

Important caveat:

- because the RF sweep is continuous, a single FFT over the full record mixes different tune states
- the more useful object is likely a sliding-window FFT or other time-frequency analysis

### 3. RF Sweep-Speed Dependence

Current script reference:

- `rf_sweep_working_example.py`
- `rf_sweep_speed_tests/rf_sweep_speed_scan.py`
- `rf_sweep_speed_tests/run_rf_sweep_speed_cases.py`

Goal:

- vary the RF sweep speed, meaning the number of hertz changed per turn
- compare losses and phase-space evolution across sweep rates
- use one-turn tracking and sample the bunch once per turn at the start of the line by default
- optionally cycle the line to another observation point such as `qd.31110` if needed later

The new diagnostics script in this directory is intended to:

- record `sweep_per_turn`
- record `at_turn`
- record turn-by-turn beam moments for `x`, `px`, `y`, `py`, `zeta`, `delta`
- use the bunch mean `x` and `y` as BPM-like centroid signals
- store mean, standard deviation, and third central moment / skewness per turn
- save sparse particle snapshots for violin plots
- save sparse particle snapshots and assemble combined phase-space-evolution plots `x-px`, `y-py`, `zeta-delta`
- densify saved snapshots after the first particle losses, so the bunch blow-up is better resolved
- save turn-by-turn summaries and loss curves as parquet
- produce normalised intensity-loss plots versus `delta`
- produce evolution plots for the per-turn mean and spread
- produce combined violin-evolution plots with snapshot turn number on the x-axis
- produce sliding-window spectrum evolution plots for centroid signals
- produce sliding-window NAFF tune tracks when `nafflib` is available
- optionally run `nafflib.harmonics(...)` window-by-window on centroid signals if `nafflib` is installed
- also produce dispersion-subtracted centroid FFT/NAFF outputs and a direct tune-estimate vs turn / vs `delta`
- save dead-particle `delta` values and, when available, estimate `(Qx, Qy)` from the `WithoutErrors` tune map

Sweep parameterisation:

- the script can be driven either by total sweep in hertz or by sweep speed in hertz per turn
- since `total_sweep = sweep_per_turn * num_turns`, changing either lets you study sweep-speed effects directly
- the current quick scan setup uses three speeds:
- `0.5 Hz/turn` for a slower sweep
- `1.0 Hz/turn` for the nominal sweep
- `2.0 Hz/turn` for a faster sweep that stays moderate
- these are run for both `DPpos` and `DPneg`, initially with `100` particles and `6000` turns

## Suggested Tune-Extraction Strategy

For both measurements and simulations:

- start from centroid signals if the goal is "machine-like" tune estimation
- use sliding FFT windows rather than one FFT over the full sweep
- explicitly inspect spectrum evolution during the sweep, not only one integrated spectrum
- apply NAFF on the same short windows if a sharper frequency estimate is needed
- apply a leakage-reducing window such as Hann or Blackman
- inspect both the dominant line and sidebands when satellites appear
- compare centroid-based tune extraction against individual-particle tracking when possible

If the tune remains ambiguous because sidebands carry real dynamics, that ambiguity should be preserved in the analysis rather than hidden by forcing one peak selection.

### Practical Window / Step Choice

The sliding-window tune extraction should not be treated as arbitrary. The useful way to choose it is to test stability across a small set of window sizes.

- `window` sets the tradeoff between frequency resolution and time resolution
- larger `window` gives cleaner frequency estimates but mixes more of the sweep evolution
- smaller `window` follows the sweep more locally but becomes noisier
- `step` should usually be about one quarter of the window so adjacent estimates overlap strongly without becoming redundant

Suggested starting values:

- `window = 128`, `step = 32`
- `window = 256`, `step = 64`
- `window = 512`, `step = 128`

Interpretation rule:

- if the extracted tune ridge / NAFF track is qualitatively stable across these settings, the estimate is likely meaningful
- if it changes strongly with the chosen window, the signal is not yet robust enough and the result should be treated cautiously
- for short runs or runs that stop early because the bunch is lost, a smaller window is often preferable so there are still enough windows to resolve the evolution before loss

## Environment Note

Expected runtime environment:

- `mamba activate xcoll`

`nafflib` is currently treated as optional. If it is not installed, the script still runs and saves the raw centroid signals needed for later harmonic analysis.

## Sextupoles-Off Notes

The exploratory work in `tests/sextupoles_off/` led to a useful correction:

- the asymmetry difference between notebook and script was not caused by poor statistics
- the notebook-style shutdown was broader than "sextupoles off"

The practical mistake was zeroing broad variable families such as:

- `kl.*`
- `ks.*`
- `kls.*`

This can also zero `mdh/mdv`-type orbit-correction or auxiliary multipole variables, so the resulting tracked line is not equivalent to a targeted sextupoles-off lattice.

Current guidance:

- if the intent is "sextupoles off only", use the targeted scripted setup
- if the intent is "broad knob shutdown", document that as a separate case
- do not interpret those two cases as statistics variations of the same study

Two command patterns are now the recommended starting points.

Line-start diagnostics with turn-by-turn moments and NAFF-style postprocessing:

```bash
python tests/sextupoles_off/run_sextupoles_off_line_start_diagnostics.py \
  --reuse-tune-map \
  --total-sweep-hz 0 \
  --planes DPneg \
  --batch-name sextupoles_off_no_sweep_diag
```

Large-statistics sweep with only loss/survival outputs:

```bash
python tests/sextupoles_off/run_sextupoles_off_monitor_scan.py \
  --loss-only \
  --reuse-tune-map \
  --num-particles 1000 \
  --batch-name sextupoles_off_1000p_lossonly
```

The loss-only mode is intentionally plain:

- no monitor
- no cycling
- no one-turn Python summary loop
- one straight `line.track(num_turns=...)`
- postprocessing only from final loss data

## Outputs Philosophy

This directory can contain:

- notebooks
- small scripts
- lightweight local documentation

Large monitor dumps and heavy result folders should preferably stay outside the repository once the workflow stabilises.
