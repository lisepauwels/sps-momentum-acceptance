# Momentum Acceptance Area

## Scientific Context

This repository exists to understand the SPS momentum-acceptance asymmetry observed during RF-frequency sweeps.

### Measurement Picture

- In SPS momentum-acceptance measurements, the RF frequency is swept to generate a momentum offset `delta`.
- Losses start at different absolute `delta` depending on sweep sign.
- The measured acceptance appears tighter on the negative-`delta` side than on the positive side.
- The negative side also appears to depend on chromaticity.
- These statements are based on machine measurements, not only on simulations.

### First Simulation Picture

- A first simulation round without errors reproduces the qualitative chromaticity dependence.
- The simulated acceptance is less restrictive than what is seen in the machine.

### Deeper Chroma Scan Interpretation

- A broader scan over both positive and negative chromaticity shows a sign-switching tendency.
- Positive chromaticity produces the strongest dependence for negative `delta` variation.
- Negative chromaticity produces the strongest dependence for positive `delta` variation.
- This points to a beam-dynamics effect rather than a simple static-aperture asymmetry.

### Tune-Based Interpretation

- Replotting the observations as tune shift `dQ` rather than `delta` shows a clearer asymmetry in `dQ`.
- The effect seems to be associated with `dQ < 0`, while `dQ > 0` is much less critical.
- Current working hypothesis:
  negative `dQ` drives the beam into a resonance, while positive `dQ` does not, so positive-side losses may instead be dominated by dispersion / aperture.

### Resonance Question

The resonance actually being crossed is still unclear. Current candidates are:

- the coupling resonance `Qx - Qy = 0` in fractional tune
- an integer resonance, effectively `Qx = 0` or `Qy = 0` in fractional tune

The tune-diagram studies in this repository are mainly there to identify which resonance is being approached and under which model assumptions.

### Model Status

- There is also a model including errors.
- Its tune evolution is supposed to match the machine more closely.
- The loss behaviour in simulation with errors is still not fully convincing.
- It is nevertheless the best available model for estimating where the beam enters a resonance region.
- Different models may lose on different resonances, so model dependence must be tracked explicitly.

### Measurements Outside This Repo

Relevant measurement data live in:

- `/Users/lisepauwels/phd/data/sps-measurements/`

Work already done there includes:

- tune scans
- chromaticity scans
- first chromaticity measurements

The chromaticity-measurement analysis is not complete yet. A more robust tune-extraction method is still needed, likely involving improved FFT-based analysis and filtering.

### Tune-Extraction Difficulty

- In both simulation and measurements, once `|delta|` becomes too large, spectral satellites appear.
- At that point the definition of "the tune" becomes ambiguous because the motion is no longer represented by a single clean line.
- This is an open methodological issue, not just a plotting problem.

Practical working suggestions for now:

- Track or measure both planes turn-by-turn and inspect the full spectrum, not only the dominant peak.
- Use short sliding windows so tune evolution can be followed during the sweep.
- Apply a window function before FFT, for example Hann or Blackman, to reduce leakage.
- Compare centroid tune extraction with single-particle or small-subset tune extraction when satellites appear.
- If needed, complement plain FFT with NAFF-style frequency extraction or peak tracking across overlapping windows.

### Previous Study Location

Previous studies were led in:

- `/Users/lisepauwels/sps_simulations/Studies/MomentumAcceptance/`

This is the clearest "main project" in the workspace /Users/lisepauwels/sps_simulations/Studies/MomentumAcceptance/

In this directory, cleared up version

## Core Structure

`Studies/MomentumAcceptance/` contains several related subprojects:

- `HelperFunctions/`
- `IntensityScan/`
- `IntensityScan2/`
- `LossLocationsCheck/`
- `TrackingNoSweepNoCav/`
- `TuneDiagramVariations/`
- `ErrorVariants/`

## What `HelperFunctions/` Is

This is the strongest candidate for the core reusable library of the momentum-acceptance work.

Important files:

- `tune_diagram.py`
  General plotting and sweep-trajectory utilities. Defines reusable concepts like `TuneDiagram`, `SweepTrajectory`, and `TuneMap`.

- `combine_death_turns.py`
  Generic result-combination tool for HTCondor-style `job_*` folders. This is already written in a reusable, study-agnostic way.

- `acceptance_centers.py`, `midpoints_analysis.py`, `intensity_loss.py`, `tune_analysis.py`, `twiss_analysis.py`, `beta_analysis.py`, `plot_helpers.py`
  Analysis helpers around acceptance, midpoint extraction, plotting, and optics/tune-related postprocessing.

- `xjson.py`
  Small JSON helper for numpy-friendly serialization.

My reading is that `HelperFunctions/` is the beginning of a real internal package, even if it is not packaged yet.

## What the Study Subfolders Are Doing

### `IntensityScan/`

Looks like an RF-sweep momentum-acceptance study over a small set of chromaticity values.

It contains:

- submission files
- job script(s)
- combined `.json.gzip` outputs
- notebook-based plotting

The submission script tracks a Gaussian bunch through an RF sweep with a TIDP collimator installed and stores `death_turns.json`.

### `IntensityScan2/`

Looks like the expanded and more systematic version of `IntensityScan/`.

It contains:

- many combined `combined_linear_*.json.gzip` and `combined_errors_*.json.gzip`
- summary figures versus chromaticity
- plotting notebooks

This looks more like a mature result area than active source code.

### `TuneDiagramVariations/`

This is the most "production workflow" subproject inside `MomentumAcceptance`.

It has:

- `scripts/chroma_vars.py`
- `scripts/tune_shifts.py`
- `submission_scripts/generate_jobs.py`
- `submission.sub`, `job.sh`, `environment.sh`
- example job specification YAML
- result folders and plotting notebooks

Interpretation:

- `chroma_vars.py` runs RF-sweep studies for varying chromaticities at fixed working point.
- `tune_shifts.py` runs RF-sweep studies for varying tune points at fixed chromaticity.
- `generate_jobs.py` is a clean HTCondor job-list generator and is definitely reusable.

This subfolder looks like a strong template for future parameter-scan studies.

### `ErrorVariants/`

Looks like a study that compares different installed magnetic-error variants.

It contains:

- input line
- large result JSON collections
- post-processing script

`results/postprocess.py` combines loss maps and particle dictionaries from `job_*` folders, again suggesting a reusable pattern for distributed studies.

### `TrackingNoSweepNoCav/`

Looks like a complementary diagnostic study:

- no RF sweep
- cavities turned off
- monitors inserted at many QD bottlenecks
- tracking done on a grid in `x_norm` and `delta`

This is more of a diagnostic / optics-understanding tool than a final scan workflow.

### `LossLocationsCheck/`

Looks like a stored analysis result area focused on loss-location JSON files and plotting notebooks. More output-heavy than code-heavy.

## Relation To `Prototyping/20260317`

`Prototyping/20260317/` looks like the current development branch of this whole topic.

Key sign:

- it imports from `Studies/MomentumAcceptance/HelperFunctions`
- it has its own workflow docs
- it defines scan configs and produces `TuneMap` outputs in a cleaner, more script-oriented way

So conceptually:

- `Studies/MomentumAcceptance/` = established study family
- `Prototyping/20260317/` = current next-generation workflow under development

For the new clean repository, I would not keep the `Prototyping/20260317` naming. I would promote the source code into a proper study folder, for example:

- `tune_scan_workflow/`

That promotion has effectively happened in the clean repo:

- `/Users/lisepauwels/phd/code/sps-momentum-acceptance/tune_scan_workflow/`

Important later clarification:

- external dependencies for this workflow should come through `config/paths.yaml`
- workflow-owned outputs should stay local to the workflow for now
- do not split its local plots/maps into many top-level external data roots yet

So the intended pattern is:

- external inputs via `paths.yaml`
- local outputs in `tune_scan_workflow/SweepTrajectoryMaps/`
- local outputs in `tune_scan_workflow/plots_md_intensity/`

This keeps the workflow self-contained while still removing hardcoded dependencies on the old absolute repository layout.

## Relation To `Studies/OffMomBumpScans`

This looks like a sibling / predecessor study, not a separate scientific universe.

It also does:

- RF sweep loss studies
- bump scans
- midpoint extraction / plotting
- simulation vs MD comparison

I would keep it close to `MomentumAcceptance`, at least initially.

## Best Candidate Repo Boundary

A future `sps-momentum-acceptance` repository would likely include:

- `Studies/MomentumAcceptance/HelperFunctions/`
- `Studies/MomentumAcceptance/TuneDiagramVariations/`
- selected parts of `Studies/MomentumAcceptance/ErrorVariants/`
- selected parts of `Studies/MomentumAcceptance/TrackingNoSweepNoCav/`
- `Studies/OffMomBumpScans/`
- promoted source files from `Prototyping/20260317/` as a proper workflow directory

But I would not include all stored result files in the main code repo.

## Current Clean-Repo Adaptations

## Sextupoles-Off Debugging Note

Recent debugging in `tests/sextupoles_off/` clarified an important failure mode.

### What was tested

- a sextupoles-off RF sweep using both notebook code and scripted workflows
- line-start diagnostics with turn-by-turn bunch moments and NAFF postprocessing
- pure loss-only sweeps with no monitor and no turn-by-turn summaries

### Main outcome

- the apparent difference between the notebook and the script was not a statistics issue
- the main asymmetry seen in the notebook survived a tune rematch
- the key discrepancy was that the notebook-style broad variable shutdown was not equivalent to "turn off sextupoles only"

### Important mistake identified

In the notebook, broad variable families such as:

- `kl.*`
- `ks.*`
- `kls.*`

were zeroed after the initial match.

This does not only switch off sextupoles. It can also switch off other powered knobs and multipole/corrector families, including `mdh/mdv`-type orbit-correction variables. That changes the machine state beyond a clean sextupoles-off configuration.

Interpretation:

- a targeted sextupoles-off workflow zeros the sextupole strengths and chromatic knobs only
- the notebook-style broad shutdown can also remove orbit-correction terms
- therefore the two cases are physically different and should not be compared as if they were the same lattice

### Why the earlier scripted result looked too symmetric

Several issues were identified in early script versions:

- a `TIDP` collimator was inserted even when the goal was to compare against the simple notebook sweep
- the early loss-only mode still passed through observation-oriented setup paths
- some workflows rematched after the sextupoles-off step while the notebook tracked a different post-zeroing state

Those differences can suppress or reshape the asymmetry and must be controlled explicitly.

### Current practical rule

When the goal is "sextupoles off":

- do not zero broad variable families unless that wider shutdown is explicitly the physics case of interest
- document whether correctors / orbit bumps / auxiliary multipole knobs are also being switched off
- treat "sextupoles off only" and "sextupoles plus broad knob families off" as separate studies

### Current scripts in `tests/sextupoles_off`

- `run_sextupoles_off_monitor_scan.py`
  general sextupoles-off sweep script
- `run_sextupoles_off_line_start_diagnostics.py`
  line-start diagnostics launcher with turn-by-turn moments and NAFF-style postprocessing
- `replot_batch_intensity_vs_twiss_x.py`
  rebuilds batch-level intensity plots including the closed-orbit `x(delta)` mapping at `qd.31110`

### Recommended command split

No-sweep / line-start diagnostics:

```bash
python tests/sextupoles_off/run_sextupoles_off_line_start_diagnostics.py \
  --reuse-tune-map \
  --total-sweep-hz 0 \
  --planes DPneg \
  --batch-name sextupoles_off_no_sweep_diag
```

Large-statistics sweep with loss-only outputs:

```bash
python tests/sextupoles_off/run_sextupoles_off_monitor_scan.py \
  --loss-only \
  --reuse-tune-map \
  --num-particles 1000 \
  --batch-name sextupoles_off_1000p_lossonly
```

The clean repository at `/Users/lisepauwels/phd/code/sps-momentum-acceptance` now also contains:

- `src/common/load_paths.py`
  Small config loader returning `Path` objects.

- `tune_scan_workflow/PlotIntensityLoss.py`
  Quick plotting script for loss-vs-delta summaries across tune and chroma scans.

- `tune_scan_workflow/PlotScanSummaries.py`
  Summary plotting script using colorbars to encode the varying scan parameter, inspired by the older `TuneDiagramVariations` figure style.

One important caution:

- `config/paths.example.yaml` in the clean repo has been manually corrected by the user and should be treated as the current working reference unless they explicitly want it rewritten.
