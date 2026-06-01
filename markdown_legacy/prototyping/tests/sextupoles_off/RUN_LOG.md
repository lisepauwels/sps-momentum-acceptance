# Sextupoles-Off Run Log

This note records the main conclusions from the debugging and the runs that were actually attempted in this test area.

## What Was Learned

- The discrepancy between notebook and script was **not** a statistics issue.
- The notebook-style broad variable shutdown was **not** equivalent to "sextupoles off only".
- Zeroing broad families such as:
  - `kl.*`
  - `ks.*`
  - `kls.*`
  can also switch off additional knobs, likely including `mdh/mdv`-type orbit-correction / multipole variables.
- That broader shutdown can preserve or create an asymmetry that is not present in the cleaner targeted sextupoles-off script.

## Important Script Corrections Made

- Added a true `--loss-only` mode in `run_sextupoles_off_monitor_scan.py`:
  - no monitor
  - no cycling
  - no one-turn summary loop
  - one straight `line.track(num_turns=...)`
  - postprocessing only from final loss data
- Added a shared sextupoles-off tune-map cache:
  - `tune_map_sextupoles_off_shared.npz`
  - `tune_map_sextupoles_off_shared.png`
- Added batch-level comparison plots versus:
  - `delta`
  - `Qx`, `Qy`
  - `dQx`, `dQy`
  - closed orbit `x_co(delta)` from `twiss4d(delta0=delta)` at `qd.31110`
- Added a dedicated no-sweep diagnostics script:
  - `run_sextupoles_off_no_sweep.py`
- Added snapshot saving and phase-space plotting in the no-sweep workflow:
  - overlays coloured by turn
  - overlays coloured by initial `delta`

## Runs Attempted

### 1. Early 1000-particle RF-style run

Attempted command pattern:

```bash
python run_sextupoles_off_monitor_scan.py \
  --observation-mode rf_style \
  --no-cycle \
  --num-particles 1000 \
  --batch-name sextupoles_off_1000part
```

Outcome:

- This was too slow for the intended purpose.
- It was using the RF-style one-turn loop, not the plain loss-only tracking path.
- It was therefore not the right workflow for a large-statistics loss study.

### 2. Loss-only 1000-particle scan

Requested command pattern:

```bash
python run_sextupoles_off_monitor_scan.py \
  --loss-only \
  --reuse-tune-map \
  --num-particles 1000 \
  --batch-name sextupoles_off_1000p_lossonly
```

Important debugging note:

- an earlier version of `--loss-only` still passed through observation-oriented setup and was corrected afterward
- the intended final behaviour is now:
  - plain sextupoles-off line
  - plain `line.track(num_turns=...)`
  - losses only

### 3. No-sweep sextupoles-off diagnostics

Current command pattern:

```bash
python run_sextupoles_off_no_sweep.py \
  --reuse-tune-map \
  --snapshot-every 100 \
  --batch-name sextupoles_off_no_sweep_ycheck
```

Purpose:

- inspect the vertical instability seen even without RF sweep
- save turn-by-turn mean / std
- inspect phase-space evolution and NAFF outputs

### 4. Manual tune-sweep studies

Current direct tune-path script:

- `tests/manual_tune_sweep/run_manual_tune_sweep.py`

Example command that was discussed:

```bash
python run_manual_tune_sweep.py \
  --sextupoles-mode off \
  --dq-per-turn-x=-2e-5 \
  --dq-per-turn-y=0 \
  --schedule-points 41 \
  --num-turns 6000 \
  --num-particles 100 \
  --progress-every 100 \
  --batch-name sext_off_qx
```

and the sextupoles-on equivalent:

```bash
python run_manual_tune_sweep.py \
  --sextupoles-mode on \
  --xi-x 0.5 \
  --xi-y 0.5 \
  --dq-per-turn-x=-2e-5 \
  --dq-per-turn-y=0 \
  --schedule-points 41 \
  --num-turns 6000 \
  --num-particles 100 \
  --progress-every 100 \
  --batch-name sext_on_qx_xi05
```

## Current Status

- The main open physics question in this folder is the vertical instability seen without RF sweep.
- The main methodological correction is now clear:
  a targeted sextupoles-off study must not be confused with a broader variable shutdown.
