# Manual Tune Sweep

This folder contains a direct tune-based tracking workflow.

## Principle

The script does **not** install an RF sweep.

Instead it:

1. prepares the requested lattice
2. fixes sextupoles either `on` or `off`
3. optionally installs the selected magnetic-error variant
4. builds a target tune path from:
   - `dq_per_turn_x`
   - `dq_per_turn_y`
5. solves a `kqf0 / kqd0` schedule that reproduces this target path
6. tracks one turn at a time while updating `kqf0 / kqd0`

The solved `kqf0 / kqd0` schedule is now cached automatically under:

- `tests/manual_tune_sweep/schedule_cache/<cache-key>/`

If you rerun the same lattice and tune-path setup, the script reuses the cached schedule instead of repeating all the `line.match(...)` calls. Use `--force-rebuild-schedule` if you want to ignore the cache and solve it again.

So the control variable is directly:

- `Qx(turn) = Qx0 + turn * dq_per_turn_x`
- `Qy(turn) = Qy0 + turn * dq_per_turn_y`

This automatically covers:

- only `Qx` motion: set `dq_per_turn_y = 0`
- only `Qy` motion: set `dq_per_turn_x = 0`
- combined motion: set both nonzero

## Current Features

The main script is:

- `run_manual_tune_sweep.py`

It saves:

- turn-by-turn mean / std / higher moments for `x, px, y, py, zeta, delta`
- saved particle snapshots every `snapshot_every` turns
- violin plots of saved distributions
- phase-space evolution plots
- phase-space overlays coloured by turn
- phase-space overlays coloured by initial `delta`
- sliding FFT / NAFF products
- tune estimates from centroid motion
- dead-particle `delta` violin plots
- dead-particle tune-diagram plots
- intensity-loss plots versus:
  - turn
  - target `Qx`
  - target `Qy`
- schedule-cache metadata:
  - `schedule_cache_info.json`
  - cached schedule and status under `schedule_cache/<cache-key>/`

## Sextupoles / Errors

The workflow supports:

- `--sextupoles-mode on`
- `--sextupoles-mode off`
- `--error-variant ...`

When sextupoles are `on`, the line is matched with the requested:

- `xi_x`
- `xi_y`

When sextupoles are `off`, sextupole strengths are zeroed and the line is rematched with quadrupoles only.

## Typical Use

Example: move both tunes in the negative direction:

```bash
python run_manual_tune_sweep.py \
  --sextupoles-mode off \
  --dq-per-turn-x -2e-5 \
  --dq-per-turn-y -1e-5 \
  --num-turns 6000 \
  --num-particles 100 \
  --batch-name manual_tune_sweep_off_both
```

Example: move only `Qx`:

```bash
python run_manual_tune_sweep.py \
  --sextupoles-mode off \
  --dq-per-turn-x -2e-5 \
  --dq-per-turn-y 0 \
  --num-turns 6000 \
  --num-particles 100 \
  --batch-name manual_tune_sweep_off_qx
```

Example: same idea with sextupoles on:

```bash
python run_manual_tune_sweep.py \
  --sextupoles-mode on \
  --xi-x 0.5 \
  --xi-y 0.5 \
  --dq-per-turn-x -2e-5 \
  --dq-per-turn-y -1e-5 \
  --num-turns 6000 \
  --num-particles 100 \
  --batch-name manual_tune_sweep_on_both
```

## To Do

The current implementation assumes a **linear tune path in turn**.

That is enough for:

- pure `Qx` scans
- pure `Qy` scans
- simple diagonal motion in `(Qx, Qy)`

If needed later, this should be extended to support:

- arbitrary user-defined `(Qx(turn), Qy(turn))` paths
- piecewise paths
- paths derived from external tables
- resonance-following or resonance-crossing strategies
- schedules that vary `Qx` and `Qy` nonlinearly and independently in time
