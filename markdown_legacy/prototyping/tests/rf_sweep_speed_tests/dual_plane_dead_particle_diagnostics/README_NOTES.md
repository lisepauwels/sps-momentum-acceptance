# Dual-Plane Dead-Particle Diagnostics Notes

This folder contains post-processed diagnostics built from the saved dual-plane batch:

- `q20_xix0p5_xiy0p5/`

## Current Status

- Replot-only tooling was added so figures can be regenerated from saved `monitor_arrays.npz` and `final_particles.npz` without rerunning tracking.
- Combined slow-scan reference plots were regenerated from the saved `rf_sweep_speed_outputs` folders, with:
  - square tune-diagram layout
  - larger serif labels
  - no plot titles
  - tightened layout
  - fixed colorbar overlap

## Important Finding

The saved dual-plane batch under:

- `q20_xix0p5_xiy0p5/`

does **not** match the old slow reference scans one-to-one.

Concrete mismatches identified:

- this saved dual-plane batch has `error_variant = none`
- the old reference slow scan that was used for comparison was `three_speed_scan_errors_all_20260415/*_slow`, i.e. `error_variant = all`
- the saved dual-plane `run_config.json` originally did not contain stored `dx_monitor/dpx_monitor/dy_monitor/dpy_monitor`, which made exact centroid-based replay less reliable until reconstructed from the matched line

So if centroid NAFF from this batch still does not resemble the old slow-scan outputs, that is not automatically a plotting bug; the underlying saved physics setup is different.

## Scripts Added

- `tests/rf_sweep_speed_tests/run_saved_single_particle_naff.py`
  - saved-data single-particle NAFF post-processing
- `tests/rf_sweep_speed_tests/run_saved_centroid_naff.py`
  - centroid NAFF post-processing from saved monitor arrays
- `tests/rf_sweep_speed_tests/run_saved_centroid_naff_exact.py`
  - wrapper around the original `rf_sweep_speed_scan` helper functions for a closer replay
- `tests/rf_sweep_speed_tests/run_saved_violin_gif.py`
  - saved-data violin GIF generation
- `tests/rf_sweep_speed_tests/run_saved_action_gif.py`
  - saved-data action-distribution GIF generation
- `tests/rf_sweep_speed_tests/plot_combined_saved_slow_scan_diagrams.py`
  - combined NAFF / dead-particle tune-diagram plots for saved slow scans

## Recommendation

If the goal is strict comparison with the old slow scans, compare only against a saved batch with the same:

- `error_variant`
- tune-map case
- sweep convention
- observation point / dispersion definition

Otherwise differences in the recovered tune track may be physical rather than plotting-related.
