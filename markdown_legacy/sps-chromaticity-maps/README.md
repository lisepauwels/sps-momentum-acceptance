# SPS Chromaticity Maps

This directory is the clean physics-keyed store for chromaticity tune maps.

## Structure

Maps are grouped by optics model, not by the original scan workflow:

- `simplified/`
- `with_errors/`
- `without_errors/`

Special error subsets live under `with_errors/`, for example:

- `with_errors/quadrupole_b6/`

## Naming

Every map file uses the same physics-point naming scheme:

```text
tune_map_Qx{qx:.3f}_Qy{qy:.3f}_xix{xi_x:.3f}_xiy{xi_y:.3f}.npz
```

Example:

```text
tune_map_Qx20.130_Qy20.180_xix0.500_xiy0.500.npz
```

There is no filename distinction based on whether a map originally came from
`ChromaScanX` or `ChromaScanY`.

## Migration notes

This store was populated from:

- `phd/code/sps-momentum-acceptance/SweepTrajectoryMaps`
- `sps_simulations/Prototyping/20260317/SweepTrajectoryMaps`

The detailed copy log is in `migration_manifest.json`.

Collision rule used for the first migration:

- repo-local maps were preferred over legacy prototyping maps
- byte-identical duplicates were skipped
- content conflicts were not overwritten and are recorded in the manifest
