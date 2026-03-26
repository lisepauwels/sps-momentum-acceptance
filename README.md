# sps-momentum-acceptance

Code for SPS momentum-acceptance studies, tune/chromaticity scan workflows, HTCondor submissions, and associated postprocessing.

## Scope

This repository is intended to hold:

- reusable helper code
- active study workflows
- submission/job-generation utilities
- lightweight plotting and postprocessing code
- documentation and configs

Large simulation outputs, monitor dumps, plots, and measurement exports should live outside the repository.

## Expected Structure

```text
studies/
dev/
config/
docs/
src/common/
```

## External Dependencies

- Xsuite / Xtrack / Xcoll / Xpart
- external data under `~/phd/data`
- optional supervisor repositories for related off-momentum workflows

## Paths

Copy `config/paths.example.yaml` to `config/paths.yaml` and adapt it locally.
