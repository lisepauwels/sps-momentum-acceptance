from __future__ import annotations

from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required to load config/paths.yaml") from exc


def repo_root(start: str | Path | None = None) -> Path:
    here = Path(start) if start is not None else Path(__file__).resolve()
    for path in [here, *here.parents]:
        if (path / "config").exists():
            return path
    raise FileNotFoundError("Could not locate repository root containing config/")


def load_paths(filename: str = "paths.yaml", required: bool = False) -> dict:
    root = repo_root()
    cfg = root / "config" / filename
    if not cfg.exists():
        if required:
            raise FileNotFoundError(f"Missing configuration file: {cfg}")
        return {}
    with cfg.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return {key: Path(str(value)).expanduser() for key, value in data.items()}


def get_path(name: str, default: str | None = None, *, required: bool = False) -> Path | None:
    paths = load_paths(required=required)
    if name in paths:
        return paths[name]
    if default is not None:
        return Path(default).expanduser()
    if required:
        raise KeyError(f"Required path '{name}' not found in config/paths.yaml")
    return None
