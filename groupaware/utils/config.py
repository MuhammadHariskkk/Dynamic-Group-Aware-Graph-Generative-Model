"""YAML config loading and override utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively update nested dictionaries."""
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _parse_scalar(value: str) -> Any:
    """Parse scalar value from CLI-style override strings."""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    if value.startswith("[") and value.endswith("]"):
        parsed = yaml.safe_load(value)
        if isinstance(parsed, list):
            return parsed
    return value


def _set_by_dotpath(config: dict[str, Any], dotpath: str, value: Any) -> None:
    """Set nested dict value via dot-separated keys."""
    keys = dotpath.split(".")
    cursor = config
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply dictionary-based overrides to a config object."""
    if not overrides:
        return config
    return _deep_update(config, overrides)


def apply_cli_overrides(config: dict[str, Any], cli_overrides: list[str] | None = None) -> dict[str, Any]:
    """
    Apply overrides of form ['trainer.lr=0.001', 'model.mlp_dims=[16,512,8]'].
    """
    if not cli_overrides:
        return config

    updated = deepcopy(config)
    for item in cli_overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected key=value.")
        key, raw_value = item.split("=", 1)
        _set_by_dotpath(updated, key.strip(), _parse_scalar(raw_value.strip()))
    return updated


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as dictionary."""
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at top level: {yaml_path}")
    return data


def load_config(
    config_path: str | Path,
    base_config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    cli_overrides: list[str] | None = None,
) -> dict[str, Any]:
    """
    Load configuration with optional base + overrides merge strategy.

    Notes:
    - `base_config_path` is an engineering convenience for scene-specific config inheritance.
    - Scene configs in this repository can also use explicit YAML anchors if preferred.
    """
    config = load_yaml(base_config_path) if base_config_path else {}
    config = _deep_update(config, load_yaml(config_path))
    config = apply_overrides(config, overrides)
    config = apply_cli_overrides(config, cli_overrides)
    return config
