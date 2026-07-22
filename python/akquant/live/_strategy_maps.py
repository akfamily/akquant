"""Stateless helpers for normalizing and validating per-strategy risk maps.

Extracted from the live runner so they can be unit-tested directly and reused by
the ``run_live`` facade. These functions hold no runner state.
"""

from typing import Any, Dict, List, Optional


def normalize_strategy_float_map(
    values: Optional[Dict[str, float]],
) -> Dict[str, float]:
    """Coerce a per-strategy mapping into ``{str: float}``."""
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise TypeError("strategy map must be a dict when provided")
    normalized: Dict[str, float] = {}
    for key, value in values.items():
        key_str = str(key).strip()
        if not key_str:
            raise ValueError("strategy id cannot be empty")
        normalized[key_str] = float(value)
    return normalized


def normalize_strategy_int_map(values: Optional[Dict[str, int]]) -> Dict[str, int]:
    """Coerce a per-strategy mapping into ``{str: int}``."""
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise TypeError("strategy map must be a dict when provided")
    normalized: Dict[str, int] = {}
    for key, value in values.items():
        key_str = str(key).strip()
        if not key_str:
            raise ValueError("strategy id cannot be empty")
        normalized[key_str] = int(value)
    return normalized


def normalize_strategy_bool_map(values: Optional[Dict[str, bool]]) -> Dict[str, bool]:
    """Coerce a per-strategy mapping into ``{str: bool}``."""
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise TypeError("strategy map must be a dict when provided")
    normalized: Dict[str, bool] = {}
    for key, value in values.items():
        key_str = str(key).strip()
        if not key_str:
            raise ValueError("strategy id cannot be empty")
        normalized[key_str] = bool(value)
    return normalized


def validate_strategy_map_keys(
    values: Dict[str, Any], configured_slot_ids: List[str], field_name: str
) -> None:
    """Raise if a risk map references strategy ids outside the configured slots."""
    if not values:
        return
    unknown = sorted(set(values.keys()).difference(set(configured_slot_ids)))
    if unknown:
        unknown_text = ", ".join(unknown)
        raise ValueError(f"{field_name} contains unknown strategy ids: {unknown_text}")
