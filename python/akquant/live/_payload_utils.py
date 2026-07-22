"""Stateless helpers for reading broker event payloads and status values.

Extracted from the live runner for direct unit testing and reuse by the
``run_live`` facade. These functions hold no runner state.
"""

from typing import Any, Dict

_TERMINAL_STATUSES = {"filled", "cancelled", "canceled", "rejected"}


def payload_field(payload: Any, field: str) -> Any:
    """Read ``field`` from a dict-like or attribute-bearing payload."""
    if isinstance(payload, dict):
        return payload.get(field, "")
    return getattr(payload, field, "")


def payload_to_dict(payload: Any) -> Dict[str, Any]:
    """Best-effort conversion of a broker payload into a plain dict."""
    if isinstance(payload, dict):
        return dict(payload)
    if hasattr(payload, "__dict__"):
        return dict(getattr(payload, "__dict__"))
    return {}


def is_terminal_status(status: Any) -> bool:
    """Return whether an order status represents a terminal state."""
    status_text = str(getattr(status, "value", status)).strip().lower()
    return status_text in _TERMINAL_STATUSES


def normalize_broker_recovery_mode(mode: Any) -> str:
    """Validate and normalize the broker recovery mode option."""
    normalized = str(mode or "compatible").strip().lower()
    if normalized not in {"compatible", "strict"}:
        raise ValueError(
            "gateway_options.recovery_mode must be 'compatible' or 'strict'"
        )
    return normalized
