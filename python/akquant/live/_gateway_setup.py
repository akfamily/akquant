"""Stateless helpers for live gateway option handling and thread startup.

Extracted from the live runner for direct unit testing and reuse by the
``run_live`` facade. These functions hold no runner state.
"""

import threading
from typing import Any, Dict, Optional

# Legacy top-level broker connection args folded into ``gateway_options``.
LEGACY_GATEWAY_OPTION_KEYS = (
    "md_front",
    "td_front",
    "broker_id",
    "user_id",
    "password",
    "app_id",
    "auth_code",
)


def normalize_gateway_options(
    gateway_options: Optional[Dict[str, Any]],
    **legacy_values: Any,
) -> Dict[str, Any]:
    """Fold legacy top-level connection args into a ``gateway_options`` dict.

    Explicit ``gateway_options`` entries win over legacy args; empty legacy
    values are ignored.
    """
    normalized = dict(gateway_options or {})
    for key in LEGACY_GATEWAY_OPTION_KEYS:
        if key in normalized:
            continue
        value = legacy_values.get(key)
        if value is None or value == "":
            continue
        normalized[key] = value
    return normalized


def start_gateway_thread(target: Any, name: str) -> threading.Thread:
    """Start ``target`` on a named daemon thread and return it."""
    thread = threading.Thread(target=target, name=name, daemon=True)
    thread.start()
    return thread
