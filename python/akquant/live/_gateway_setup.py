"""Stateless helpers for live gateway option handling and thread startup.

Extracted from the live runner for direct unit testing and reuse by the
``run_live`` facade. These functions hold no runner state.
"""

import threading
from typing import Any


def start_gateway_thread(target: Any, name: str) -> threading.Thread:
    """Start ``target`` on a named daemon thread and return it."""
    thread = threading.Thread(target=target, name=name, daemon=True)
    thread.start()
    return thread
