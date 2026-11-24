"""Golden tests."""

import sys
from pathlib import Path

import pytest

# Add tests/golden to path
GOLDEN_DIR = Path(__file__).parent
sys.path.append(str(GOLDEN_DIR))

from runner import main as run_golden_suite  # noqa: E402
from runner import (  # noqa: E402
    run_simulated_tick_invariance_suite,
    run_stream_consistency_suite,
)


def test_golden_suite() -> None:
    """Run the Golden Test Suite.

    This ensures that current implementation matches the baselines.
    """
    # Capture SystemExit
    with pytest.raises(SystemExit) as excinfo:
        run_golden_suite(generate_baseline=False)

    assert excinfo.value.code == 0, "Golden Suite Failed"


def test_golden_stream_consistency_suite() -> None:
    """Run stream consistency checks on golden scenarios."""
    failures = run_stream_consistency_suite()
    assert not failures, f"Golden Stream Consistency Failed: {failures}"


def test_golden_simulated_tick_invariance_suite() -> None:
    """打开 simulate_ticks、on_tick 为空时, 六个场景与基线逐项相同."""
    failures = run_simulated_tick_invariance_suite()
    assert not failures, f"simulate_ticks 不变量被破坏: {failures}"
