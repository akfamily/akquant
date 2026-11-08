"""订单事件回调轨迹与基线(旧实现生成)逐条一致.

基线: tests/fixtures/order_event_traces.json, 由 tests/_order_event_scenarios.py
在订单账本改造前生成。轨迹不一致即回调种类、顺序、时机或数值发生了变化。
"""

import json

import pytest
from _order_event_scenarios import FIXTURE, SCENARIOS, run_scenario


def _baseline() -> dict:
    data: dict = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_trace_matches_baseline(name: str) -> None:
    """场景轨迹与基线一致."""
    assert run_scenario(name) == _baseline()[name]


def _kinds(trace: list) -> set:
    return {(row[0], row[3]) for row in trace if row[0] != "on_trade"}


def test_baseline_covers_target_behaviors() -> None:
    """基线本身覆盖到目标行为, 防止场景跑空(恒真)."""
    base = _baseline()
    assert ("on_order", "Cancelled") in _kinds(base["cancel_resubmit"])
    assert ("on_order", "PartiallyFilled") in _kinds(base["partial_fills"])
    assert ("on_reject", "Rejected") in _kinds(base["rejects"])
    assert ("on_order", "Expired") in _kinds(base["day_expiry"])
    assert any(row[0] == "on_trade" for row in base["bracket_oco"])
    assert base["multi_slot"]["alpha"] and base["multi_slot"]["beta"]
    assert len(base["random"]) > 50
    assert len(base["random_checkpoint"]) > 20
