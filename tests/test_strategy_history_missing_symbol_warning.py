"""get_history/get_history_multi 对未登记 symbol 的静默 NaN 补告警.

核心防线: 必须严格区分两种成因的 NaN --

- ``arr is None``(该 symbol 在历史缓冲中完全没有记录, 通常是配置错误)
  ->首次 WARNING 点名, 之后同 (symbol, field) 降 DEBUG。
- ``len(arr) < count``(有数据但不够长, 预热不足)-> 正常语义, 绝不告警,
  否则会在回测/实盘预热期每根 bar 刷屏。

用最小 ctx/strategy 替身直接单测 ``strategy_history`` 里的两个函数, 不经
run_backtest/Rust 引擎, 与 ``gateway/broker_event_bridge.py`` 里对
``_log_foreign_symbol`` 的单测风格一致。
"""

import logging
from typing import Optional

import numpy as np
import pytest
from akquant import strategy_history

LOGGER_NAME = "akquant.strategy"


class _FakeCtx:
    """最小 ctx 替身: 按 (symbol, field) 查表返回历史数组或 None."""

    def __init__(self, table: dict[tuple[str, str], np.ndarray]) -> None:
        """记录 (symbol, field) -> 历史数组的查表."""
        self._table = table

    def history(
        self,
        symbol: str,
        field: str,
        count: int,
        end_before_ns: Optional[int],
        freq: Optional[str],
    ) -> Optional[np.ndarray]:
        """按 (symbol, field) 查表; 未登记的组合返回 None(模拟 Rust 侧行为)."""
        return self._table.get((symbol, field))

    def history_multi(
        self,
        symbol: str,
        fields: tuple[str, ...],
        count: int,
        end_before_ns: Optional[int],
        freq: Optional[str],
    ) -> Optional[dict[str, np.ndarray]]:
        """Symbol 在任何字段上都无记录时整体返回 None, 否则按字段查表."""
        if not any((symbol, field) in self._table for field in fields):
            return None
        return {
            field: self._table[(symbol, field)]
            for field in fields
            if (symbol, field) in self._table
        }


class _FakeStrategy:
    """最小策略替身: 只提供 strategy_history 依赖的三个属性."""

    def __init__(self, ctx: _FakeCtx, history_depth: int = 10) -> None:
        """绑定 ctx 并设好非零 history_depth(否则会先炸 RuntimeError)."""
        self.ctx = ctx
        self._history_depth = history_depth

    def _resolve_symbol(self, symbol: Optional[str]) -> str:
        """本替身不需要默认标的推断, 显式传入的 symbol 原样返回."""
        assert symbol is not None
        return symbol


def test_get_history_missing_symbol_warns_with_symbol_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """未登记 symbol(``arr is None``)首次调用必须 WARNING 且点名该 symbol."""
    strategy = _FakeStrategy(_FakeCtx({}))

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        result = strategy_history.get_history(
            strategy, count=3, symbol="600016", field="close"
        )

    assert np.isnan(result).all()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "600016" in warnings[0].getMessage()


def test_get_history_missing_symbol_second_call_downgrades_to_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """同一 (symbol, field) 第二次调用不再是 WARNING(防刷屏)."""
    strategy = _FakeStrategy(_FakeCtx({}))

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        strategy_history.get_history(strategy, count=3, symbol="600016", field="close")
        caplog.clear()
        strategy_history.get_history(strategy, count=3, symbol="600016", field="close")

    assert not any(r.levelno == logging.WARNING for r in caplog.records)
    assert any(r.levelno == logging.DEBUG for r in caplog.records)


def test_get_history_warmup_shortfall_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``len(arr) < count``(预热不足)是正常语义, 绝不能告警(防刷屏核心防线)."""
    strategy = _FakeStrategy(_FakeCtx({("X", "close"): np.array([1.0, 2.0])}))

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        result = strategy_history.get_history(
            strategy, count=5, symbol="X", field="close"
        )

    assert np.isnan(result[:3]).all()
    assert list(result[3:]) == [1.0, 2.0]
    assert len(caplog.records) == 0


def test_get_history_multi_missing_symbol_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """get_history_multi 的同款分支: 未登记 symbol 逐字段告警且点名."""
    strategy = _FakeStrategy(_FakeCtx({}))

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        out = strategy_history.get_history_multi(
            strategy, count=3, symbol="600016", fields=("close", "open")
        )

    assert np.isnan(out["close"]).all()
    assert np.isnan(out["open"]).all()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert all("600016" in r.getMessage() for r in warnings)


def test_get_history_multi_warmup_shortfall_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """get_history_multi 里"有数据但不够长"的分支同样不能告警."""
    strategy = _FakeStrategy(
        _FakeCtx(
            {
                ("X", "close"): np.array([1.0, 2.0]),
                ("X", "open"): np.array([3.0, 4.0]),
            }
        )
    )

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        out = strategy_history.get_history_multi(
            strategy, count=5, symbol="X", fields=("close", "open")
        )

    assert np.isnan(out["close"][:3]).all()
    assert list(out["close"][3:]) == [1.0, 2.0]
    assert len(caplog.records) == 0


def test_get_history_multi_mixed_missing_and_shortfall_only_warns_for_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """混合场景: 同一次调用里, 缺失字段告警而预热不足字段保持沉默.

    field='close' 完全未登记(应告警), field='open' 有数据但不够长(不该告警),
    两者不能互相污染对方的判定。
    """
    strategy = _FakeStrategy(_FakeCtx({("X", "open"): np.array([3.0, 4.0])}))

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        out = strategy_history.get_history_multi(
            strategy, count=5, symbol="X", fields=("close", "open")
        )

    assert np.isnan(out["close"]).all()
    assert np.isnan(out["open"][:3]).all()
    assert list(out["open"][3:]) == [3.0, 4.0]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "close" in warnings[0].getMessage()
