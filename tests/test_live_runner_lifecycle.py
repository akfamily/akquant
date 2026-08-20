"""实盘会话生命周期回调的补发测试.

与 ``test_live_runner.py`` 的区别: 那边全是 ``LiveRunner.__new__`` 绕过
``__init__`` 的装配级单测, 这里首次驱动完整的 ``LiveRunner.run()``,
验证收尾期回调真的被触发。
"""

from typing import Any

from akquant.strategy import Strategy


def test_on_stop_live_internal_skips_backtest_only_coverage_checks() -> None:
    """实盘收尾只补发结束钩子 + on_stop, 不跑三个回测语义的数据覆盖校验.

    三个校验(``_check_symbol_data_coverage`` / ``_check_warmup_symbol_coverage``
    / ``_check_incremental_hl_bar_coverage``)在实盘分别会误报、误报、以及
    **抛** ``StrategyConfigurationError`` 打断收尾 —— 盘中无成交标的、warmup
    未攒满、纯 tick 会话在实盘都是常态。
    """
    called: list[str] = []

    class Strat(Strategy):
        def on_bar(self, bar: Any) -> None:
            pass

        def on_stop(self) -> None:
            called.append("on_stop")

        def _check_symbol_data_coverage(self) -> None:
            called.append("symbol_data")

        def _check_warmup_symbol_coverage(self) -> None:
            called.append("warmup")

        def _check_incremental_hl_bar_coverage(self) -> None:
            called.append("hl_bar")

    strategy = Strat()
    strategy._on_stop_live_internal()

    assert called == ["on_stop"]


def test_on_stop_live_internal_is_idempotent() -> None:
    """重复派发实盘收尾不应重复执行用户 on_stop.

    ``_framework_stop_flushed`` 只护住 ``on_after_trading`` 的补发,
    护不住 ``on_stop`` 本身, 所以需要独立标志。
    """
    calls: list[int] = []

    class Strat(Strategy):
        def on_bar(self, bar: Any) -> None:
            pass

        def on_stop(self) -> None:
            calls.append(1)

    strategy = Strat()
    strategy._on_stop_live_internal()
    strategy._on_stop_live_internal()

    assert len(calls) == 1


def test_live_stop_flag_not_carried_by_checkpoint() -> None:
    """实盘收尾标志属会话内派生态: 不随存档带走, 续跑能再收尾一次."""

    class Strat(Strategy):
        def on_bar(self, bar: Any) -> None:
            pass

    strategy = Strat()
    strategy._on_stop_live_internal()
    assert strategy._framework_live_stop_dispatched is True

    state = strategy.__getstate__()
    assert "_framework_live_stop_dispatched" not in state
