"""symbols 语义: 传了就只跑这些标的, 不传则沿用「数据即订阅」.

设计见 docs/superpowers/specs/2026-08-14-symbols-filtering-design.md
"""

from typing import Any

import akquant as aq
import pytest


class _RecordingStrategy(aq.Strategy):
    """记录每个标的的 on_bar 触发次数."""

    def on_start(self) -> None:
        """初始化命中表."""
        self.hits: dict[str, int] = {}

    def on_bar(self, bar: aq.Bar) -> None:
        """累计该标的的触发次数."""
        self.hits[bar.symbol] = self.hits.get(bar.symbol, 0) + 1


def _bars() -> list[aq.Bar]:
    """X 与 Y 各 3 根 bar."""
    out = []
    for i in range(3):
        for symbol, base in (("X", 10.0), ("Y", 100.0)):
            out.append(
                aq.Bar(
                    timestamp=f"2025-01-{2 + i:02d}",  # type: ignore[arg-type]
                    symbol=symbol,
                    open=base + i,
                    high=base + i,
                    low=base + i,
                    close=base + i,
                    volume=100.0,
                )
            )
    return out


def _feed() -> Any:
    """同样的数据装进 DataFeed —— 走 Rust 层过滤这条路径."""
    feed = aq.DataFeed()
    for bar in _bars():
        feed.add_bar(bar)
    feed.sort()
    return feed


def _run(data: Any, **kwargs: Any) -> dict[str, int]:
    strategy = _RecordingStrategy()
    aq.run_backtest(strategy=strategy, data=data, initial_cash=100000, **kwargs)
    return strategy.hits


def test_explicit_symbols_filters_list_of_bars() -> None:
    """传了 symbols=['X'] 时 Y 不得参与(List[Bar] 形态)."""
    assert _run(_bars(), symbols=["X"]) == {"X": 3}


def test_explicit_symbols_filters_data_feed() -> None:
    """DataFeed 形态同样被过滤 —— 这条只能靠 Rust 层白名单."""
    assert _run(_feed(), symbols=["X"]) == {"X": 3}


def test_omitting_symbols_keeps_data_as_subscription() -> None:
    """不传 symbols 时沿用「数据即订阅」, 两个标的都跑(回归底线)."""
    assert _run(_bars()) == {"X": 3, "Y": 3}


def test_empty_symbols_list_is_rejected() -> None:
    """显式传空列表是参数错误, 必须报错而非静默跑出空回测."""
    with pytest.raises(ValueError, match="symbols"):
        _run(_bars(), symbols=[])


def _bars_with_benchmark_symbol() -> list[aq.Bar]:
    """标的代码分别为 BENCHMARK 与 OTHER, 各 3 根 bar.

    BENCHMARK 同时是本文件内部代表"未指定标的"的哨兵字面量, 也是这里用作
    真实标的代码的测试数据 —— 二者刚好撞了同一个字符串, 这正是要验证的场景。
    """
    out = []
    for i in range(3):
        for symbol, base in (("BENCHMARK", 10.0), ("OTHER", 100.0)):
            out.append(
                aq.Bar(
                    timestamp=f"2025-01-{2 + i:02d}",  # type: ignore[arg-type]
                    symbol=symbol,
                    open=base + i,
                    high=base + i,
                    low=base + i,
                    close=base + i,
                    volume=100.0,
                )
            )
    return out


def test_explicit_symbols_literal_benchmark_still_filters() -> None:
    """symbols=["BENCHMARK"] 显式传入时必须真的过滤掉 "OTHER".

    回归护栏: 曾经的实现把"显式传入的 BENCHMARK"错误折算成"未显式传入",
    导致这种情况下过滤永远不生效、"OTHER" 会被静默一并跑出。
    """
    assert _run(_bars_with_benchmark_symbol(), symbols=["BENCHMARK"]) == {
        "BENCHMARK": 3
    }


def test_all_input_forms_agree_under_same_symbols() -> None:
    """同一份数据、同一个 symbols, 各输入形态的命中集合必须一致.

    这是本次变更的核心保证: 此前 DataFrame 形态与 DataFeed 形态在
    symbols 上的行为并不一致, 而不一致本身就是要修的缺陷。
    """
    import pandas as pd

    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [f"2025-01-{2 + i:02d}" for i in range(3) for _ in range(2)]
            ),
            "symbol": ["X", "Y"] * 3,
            "open": [10.0, 100.0] * 3,
            "high": [10.0, 100.0] * 3,
            "low": [10.0, 100.0] * 3,
            "close": [10.0, 100.0] * 3,
            "volume": [100.0, 100.0] * 3,
        }
    )
    expected = {"X": 3}
    assert _run(_bars(), symbols=["X"]) == expected
    assert _run(_feed(), symbols=["X"]) == expected
    assert _run(frame, symbols=["X"]) == expected
    assert _run({"X": frame[frame["symbol"] == "X"]}, symbols=["X"]) == expected


def test_filtered_symbols_are_logged_once_in_summary(caplog: Any) -> None:
    """被过滤掉的标的只发一条汇总日志, 不逐个刷屏.

    传全市场数据只关心几个标的是本变更的主要动机场景, 逐标的告警会淹没输出。
    """
    with caplog.at_level("INFO"):
        _run(_bars(), symbols=["X"])
    filtered_lines = [r for r in caplog.records if "过滤" in r.getMessage()]
    assert len(filtered_lines) == 1
    assert "1" in filtered_lines[0].getMessage()


def test_dataframe_and_dict_forms_also_log_filtered_summary(caplog: Any) -> None:
    """DataFrame 与 dict 形态的既有过滤同样要发汇总日志(fix round 1, finding a).

    此前只有新增的 List[Bar] 段会写 filtered_out_symbols, DataFrame/dict 两段
    的既有过滤代码从未写入这个集合 —— 导致"传全市场数据只关心几个标的"这个
    主要动机场景下, 用 DataFrame 或 dict(DataFrame 是最常见的输入形态)传入
    时完全没有汇总日志, 只有 List[Bar] 才有。
    """
    import pandas as pd

    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [f"2025-01-{2 + i:02d}" for i in range(3) for _ in range(2)]
            ),
            "symbol": ["X", "Y"] * 3,
            "open": [10.0, 100.0] * 3,
            "high": [10.0, 100.0] * 3,
            "low": [10.0, 100.0] * 3,
            "close": [10.0, 100.0] * 3,
            "volume": [100.0, 100.0] * 3,
        }
    )

    with caplog.at_level("INFO"):
        _run(frame, symbols=["X"])
    df_lines = [r for r in caplog.records if "过滤" in r.getMessage()]
    assert len(df_lines) == 1
    assert "Y" in df_lines[0].getMessage()

    caplog.clear()

    data_dict = {
        "X": frame[frame["symbol"] == "X"],
        "Y": frame[frame["symbol"] == "Y"],
    }
    with caplog.at_level("INFO"):
        _run(data_dict, symbols=["X"])
    dict_lines = [r for r in caplog.records if "过滤" in r.getMessage()]
    assert len(dict_lines) == 1
    assert "Y" in dict_lines[0].getMessage()


class _LeakyAdapter:
    """故意在响应里混入未被请求的标的, 模拟违反 DataFeedAdapter 契约的实现.

    仅需一个可调用的 `load` 属性即满足 `_is_data_feed_adapter` 的 duck-type
    判据(hasattr(value, "load") and callable(...)), 无需继承任何基类。
    """

    def load(self, request: Any) -> Any:
        """无视 request.symbol, 总是额外搭售一份 'LEAK' 标的的数据."""
        import pandas as pd

        rows = []
        for symbol, base in ((str(request.symbol), 10.0), ("LEAK", 999.0)):
            for i in range(3):
                rows.append(
                    {
                        "timestamp": pd.Timestamp(f"2025-01-{2 + i:02d}"),
                        "symbol": symbol,
                        "open": base + i,
                        "high": base + i,
                        "low": base + i,
                        "close": base + i,
                        "volume": 100.0,
                    }
                )
        return pd.DataFrame(rows).set_index("timestamp")


def test_adapter_leaked_symbol_does_not_pollute_whitelist(caplog: Any) -> None:
    """Adapter 违反契约返回未请求的标的时, 该标的不得进白名单(fix round 1, finding b).

    调用点会把 adapter 返回的 data_map 里每个 key 无条件 append 进 symbols,
    而 symbols 正是随后设进 Rust 引擎 set_symbol_whitelist 的同一个列表 ——
    不做处理的话, 过滤会对这个泄漏进来的标的静默失效(白名单本身被污染)。
    还要求发一条 warning(而不是混进 filtered_out_symbols 那条 INFO 汇总),
    因为这是 adapter 违反契约, 不是用户主动排除。
    """
    with caplog.at_level("WARNING"):
        hits = _run(_LeakyAdapter(), symbols=["X"])
    assert "LEAK" not in hits
    assert hits == {"X": 3}
    warning_lines = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("LEAK" in r.getMessage() for r in warning_lines)
    assert not any("过滤" in r.getMessage() for r in warning_lines)


def test_subscribe_outside_whitelist_raises() -> None:
    """传了 symbols 后, on_start 里 subscribe 白名单外的标的必须报错.

    时序上无法自动并入: Engine::run(src/engine/python.rs:1137) 内部先调
    on_start(:1151), 而数据加载与 add_data 发生在 run() **之前** —— 前置过滤
    执行时 _subscriptions 还是空的。且「声明只跑 X, 又订阅 Y」本身自相矛盾,
    报错比静默择一更清晰。
    """

    class _SubscribingStrategy(_RecordingStrategy):
        def on_start(self) -> None:
            """订阅一个白名单外的标的."""
            super().on_start()
            self.subscribe("Y")

    strategy = _SubscribingStrategy()
    with pytest.raises(ValueError, match="Y"):
        aq.run_backtest(
            strategy=strategy, data=_bars(), symbols=["X"], initial_cash=100000
        )


def test_subscribe_inside_whitelist_is_fine() -> None:
    """Subscribe 白名单内的标的不受影响."""

    class _SubscribingStrategy(_RecordingStrategy):
        def on_start(self) -> None:
            """Subscribe 一个白名单内的标的."""
            super().on_start()
            self.subscribe("X")

    strategy = _SubscribingStrategy()
    aq.run_backtest(strategy=strategy, data=_bars(), symbols=["X"], initial_cash=100000)
    assert strategy.hits == {"X": 3}


def test_subscribe_without_explicit_symbols_never_raises() -> None:
    """不传 symbols 时 subscribe 不受任何约束(回归底线)."""

    class _SubscribingStrategy(_RecordingStrategy):
        def on_start(self) -> None:
            """订阅任意标的."""
            super().on_start()
            self.subscribe("Z")

    strategy = _SubscribingStrategy()
    aq.run_backtest(strategy=strategy, data=_bars(), initial_cash=100000)
    assert strategy.hits == {"X": 3, "Y": 3}


def test_slot_strategy_subscribe_outside_whitelist_raises() -> None:
    """多策略 slot 拓扑下, slot 策略在 on_start 里 subscribe 白名单外的标的同样报错.

    两处白名单下发(run_backtest / run_from_checkpoint)都对
    slot_strategy_instances.values() 逐一设置 _symbol_whitelist —— 只测主策略无法
    锁住这条 for 循环: 若未来有人误删它, 校验会对 slot 策略静默失效, 但只测主策略的
    用例不会报警。
    """

    class _SubscribingSlotStrategy(_RecordingStrategy):
        def on_start(self) -> None:
            """Slot 策略订阅一个白名单外的标的."""
            super().on_start()
            self.subscribe("Y")

    with pytest.raises(ValueError, match="Y"):
        aq.run_backtest(
            strategy=_RecordingStrategy,
            data=_bars(),
            symbols=["X"],
            initial_cash=100000,
            strategies_by_slot={"beta": _SubscribingSlotStrategy},
        )


class _SubscribingWarmStartStrategy(_RecordingStrategy):
    """恢复后订阅一个白名单外的标的; 首次启动时不订阅.

    定义在模块顶层(而非测试函数内部)是必须的: save_checkpoint 要 pickle 策略
    实例, 而 pickle 无法处理定义在函数局部作用域里的类
    (`Can't get local object '...<locals>....'`)。
    """

    def on_start(self) -> None:
        """恢复后订阅一个白名单外的标的; 首次启动时不订阅."""
        super().on_start()
        if self.is_restored:
            self.subscribe("Y")


def test_checkpoint_resume_subscribe_outside_whitelist_raises(tmp_path: Any) -> None:
    """热启动(run_from_checkpoint)下, 恢复后 on_start 里 subscribe 白名单外标的同样报错.

    run_from_checkpoint 的白名单下发是与 run_backtest 结构对称但物理独立的一段代码
    (engine.py 里两处不同的 if symbols_explicit 分支), 只测 run_backtest 无法覆盖它。

    阶段一刻意**不传** symbols(symbols_explicit=False), 使 `_symbol_whitelist`
    在存档前后都是 None —— 若阶段一也传 symbols, pickle 会把阶段一算出的白名单
    原样带进恢复后的实例, 阶段二哪怕真的漏发白名单, 测试也会"意外"通过(靠的是
    阶段一残留状态, 不是阶段二的下发代码), 起不到锁住 checkpoint 路径本身的作用。
    阶段二显式传 symbols=["X"], 恢复后的白名单只能来自 run_from_checkpoint 自己
    的下发代码。

    on_start 在恢复态下会被再次调用(见 Strategy.on_start 文档: "如果策略是从快照
    恢复的, 此方法仍会被调用"), 所以用 self.is_restored 把 subscribe("Y") 限定在
    恢复之后触发, 避免阶段一就提前报错。
    """
    checkpoint_path = tmp_path / "symbols_whitelist_checkpoint.pkl"
    phase1 = aq.run_backtest(
        strategy=_SubscribingWarmStartStrategy,
        data=_bars(),
        initial_cash=100000,
    )
    aq.save_checkpoint(phase1.engine, phase1.strategy, str(checkpoint_path))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Y"):
        aq.run_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            data=_bars(),
            symbols=["X"],
        )


def _later_bars() -> list[aq.Bar]:
    """X 与 Y 各 3 根 bar, 日期比 _bars() 晚(供热启动阶段二使用, 避免与阶段一重叠)."""
    out = []
    for i in range(3):
        for symbol, base in (("X", 10.0), ("Y", 100.0)):
            out.append(
                aq.Bar(
                    timestamp=f"2025-01-{5 + i:02d}",  # type: ignore[arg-type]
                    symbol=symbol,
                    open=base + i,
                    high=base + i,
                    low=base + i,
                    close=base + i,
                    volume=100.0,
                )
            )
    return out


class _SubscribingAfterUnfilteredResumeStrategy(_RecordingStrategy):
    """恢复时订阅一个"上一段 checkpoint 的白名单"之外的标的.

    on_resume 早于 on_start(见 Strategy._on_start_internal), 在此订阅是为了让
    校验判据(若残留生效)在 on_start 重置 hits 之前就先抛出, 与
    _SubscribingWarmStartStrategy 用 on_start+is_restored 是等价时机, 换成
    on_resume 只是顺带覆盖这个另外存在的钩子。
    """

    def on_resume(self) -> None:
        """恢复时订阅一个曾经的白名单外标的."""
        self.subscribe("Y")


def test_checkpoint_resume_without_symbols_does_not_inherit_stale_whitelist(
    tmp_path: Any,
) -> None:
    """阶段一传了 symbols, 阶段二热启动**不再**传 symbols 时, 旧白名单不能残留.

    `_symbol_whitelist` 是策略实例的普通属性, 会随对象一起被 `save_checkpoint`
    pickle 下来; `load_checkpoint` 用默认 `__dict__` 整体恢复, 会把这个旧值原样
    带回来。若白名单下发代码只在 `symbols_explicit` 为真时才赋值, 阶段二不传
    `symbols` 时: 引擎层(`engine.set_symbol_whitelist`)因为同一个判据不执行,
    正确地"不过滤"; 但策略层的 `_symbol_whitelist` 却仍是阶段一的旧集合 ——
    引擎放行、策略却按旧白名单拦截 subscribe, 出现自相矛盾的两副面孔, 且报错
    文案里的白名单是这次调用根本没传的值。

    这里刻意复用了 fix round 1 里"阶段一也传 symbols"这个构造——上一版测试要
    刻意避开它(否则测不到 run_from_checkpoint 自己的下发代码), 这次反过来专门
    用它验证残留不会生效: 恢复后 subscribe 旧白名单外的标的必须**不报错**, 且
    该标的能正常参与回测(引擎层本就不过滤)。
    """
    checkpoint_path = tmp_path / "stale_whitelist_checkpoint.pkl"
    phase1 = aq.run_backtest(
        strategy=_SubscribingAfterUnfilteredResumeStrategy,
        data=_bars(),
        symbols=["X"],
        initial_cash=100000,
    )
    aq.save_checkpoint(phase1.engine, phase1.strategy, str(checkpoint_path))  # type: ignore[arg-type]

    result = aq.run_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        data=_later_bars(),
    )
    strategy = result.strategy
    assert strategy is not None
    assert strategy.hits.get("Y", 0) > 0


def test_symbol_with_no_data_warns(caplog: Any) -> None:
    """白名单里的标的全程没有数据时必须告警(多为标的代码写错)."""
    with caplog.at_level("WARNING"):
        _run(_bars(), symbols=["X", "NOSUCH"])
    messages = [r.getMessage() for r in caplog.records]
    assert any("NOSUCH" in m for m in messages)
    assert not any("'X'" in m and "没有" in m for m in messages)


def test_symbol_with_data_is_not_reported(caplog: Any) -> None:
    """有数据的标的不得被误报为零数据."""
    with caplog.at_level("WARNING"):
        _run(_bars(), symbols=["X"])
    assert not any(
        "X" in r.getMessage() and "没有" in r.getMessage() for r in caplog.records
    )


def test_data_feed_form_also_warns(caplog: Any) -> None:
    """DataFeed 形态无法预先枚举, 靠会话末兜底同样要报出来."""
    with caplog.at_level("WARNING"):
        _run(_feed(), symbols=["X", "NOSUCH"])
    assert any("NOSUCH" in r.getMessage() for r in caplog.records)
