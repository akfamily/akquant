"""symbols 语义: 传了就只跑这些标的, 不传则沿用「数据即订阅」.

设计见 docs/superpowers/specs/2026-08-14-symbols-filtering-design.md
"""

from typing import Any

import akquant as aq
import pytest
from akquant import IntParam


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


def _frame_xy() -> Any:
    """X 与 Y 各 3 根 bar 的 DataFrame 形态(与 `_bars()` 同一份数据)."""
    import pandas as pd

    return pd.DataFrame(
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


def _frame_benchmark_other() -> Any:
    """BENCHMARK 与 OTHER 各 3 根 bar 的 DataFrame 形态.

    与 `_bars_with_benchmark_symbol()` 同一份数据。
    """
    import pandas as pd

    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [f"2025-01-{2 + i:02d}" for i in range(3) for _ in range(2)]
            ),
            "symbol": ["BENCHMARK", "OTHER"] * 3,
            "open": [10.0, 100.0] * 3,
            "high": [10.0, 100.0] * 3,
            "low": [10.0, 100.0] * 3,
            "close": [10.0, 100.0] * 3,
            "volume": [100.0, 100.0] * 3,
        }
    )


def test_dataframe_symbols_with_benchmark_sentinel_still_filters_c1() -> None:
    """C1(全分支终审 Critical): DataFrame(含 X,Y) + symbols=['X','BENCHMARK'] 只放行 X.

    根因: 引擎白名单下发点(engine.py, `run_backtest` 内 `engine.set_symbol_whitelist`
    调用处)曾直接复用局部变量 `symbols`——DataFrame 分支在数据加载时会把从数据里
    检测到的标的(这里是 "Y")反向 append 进这个同名变量, 而 Python 前置过滤又因为
    "BENCHMARK" 出现在 symbols 里被既有判据(`"BENCHMARK" not in symbols`)短路跳过
    (这一层本身只是性能优化, 语义应由 Rust 层白名单兜底)。两者叠加的实测结果是
    {'X': 3, 'Y': 3}——"Y" 带着默认合约参数混进撮合, 正是本次改动要消灭的缺陷。
    修复后引擎白名单改用未被数据加载污染的
    `_merge_symbol_whitelist_sources(effective_symbols, ...)`, "Y" 不再进引擎。
    """
    assert _run(_frame_xy(), symbols=["X", "BENCHMARK"]) == {"X": 3}


def test_list_of_bars_symbols_with_benchmark_sentinel_is_the_reference_c1() -> None:
    """C1 对照组: List[Bar] 形态下同样的 symbols=['X','BENCHMARK'] 一直是对的.

    List[Bar] 分支从不把检测到的标的写回 `symbols` 变量, 不受 C1 根因污染,
    终审用它作为"正确"的参照组。这里锁住它, 确保修复 DataFrame 形态时没有
    反过来改坏这条本就正确的路径。
    """
    assert _run(_bars(), symbols=["X", "BENCHMARK"]) == {"X": 3}


def test_checkpoint_symbols_with_benchmark_sentinel_is_the_reference_c1(
    tmp_path: Any,
) -> None:
    """C1 对照组: run_from_checkpoint 同样的 symbols=['X','BENCHMARK'] 一直是对的.

    `run_from_checkpoint` 的白名单下发点已经在用
    `_merge_symbol_whitelist_sources` 重新计算(而非复用被数据加载污染的
    `symbols`), 是终审里唯一给出正确结果 {'X': 3} 的路径, 本次修复让
    `run_backtest` 与它对齐, 而不是反过来改坏它——这里锁住其结果不变。
    """
    checkpoint_path = tmp_path / "c1_reference_checkpoint.pkl"
    phase1 = aq.run_backtest(
        strategy=_RecordingStrategy,
        data=_bars(),
        initial_cash=100000,
    )
    aq.save_checkpoint(phase1.engine, phase1.strategy, str(checkpoint_path))  # type: ignore[arg-type]

    result = aq.run_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        data=_later_bars_x_only(),
        symbols=["X", "BENCHMARK"],
    )
    strategy = result.strategy
    assert strategy is not None
    assert strategy.hits == {"X": 3}


def test_dataframe_explicit_symbols_literal_benchmark_still_filters_c1() -> None:
    """C1 变体: DataFrame + symbols=['BENCHMARK'] 必须像 List[Bar] 一样真过滤.

    修复前, DataFrame 分支在 symbols 恰好等于 `["BENCHMARK"]` 时(engine.py 里
    `if not symbols or symbols == ["BENCHMARK"]:` 分支)会把 `symbols` 整体
    替换成数据里检测到的全部标的, 当成"未指定, 不过滤"处理——实测结果是
    "OTHER" 被放行照跑, 与 List[Bar] 形态下的既有回归护栏
    `test_explicit_symbols_literal_benchmark_still_filters` 行为不一致。
    修复后两种形态必须给出同样"只放行字面量为 BENCHMARK 的标的"的结果。
    """
    assert _run(_frame_benchmark_other(), symbols=["BENCHMARK"]) == {"BENCHMARK": 3}


def test_init_subscribe_of_benchmark_sentinel_does_not_disable_filtering_c1() -> None:
    """C1 变体: symbols=['X'] 时 __init__ 里 subscribe('BENCHMARK') 不得让过滤整体失效.

    根因: 合并后的白名单(effective_symbols ∪ config.instruments ∪
    `_subscriptions`)一旦含有字面量 "BENCHMARK"(哪怕只是因为策略主动订阅了
    这个真实标的, 与"未传 symbols"的哨兵值撞了同一个字符串), 就会让 Python
    前置过滤判据(`"BENCHMARK" not in symbols`)短路——这一层短路本身是既有
    代码、本轮不改语义(见任务说明), 语义应由 Rust 层白名单兜底。

    但修复前, DataFrame 分支在前置过滤判据短路后走的是"未过滤"分支
    (`if not symbols or symbols == ["BENCHMARK"]:` 为假, 进入 else 分支逐个
    append 检测到的标的), 会把数据里检测到的 "Y" 反向追加进这个已经含
    "BENCHMARK" 的 `symbols` 变量, 而这个被两次污染的变量正是引擎白名单
    下发点直接复用的那个——"Y" 就此混入引擎, 过滤对 DataFrame 形态整体失效。
    必须用 DataFrame(而非 List[Bar])形态才能复现: List[Bar] 分支不做这种
    "反向追加检测到的标的"的动作, 不受此叠加效应影响。
    """

    class _InitSubscribingStrategy(_RecordingStrategy):
        def __init__(self) -> None:
            """在 __init__ 阶段订阅哨兵字面量 "BENCHMARK"."""
            super().__init__()
            self.subscribe("BENCHMARK")

    strategy = _InitSubscribingStrategy()
    aq.run_backtest(
        strategy=strategy, data=_frame_xy(), symbols=["X"], initial_cash=100000
    )
    assert strategy.hits == {"X": 3}


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
    # 用实际告警模板("标的 X ...")而非松散的子串组合去精确排除误报——
    # 原断言 "'X'" in m 是死断言(告警模板不带引号, 永远不匹配), 甄别力为零。
    assert not any("标的 X " in m for m in messages)


def test_symbol_with_data_is_not_reported(caplog: Any) -> None:
    """有数据的标的不得被误报为零数据."""
    with caplog.at_level("WARNING"):
        _run(_bars(), symbols=["X"])
    assert not any("标的 X " in r.getMessage() for r in caplog.records)


def test_data_feed_form_also_warns(caplog: Any) -> None:
    """DataFeed 形态无法预先枚举, 靠会话末兜底同样要报出来."""
    with caplog.at_level("WARNING"):
        _run(_feed(), symbols=["X", "NOSUCH"])
    assert any("NOSUCH" in r.getMessage() for r in caplog.records)
    assert not any("标的 X " in r.getMessage() for r in caplog.records)


def test_symbols_as_string_does_not_split_into_characters() -> None:
    """Symbols 传字符串(受支持的写法)时, 白名单不能被按字符拆开(fix round 2, Critical).

    根因: 白名单下发点(engine.py, on_start 之前)曾直接对 `_resolve_effective_symbols`
    的第一个返回值(未归一的原始 `symbols`)取 `set(...)`——对字符串这等于按字符
    拆分, `set("BENCHMARK")` 会变成 `{'B','E','N','C','H','M','A','R','K'}` 这种
    坏集合。用策略在 `on_start` 里 subscribe 自己的标的本身来复现: 若白名单被拆
    成单字符集合, "BENCHMARK" 这个多字符标的必然不在其中, `subscribe()` 会把
    合法的自身标的错误地当成"白名单外"而抛 `ValueError`。修复后必须改用已归一
    的 `effective_symbols` 合并而来的白名单, 不再受原始入参形态影响。
    """

    class _SelfSubscribingStrategy(_RecordingStrategy):
        def on_start(self) -> None:
            """订阅 symbols 里传入的那个标的本身(用字符串形态传入)."""
            super().on_start()
            self.subscribe("BENCHMARK")

    strategy = _SelfSubscribingStrategy()
    aq.run_backtest(
        strategy=strategy,
        data=_bars_with_benchmark_symbol(),
        symbols="BENCHMARK",
        initial_cash=100000,
    )
    assert strategy.hits == {"BENCHMARK": 3}


def test_tick_only_symbol_is_not_misreported_as_zero_data(caplog: Any) -> None:
    """纯 tick、靠 on_tick 交易的标的不得被误报为零数据(fix round 2, Important ①).

    根因: 会话末兜底(`Strategy._check_symbol_data_coverage`)原先用
    `_symbol_bar_counts` 判定"该标的有没有出现过"——但这个计数器只在
    `on_bar_event`(strategy_events.py)里递增, `on_tick_event` 从不碰它。
    纯 tick 输入会整体转成 `DataFeed`(`data_map_for_indicators` 因此为空),
    使运行前比对失效、只剩会话末兜底兜底——而兜底本身对 tick-only 标的必然
    误判成"零数据", 直接打脸刚上线的 tick/bar 双流特性(tick-only 标的明明
    在正常靠 on_tick 交易)。修复后改用 `_last_prices`(bar/tick 两条路径都会
    写入)做判据, 此处验证误报不再出现, 且策略确实收到了全部 tick 并据此交易。
    """
    ticks = [
        aq.Tick(
            timestamp=f"2025-01-{2 + i:02d}",  # type: ignore[arg-type]
            price=10.0 + i,
            volume=100.0,
            symbol="X",
        )
        for i in range(3)
    ]

    class _TickTradingStrategy(aq.Strategy):
        def on_start(self) -> None:
            """初始化 tick 记录表."""
            self.tick_prices: list[float] = []

        def on_tick(self, tick: aq.Tick) -> None:
            """记录收到的 tick 价格 —— 代表策略确实靠 on_tick 在做决策/交易."""
            self.tick_prices.append(tick.price)

    strategy = _TickTradingStrategy()
    with caplog.at_level("WARNING"):
        aq.run_backtest(
            strategy=strategy,
            data=ticks,
            symbols=["X"],
            initial_cash=100000,
        )
    assert strategy.tick_prices == [10.0, 11.0, 12.0]
    assert not any("标的 X " in r.getMessage() for r in caplog.records)


class _SubscribingOnStartStrategy(_RecordingStrategy):
    """on_start 无条件订阅 "Y"(不受 is_restored 限制, 供 checkpoint 持久化用).

    定义在模块顶层是必须的: save_checkpoint 要 pickle 策略实例, pickle 无法
    处理定义在函数局部作用域里的类。
    """

    def on_start(self) -> None:
        """无条件订阅 "Y", 使其进入 `_subscriptions` 并随存档持久化."""
        super().on_start()
        self.subscribe("Y")


def _later_bars_x_only() -> list[aq.Bar]:
    """只有 X 的 3 根 bar, 日期比 _bars() 晚(供热启动阶段二使用, 且刻意不含 Y)."""
    out = []
    for i in range(3):
        out.append(
            aq.Bar(
                timestamp=f"2025-01-{5 + i:02d}",  # type: ignore[arg-type]
                symbol="X",
                open=10.0 + i,
                high=10.0 + i,
                low=10.0 + i,
                close=10.0 + i,
                volume=100.0,
            )
        )
    return out


def test_checkpoint_resume_zero_data_check_uses_merged_whitelist(
    tmp_path: Any, caplog: Any
) -> None:
    """run_from_checkpoint 的零数据核验要用「实际下发给引擎的白名单」.

    fix round 2, Important finding 2.

    阶段一**不传** symbols(symbols_explicit=False), on_start 里无条件
    subscribe("Y")——此时白名单为 None, 订阅不受限制、正常记入 `_subscriptions`,
    并随策略实例被 `save_checkpoint` 持久化下来。阶段二显式传 symbols=["X"]
    (不含 "Y"), 但引擎层实际下发的白名单是 `_merge_symbol_whitelist_sources`
    的产物: effective_symbols(["X"]) 并上持久化的 `_subscriptions`(含 "Y")
    = {"X", "Y"}——"Y" 正是靠这条路径混进白名单的。阶段二数据只含 X。

    修复前, 零数据比对用的是未合并的 `effective_symbols`(只有 "X"), "Y" 压根
    不在这个集合里、被漏检; 修复后改用同一份已合并的白名单去比对, "Y" 必须被
    正确报出零数据告警。
    """
    checkpoint_path = tmp_path / "merged_whitelist_checkpoint.pkl"
    phase1 = aq.run_backtest(
        strategy=_SubscribingOnStartStrategy,
        data=_bars(),
        initial_cash=100000,
    )
    aq.save_checkpoint(phase1.engine, phase1.strategy, str(checkpoint_path))  # type: ignore[arg-type]

    with caplog.at_level("WARNING"):
        aq.run_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            data=_later_bars_x_only(),
            symbols=["X"],
        )
    assert any("标的 Y " in r.getMessage() for r in caplog.records)


class _SubscribingOutOfDataStrategy(aq.Strategy):
    """I2(全分支终审 Important 2): on_start 里 subscribe 一个数据里不存在的标的.

    定义在模块顶层是必须的: run_grid_search/run_walk_forward 用进程池并行,
    策略类要能被 pickle。直接调用 run_backtest 且不传 symbols 时, 这类
    subscribe() 因为 `_symbol_whitelist` 未启用(为 None)而是无害 no-op——
    这正是本测试要在 run_grid_search / run_walk_forward 里守住的基线行为。
    """

    dummy = IntParam(0)

    def on_start(self) -> None:
        """订阅数据里不存在的标的 "OUT_OF_DATA"."""
        self.subscribe("OUT_OF_DATA")

    def on_bar(self, bar: aq.Bar) -> None:
        """No-op: 仅用于验证 symbols 未显式传入时的转发路径, 不关心交易逻辑."""
        return


def _i2_optimize_data() -> Any:
    """run_grid_search / run_walk_forward 用的最小单标的合成数据."""
    import numpy as np
    import pandas as pd

    n = 24
    dates = pd.date_range("2020-01-01", periods=n, freq="min", tz="UTC")
    price = np.full(n, 10.0)
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": np.full(n, 100.0),
            "symbol": "I2_DATA",
        }
    )


def test_run_grid_search_without_symbols_skips_subscribe_validation_i2() -> None:
    """I2: 不传 symbols 时 run_grid_search 不应触发过滤/subscribe 白名单校验.

    根因: `_resolve_optimization_backtest_kwargs` 曾在用户未传 symbol/symbols
    时, 仍会把从数据里推断出的值回填进转发给 run_backtest 的 kwargs, 使其被
    当成"显式传了 symbols"——策略在 on_start 里 subscribe 数据外的标的(直接
    调用 run_backtest 不传 symbols 时是无害 no-op)因此触发白名单校验报错,
    在 run_grid_search 的多进程执行里被吞成 error 列(sharpe/return = -999),
    而不是像直接 run_backtest 一样正常跑完。修复后不传 symbols 时行为必须与
    直接调用 run_backtest 完全一致(不触发校验)。
    """
    import pandas as pd

    results = aq.run_grid_search(
        strategy=_SubscribingOutOfDataStrategy,
        param_grid={"dummy": [1, 2]},
        data=_i2_optimize_data(),
        initial_cash=100_000.0,
        max_workers=1,
        show_progress=False,
    )
    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    assert "error" not in results.columns or results["error"].isna().all()
    assert (results["sharpe_ratio"] != -999).all()


def test_run_walk_forward_without_symbols_skips_subscribe_validation_i2() -> None:
    """I2: 不传 symbols 时 run_walk_forward 同样不应触发过滤/subscribe 白名单校验.

    与上一条同根同因, 覆盖 run_walk_forward 自身对样本外 run_backtest 的
    调用路径(不止 run_grid_search 内部的并行任务)。
    """
    import pandas as pd

    results = aq.run_walk_forward(
        strategy=_SubscribingOutOfDataStrategy,
        param_grid={"dummy": [1, 2]},
        data=_i2_optimize_data(),
        train_period=10,
        test_period=5,
        initial_cash=100_000.0,
        max_workers=1,
        show_progress=False,
    )
    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    assert "equity" in results.columns
