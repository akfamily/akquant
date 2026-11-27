"""self.run_info: 调用方传入、策略只读的运行信息.

框架不解释其中的键, 只负责原样(深拷贝)、只读地交给策略; 各运行入口都接受同名参数。
"""

from __future__ import annotations

import logging
import math
import pickle
from collections.abc import Mapping
from typing import Any, Dict, List

import akquant as aq
import numpy as np
import pandas as pd
import pytest
from akquant import Bar, IntParam, Strategy, run_backtest

SYMBOL = "X"
INFO: Dict[str, Any] = {
    "user_id": "u-1",
    "mode": "backtest",
    "limits": {"max_lots": 3, "tags": ["a", "b"]},
    "flags": [1, 2.5, True, None],
}

#: 入口校验报错的开头: 指出 run_info 里哪一处、什么问题
BAD = r"^run_info(\[[^ ]*\])? (须为|的键必须是字符串|只接受|不接受|含循环引用)"


def _data(n: int = 3) -> pd.DataFrame:
    index = pd.date_range("2026-01-05", periods=n, freq="B")
    return pd.DataFrame(
        {
            "date": index,
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 1e7,
            "symbol": SYMBOL,
        }
    )


def _run(strategy: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("data", _data())
    return run_backtest(
        strategy=strategy, symbols=SYMBOL, show_progress=False, **kwargs
    )


class _Recorder(Strategy):
    """记录各回调里读到的 run_info."""

    def __init__(self) -> None:
        self.seen: Dict[str, Any] = {"init": self.run_info}
        self.bars: List[Any] = []

    def on_start(self) -> None:
        self.seen["start"] = self.run_info

    def on_bar(self, bar: Bar) -> None:
        self.bars.append(self.run_info)

    def on_stop(self) -> None:
        self.seen["stop"] = self.run_info


def _frozen_equal(frozen: Any, plain: Any) -> bool:
    """只读结构与原始 dict/list 内容一致(列表冻结成 tuple)."""
    if isinstance(plain, dict):
        return (
            isinstance(frozen, Mapping)
            and set(frozen) == set(plain)
            and all(_frozen_equal(frozen[k], v) for k, v in plain.items())
        )
    if isinstance(plain, list):
        return (
            isinstance(frozen, tuple)
            and len(frozen) == len(plain)
            and all(_frozen_equal(a, b) for a, b in zip(frozen, plain))
        )
    return type(frozen) is type(plain) and frozen == plain


# ---- 默认与可见性


def test_empty_by_default_and_in_init() -> None:
    """不传时为空映射; __init__ 里(还没交给引擎)也是空映射."""
    result = _run(_Recorder)
    strategy = result.strategy
    for key in ("init", "start", "stop"):
        assert isinstance(strategy.seen[key], Mapping)
        assert dict(strategy.seen[key]) == {}
    assert strategy.bars and all(dict(m) == {} for m in strategy.bars)


def test_visible_from_on_start_in_every_callback() -> None:
    """on_start、on_bar、on_stop 读到的都是传入的内容; __init__ 里仍为空."""
    result = _run(_Recorder, run_info=INFO)
    strategy = result.strategy
    assert dict(strategy.seen["init"]) == {}
    for key in ("start", "stop"):
        assert _frozen_equal(strategy.seen[key], INFO)
    assert len(strategy.bars) == 3
    assert all(_frozen_equal(m, INFO) for m in strategy.bars)


def test_instance_rerun_without_run_info_is_reset() -> None:
    """同一个实例再跑一次且不传 run_info: 读到空映射, 不残留上一次的值."""
    strategy = _Recorder()
    _run(strategy, run_info={"user_id": "first"})
    assert len(strategy.bars) == 3
    assert strategy.bars[-1]["user_id"] == "first"
    _run(strategy)
    assert len(strategy.bars) == 6
    assert all(dict(m) == {} for m in strategy.bars[3:])
    assert dict(strategy.run_info) == {}


# ---- 只读与深拷贝


class _Mutator(Strategy):
    """在 on_start 里尝试各种改法, 记录每一种的结果."""

    def __init__(self) -> None:
        self.outcomes: Dict[str, str] = {}

    def _attempt(self, label: str, action: Any) -> None:
        try:
            action()
        except (TypeError, AttributeError) as exc:
            self.outcomes[label] = type(exc).__name__
        else:
            self.outcomes[label] = "mutated"

    def on_start(self) -> None:
        info = self.run_info

        def set_item() -> None:
            info["user_id"] = "someone-else"  # type: ignore[index]

        def del_item() -> None:
            del info["user_id"]  # type: ignore[attr-defined]

        def reassign() -> None:
            self.run_info = {"user_id": "someone-else"}  # type: ignore[misc]

        def nested_set() -> None:
            info["limits"]["max_lots"] = 99

        def nested_list_append() -> None:
            info["limits"]["tags"].append("c")

        def nested_list_set() -> None:
            info["flags"][0] = 0

        self._attempt("set_item", set_item)
        self._attempt("del_item", del_item)
        self._attempt("reassign", reassign)
        self._attempt("nested_set", nested_set)
        self._attempt("nested_list_append", nested_list_append)
        self._attempt("nested_list_set", nested_list_set)
        self.after = self.run_info

    def on_bar(self, bar: Bar) -> None:
        return


def test_read_only_top_level_and_nested() -> None:
    """顶层与嵌套都改不了, 也不能整体赋值替换; 改的尝试过后内容不变."""
    result = _run(_Mutator, run_info=INFO)
    strategy = result.strategy
    assert strategy.outcomes == {
        "set_item": "TypeError",
        "del_item": "TypeError",
        "reassign": "AttributeError",
        "nested_set": "TypeError",
        "nested_list_append": "AttributeError",
        "nested_list_set": "TypeError",
    }
    assert _frozen_equal(strategy.after, INFO)


def test_reassign_error_message_points_to_entry_parameter() -> None:
    """赋值报错时说明 run_info 由运行入口传入."""
    strategy = _Recorder()
    with pytest.raises(AttributeError, match="run_info 是只读属性"):
        strategy.run_info = {}  # type: ignore[misc]


def test_deep_copy_caller_mutation_not_visible() -> None:
    """调用方传入之后再改自己的字典(含嵌套), 策略读到的不变."""
    caller: Dict[str, Any] = {"user_id": "u-1", "limits": {"tags": ["a"]}}

    class _MutateCaller(_Recorder):
        def on_bar(self, bar: Bar) -> None:
            caller["user_id"] = "changed"
            caller["limits"]["tags"].append("b")
            caller["new"] = 1
            super().on_bar(bar)

    result = _run(_MutateCaller, run_info=caller)
    expected = {"user_id": "u-1", "limits": {"tags": ["a"]}}
    assert all(_frozen_equal(m, expected) for m in result.strategy.bars)
    assert _frozen_equal(result.strategy.seen["stop"], expected)


class _Plain(Strategy):
    """不把 run_info 存进自己属性的策略."""

    def on_bar(self, bar: Bar) -> None:
        return


def _detached(strategy: Any) -> Any:
    """去掉引擎引用(同 save_checkpoint 的做法), 引擎对象本身不能 pickle."""
    for attr in ("_engine", "_analyzer_manager"):
        strategy.__dict__.pop(attr, None)
    return strategy


def test_strategy_pickle_drops_framework_run_info() -> None:
    """Pickle 策略时不带框架的 run_info; 恢复出来的实例读到空映射."""
    result = _run(_Plain, run_info={"user_id": "secret-user"})
    assert result.strategy.run_info["user_id"] == "secret-user"
    payload = pickle.dumps(_detached(result.strategy))
    assert b"secret-user" not in payload
    restored = pickle.loads(payload)
    assert dict(restored.run_info) == {}
    # pickle 不改原实例
    assert result.strategy.run_info["user_id"] == "secret-user"


def test_strategy_keeping_its_own_copy_still_pickles() -> None:
    """策略把 run_info(含嵌套)存进自己的属性后仍能 pickle, 内容按原样还原且仍只读."""
    result = _run(_Recorder, run_info=INFO)
    restored = pickle.loads(pickle.dumps(_detached(result.strategy)))
    kept = restored.seen["start"]
    assert _frozen_equal(kept, INFO)
    with pytest.raises(TypeError):
        kept["limits"]["max_lots"] = 0
    assert dict(restored.run_info) == {}


# ---- 取值校验


@pytest.mark.parametrize(
    ("bad", "error"),
    [
        (["not", "a", "mapping"], TypeError),
        ({1: "non-str key"}, TypeError),
        ({"k": object()}, TypeError),
        ({"k": {"a", "b"}}, TypeError),
        ({"k": b"bytes"}, TypeError),
        ({"k": {"nested": object()}}, TypeError),
        ({"k": {2: "non-str nested key"}}, TypeError),
        ({"k": [1, [2, object()]]}, TypeError),
        ({"k": np.int64(1)}, TypeError),
        ({"k": pd.Timestamp("2026-01-05")}, TypeError),
        ({"k": math.nan}, ValueError),
        ({"k": [math.inf]}, ValueError),
    ],
)
def test_invalid_values_rejected_before_run(bad: Any, error: type) -> None:
    """非简单类型(含 numpy 标量、非字符串键、NaN/inf)在入口处报错, 策略不启动."""

    class _Never(_Recorder):
        def on_start(self) -> None:
            raise AssertionError("must not start")

    with pytest.raises(error, match=BAD):
        _run(_Never, run_info=bad)


def test_str_subclass_rejected() -> None:
    """str/int 的子类不收: 子类可以带可变属性, 绕过只读."""

    class _Tagged(str):
        pass

    with pytest.raises(TypeError, match=BAD):
        _run(_Recorder, run_info={"user_id": _Tagged("u-1")})


def test_cyclic_value_rejected() -> None:
    """自引用的列表报 ValueError, 不递归到栈溢出."""
    loop: List[Any] = []
    loop.append(loop)
    with pytest.raises(ValueError, match=BAD):
        _run(_Recorder, run_info={"loop": loop})


def test_tuple_and_mapping_inputs_accepted() -> None:
    """Tuple 当列表收; 任意 Mapping(含另一个策略的 run_info)都可以传入."""
    first = _run(_Recorder, run_info={"pair": (1, 2), "m": {"a": 1}})
    info = first.strategy.seen["start"]
    assert info["pair"] == (1, 2)
    second = _run(_Recorder, run_info=info)
    assert dict(second.strategy.seen["start"]["m"]) == {"a": 1}
    assert second.strategy.seen["start"]["pair"] == (1, 2)


# ---- 与 context 的关系、撞名


def test_context_behaviour_unchanged_and_independent() -> None:
    """Context 仍逐项设成可写属性; 与 run_info 同名的键互不影响."""

    class _Ctx(_Recorder):
        def on_start(self) -> None:
            super().on_start()
            self.user_id_attr = getattr(self, "user_id")
            self.user_id = "rewritten"  # context 设的属性可写

    result = _run(
        _Ctx, context={"user_id": "from-context"}, run_info={"user_id": "from-host"}
    )
    strategy = result.strategy
    assert strategy.user_id_attr == "from-context"
    assert strategy.user_id == "rewritten"
    assert strategy.seen["start"]["user_id"] == "from-host"


def test_context_key_run_info_cannot_overwrite() -> None:
    """Context 里带 run_info 键不能把框架属性顶掉."""
    with pytest.raises(AttributeError, match="run_info 是只读属性"):
        _run(_Recorder, context={"run_info": {"user_id": "fake"}})


def test_subclass_defining_run_info_raises_at_class_definition() -> None:
    """子类(或经 mixin)自己定义 run_info 时在类定义处报错, 不静默遮住框架的值."""
    with pytest.raises(TypeError, match="run_info 是框架属性"):

        class _Own(Strategy):  # noqa: F841
            run_info = {"user_id": "hard-coded"}

    with pytest.raises(TypeError, match="run_info 是框架属性"):

        class _OwnMethod(Strategy):  # noqa: F841
            def run_info(self) -> Dict[str, Any]:  # type: ignore[override]
                return {}

    class _Mixin:
        run_info: Dict[str, Any] = {}

    with pytest.raises(TypeError, match="run_info 是框架属性"):

        class _Mixed(_Mixin, Strategy):  # type: ignore[misc]  # noqa: F841
            pass


# ---- 多槽位、指标脚本、函数式


def test_all_slots_see_the_same_mapping() -> None:
    """strategies_by_slot 的各槽位读到同一份内容."""
    main = _Recorder()
    other = _Recorder()
    _run(main, strategies_by_slot={"other": other}, run_info=INFO)
    assert _frozen_equal(main.seen["start"], INFO)
    assert _frozen_equal(other.seen["start"], INFO)
    assert other.bars and all(_frozen_equal(m, INFO) for m in other.bars)


class _ScriptProbe(aq.Indicator, title="probe"):
    """指标脚本: 在 on_start 里读 run_info."""

    seen: List[Any] = []

    def on_start(self) -> None:
        type(self).seen.append(self.run_info)


def test_indicator_script_sees_run_info() -> None:
    """indicators= 挂载的指标脚本同样可读."""
    _ScriptProbe.seen = []
    _run(_Recorder, indicators=[_ScriptProbe], run_info=INFO)
    assert len(_ScriptProbe.seen) == 1
    assert _frozen_equal(_ScriptProbe.seen[0], INFO)


def test_function_style_strategy_reads_ctx_run_info() -> None:
    """函数式策略经 ctx.run_info 读取: initialize、on_start 与 on_bar 都可读."""
    seen: Dict[str, List[Any]] = {"init": [], "start": [], "bar": []}

    def initialize(ctx: Any) -> None:
        seen["init"].append(ctx.run_info)

    def on_start(ctx: Any) -> None:
        seen["start"].append(ctx.run_info)

    def on_bar(ctx: Any, bar: Bar) -> None:
        seen["bar"].append(ctx.run_info)

    _run(on_bar, initialize=initialize, on_start=on_start, run_info=INFO)
    assert len(seen["init"]) == 1 and len(seen["start"]) == 1
    assert len(seen["bar"]) == 3
    for values in seen.values():
        assert all(_frozen_equal(m, INFO) for m in values)


# ---- 在 on_start 里拒绝运行


ALLOWED = {"u-1"}


class _Guarded(Strategy):
    """文档里的写法: 不在授权名单里就抛异常."""

    dummy = IntParam(0)

    def __init__(self) -> None:
        self.bar_count = 0

    def on_start(self) -> None:
        user = self.run_info.get("user_id")
        if user not in ALLOWED:
            raise PermissionError(f"user {user!r} is not allowed")

    def on_bar(self, bar: Bar) -> None:
        self.bar_count += 1


def test_raising_in_on_start_aborts_backtest_with_visible_error() -> None:
    """回测: on_start 里抛的异常原样抛给调用方, 一根 bar 都不跑."""
    strategy = _Guarded()
    with pytest.raises(PermissionError, match="'u-2' is not allowed"):
        _run(strategy, run_info={"user_id": "u-2"})
    assert strategy.bar_count == 0
    ok = _run(_Guarded, run_info={"user_id": "u-1"})
    assert ok.strategy.bar_count == 3


# ---- 网格搜索 / walk-forward


def test_grid_search_workers_receive_run_info() -> None:
    """并行网格搜索的子进程收到 run_info: 授权用户成功, 其他用户每组都报错."""
    allowed = aq.run_grid_search(
        strategy=_Guarded,
        param_grid={"dummy": [1, 2]},
        data=_data(5),
        symbols=SYMBOL,
        max_workers=2,
        return_df=True,
        show_progress=False,
        run_info={"user_id": "u-1"},
    )
    assert isinstance(allowed, pd.DataFrame)
    assert len(allowed) == 2
    assert "error" not in allowed.columns or allowed["error"].isna().all()

    denied = aq.run_grid_search(
        strategy=_Guarded,
        param_grid={"dummy": [1, 2]},
        data=_data(5),
        symbols=SYMBOL,
        max_workers=2,
        return_df=False,
        raise_on_all_failed=False,
        show_progress=False,
        run_info={"user_id": "u-2"},
    )
    assert isinstance(denied, list) and len(denied) == 2
    assert all("'u-2' is not allowed" in str(r.error) for r in denied)


def test_grid_search_rejects_invalid_run_info_before_spawning() -> None:
    """网格搜索在派发任务之前校验 run_info."""
    with pytest.raises(TypeError, match=BAD):
        aq.run_grid_search(
            strategy=_Guarded,
            param_grid={"dummy": [1, 2]},
            data=_data(5),
            symbols=SYMBOL,
            max_workers=2,
            show_progress=False,
            run_info={"k": object()},
        )


class _WfoRecorder(Strategy):
    """walk-forward: 每次运行在 on_start 记一次 run_info(同进程, 类级列表)."""

    dummy = IntParam(0)
    seen: List[Any] = []

    def on_start(self) -> None:
        type(self).seen.append(self.run_info)

    def on_bar(self, bar: Bar) -> None:
        return


def test_walk_forward_passes_run_info_to_train_and_test_runs() -> None:
    """walk-forward 的样本内网格与样本外回测都收到 run_info."""
    _WfoRecorder.seen = []
    result = aq.run_walk_forward(
        strategy=_WfoRecorder,
        param_grid={"dummy": [1]},
        data=_data(12),
        train_period=4,
        test_period=4,
        symbols=SYMBOL,
        max_workers=1,
        show_progress=False,
        run_info={"user_id": "u-1"},
    )
    assert isinstance(result, pd.DataFrame) and not result.empty
    # 两个窗口, 每个窗口样本内一次 + 样本外一次
    assert len(_WfoRecorder.seen) == 4
    assert all(dict(m) == {"user_id": "u-1"} for m in _WfoRecorder.seen)


# ---- 存档续跑


#: 续跑测试的观测记录放在模块里, 不放进策略属性(否则会随存档写进文件)
_RESUME_LOG: List[Any] = []


class _ResumeRecorder(Strategy):
    def on_resume(self) -> None:
        _RESUME_LOG.append(("resume", self.run_info))

    def on_bar(self, bar: Bar) -> None:
        _RESUME_LOG.append(("bar", self.run_info))


def _logged(kind: str) -> List[Any]:
    return [info for k, info in _RESUME_LOG if k == kind]


def test_checkpoint_not_stored_and_resume_uses_new_value(tmp_path: Any) -> None:
    """存档里不保存 run_info; 续跑以本次传入的为准, 不传则为空, 从不沿用旧值."""
    from akquant import run_from_checkpoint, save_checkpoint

    _RESUME_LOG.clear()
    first = _run(_ResumeRecorder, data=_data(2), run_info={"user_id": "stale-user-A"})
    assert [m["user_id"] for m in _logged("bar")] == ["stale-user-A"] * 2
    path = tmp_path / "run.ckpt"
    save_checkpoint(first.engine, first.strategy, str(path))
    assert b"stale-user-A" not in path.read_bytes()
    # 存档之后原实例上的值仍在(存档只是不写进文件)
    assert first.strategy.run_info["user_id"] == "stale-user-A"

    later = _data(4).iloc[2:]
    _RESUME_LOG.clear()
    run_from_checkpoint(
        str(path),
        data=later,
        symbols=SYMBOL,
        show_progress=False,
        run_info={"user_id": "fresh-user-B"},
    )
    assert [m["user_id"] for m in _logged("resume")] == ["fresh-user-B"]
    assert [m["user_id"] for m in _logged("bar")] == ["fresh-user-B"] * 2

    _RESUME_LOG.clear()
    run_from_checkpoint(str(path), data=later, symbols=SYMBOL, show_progress=False)
    assert [dict(m) for m in _logged("resume")] == [{}]
    assert [dict(m) for m in _logged("bar")] == [{}, {}]


def test_run_from_checkpoint_rejects_invalid_run_info(tmp_path: Any) -> None:
    """续跑同样校验 run_info."""
    from akquant import run_from_checkpoint, save_checkpoint

    first = _run(_ResumeRecorder, data=_data(2))
    path = tmp_path / "run.ckpt"
    save_checkpoint(first.engine, first.strategy, str(path))
    with pytest.raises(TypeError, match=BAD):
        run_from_checkpoint(
            str(path),
            data=_data(4).iloc[2:],
            symbols=SYMBOL,
            show_progress=False,
            run_info={"k": object()},
        )


# ---- 实盘(paper)


_FUT = "RB2310"


def _run_paper(strategy: Any, **kwargs: Any) -> Any:
    """Replay 行情 + paper 撮合(同 tests/test_bracket_fill_modes.py::_run_paper)."""
    index = pd.date_range(
        "2024-01-02 09:31:00", periods=3, freq="1min", tz="Asia/Shanghai"
    )
    bars = [Bar(int(ts.value), 10.0, 10.0, 10.0, 10.0, 1e6, _FUT) for ts in index]
    aq.run_live(
        strategy_cls=strategy,
        instruments=[
            aq.Instrument(
                symbol=_FUT,
                asset_type=aq.AssetType.Futures,
                multiplier=10.0,
                margin_ratio=0.1,
                tick_size=0.01,
                lot_size=1,
                option_type=None,
                strike_price=None,
                expiry_date=None,
            )
        ],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": bars, "freq": "1min"},
        cash=1_000_000.0,
        show_progress=False,
        duration="60s",
        **kwargs,
    )
    return strategy


def test_paper_live_main_slot_and_indicator_see_run_info() -> None:
    """run_live(trading_mode="paper"): 主策略、槽位、指标脚本都可读, 从 on_start 起."""
    _ScriptProbe.seen = []
    main = _Recorder()
    slot = _Recorder()
    _run_paper(
        main,
        strategies_by_slot={"other": slot},
        indicators=[_ScriptProbe],
        run_info=INFO,
    )
    for strategy in (main, slot):
        assert dict(strategy.seen["init"]) == {}
        assert _frozen_equal(strategy.seen["start"], INFO)
        assert strategy.bars and all(_frozen_equal(m, INFO) for m in strategy.bars)
    assert len(_ScriptProbe.seen) == 1
    assert _frozen_equal(_ScriptProbe.seen[0], INFO)


def test_paper_live_default_empty() -> None:
    """run_live 不传时为空映射."""
    strategy = _run_paper(_Recorder())
    assert dict(strategy.seen["start"]) == {}
    assert strategy.bars and all(dict(m) == {} for m in strategy.bars)


def test_paper_live_function_style_reads_ctx_run_info() -> None:
    """run_live 的函数式策略同样经 ctx.run_info 读取."""
    seen: List[Any] = []

    def initialize(ctx: Any) -> None:
        seen.append(ctx.run_info)

    def on_bar(ctx: Any, bar: Bar) -> None:
        seen.append(ctx.run_info)

    _run_paper(on_bar, initialize=initialize, run_info=INFO)
    assert len(seen) >= 2
    assert all(_frozen_equal(m, INFO) for m in seen)


def test_run_live_rejects_invalid_run_info() -> None:
    """run_live 在连接网关之前校验 run_info."""
    with pytest.raises(TypeError, match=BAD):
        _run_paper(_Recorder(), run_info={"k": object()})


def test_raising_in_on_start_aborts_paper_live(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """实盘: on_start 里抛的异常让会话以错误中止, 原因(含异常)记在 CRITICAL 日志里."""

    class _GuardedWithStop(_Guarded):
        def on_stop(self) -> None:
            self.stopped = True

    strategy = _GuardedWithStop()
    with caplog.at_level(logging.CRITICAL):
        _run_paper(strategy, run_info={"user_id": "u-2"})
    assert strategy.bar_count == 0
    assert getattr(strategy, "stopped", False), "on_stop 仍应调用"
    critical = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
    assert any(
        r.exc_info is not None
        and isinstance(r.exc_info[1], PermissionError)
        and "'u-2' is not allowed" in str(r.exc_info[1])
        for r in critical
    )


def _plain_frame() -> pd.DataFrame:
    index = pd.to_datetime(
        ["2024-01-02 15:00", "2024-01-03 15:00", "2024-01-04 15:00"]
    ).tz_localize("Asia/Shanghai")
    return pd.DataFrame(
        {"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 1000.0},
        index=index,
    )


def test_reassigning_run_info_on_the_class_is_rejected_at_run_start() -> None:
    """类定义之后再赋值绕过了类定义时的检查: 运行入口发现后报错, 不带着假值跑."""

    class Tampered(Strategy):
        def on_bar(self, bar: Any) -> None:
            pass

    Tampered.run_info = {"user_id": "forged"}  # type: ignore[misc,assignment]

    with pytest.raises(TypeError, match="run_info 是框架的只读属性"):
        aq.run_backtest(
            data={"000001.SZ": _plain_frame()},
            strategy=Tampered,
            symbols="000001.SZ",
            show_progress=False,
            run_info={"user_id": "real"},
        )


def test_replacing_the_base_property_is_rejected_for_later_runs() -> None:
    """替换 Strategy.run_info 会影响同进程之后的所有运行: 之后的运行入口报错."""

    class Innocent(Strategy):
        def on_bar(self, bar: Any) -> None:
            pass

    original = vars(Strategy)["run_info"]
    Strategy.run_info = property(  # type: ignore[misc,assignment]
        lambda self: {"user_id": "forged"}
    )
    try:
        with pytest.raises(TypeError, match="run_info 是框架的只读属性"):
            aq.run_backtest(
                data={"000001.SZ": _plain_frame()},
                strategy=Innocent,
                symbols="000001.SZ",
                show_progress=False,
                run_info={"user_id": "real"},
            )
    finally:
        Strategy.run_info = original  # type: ignore[misc,assignment]


@pytest.mark.parametrize("key", ["_framework_run_info", "_framework_freq"])
def test_context_cannot_overwrite_framework_private_state(key: str) -> None:
    """Context 的键会被设成策略属性: 不能借它改写框架内部状态(如运行信息)."""
    with pytest.raises(ValueError, match="_framework_"):
        _run(
            _Recorder,
            context={key: {"user_id": "forged"}},
            run_info={"user_id": "host"},
        )


def test_paper_live_context_cannot_overwrite_framework_private_state() -> None:
    """实盘入口同样拒绝."""
    with pytest.raises(ValueError, match="_framework_"):
        aq.run_live(
            strategy_cls=_Recorder,
            instruments=[],
            broker="replay",
            trading_mode="paper",
            gateway_options={"bars": [], "freq": "1min"},
            context={"_framework_run_info": {"user_id": "forged"}},
            run_info={"user_id": "host"},
            show_progress=False,
            duration="5s",
        )


@pytest.mark.parametrize("key", ["__dict__", "__class__"])
def test_context_cannot_replace_instance_internals(key: str) -> None:
    """``__dict__`` 整体替换实例属性也能带进假的运行信息: 双下划线键一并拒绝."""
    with pytest.raises(ValueError, match="双下划线"):
        _run(
            _Recorder,
            context={key: {"_framework_run_info": {"user_id": "forged"}}},
            run_info={"user_id": "host"},
        )


def test_run_info_is_written_again_after_context_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """键名检查之外的第二道: context 注入之后重写, 不依赖键名清单是否列全."""
    import akquant.backtest.engine as engine_module

    monkeypatch.setattr(
        engine_module, "reject_framework_context_keys", lambda context: None
    )
    monkeypatch.setattr(
        engine_module,
        "reject_context_keys_shadowing_internals",
        lambda strategy, context: None,
    )
    result = _run(
        _Recorder,
        context={"_framework_run_info": {"user_id": "forged"}},
        run_info={"user_id": "host"},
    )
    seen = result.strategy.seen
    assert dict(seen["start"]) == {"user_id": "host"}
    assert result.strategy.bars
    assert all(dict(m) == {"user_id": "host"} for m in result.strategy.bars)


def test_paper_live_run_info_is_written_again_after_context_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """实盘入口同样在 context 注入之后重写."""
    import akquant.live._runner as runner_module

    monkeypatch.setattr(
        runner_module, "reject_framework_context_keys", lambda context: None
    )
    monkeypatch.setattr(
        runner_module,
        "reject_context_keys_shadowing_internals",
        lambda strategy, context: None,
    )
    strategy = _Recorder()
    _run_paper(
        strategy,
        context={"_framework_run_info": {"user_id": "forged"}},
        run_info={"user_id": "host"},
    )
    assert dict(strategy.seen["start"]) == {"user_id": "host"}
    assert strategy.bars
    assert all(dict(m) == {"user_id": "host"} for m in strategy.bars)


def test_functional_strategy_constructor_rejects_framework_context_keys() -> None:
    """直接构造函数式策略(不经运行入口)时 context 同样受检."""
    from akquant.backtest.engine import FunctionalStrategy

    with pytest.raises(ValueError, match="_framework_"):
        FunctionalStrategy(
            None,
            lambda ctx, bar: None,
            context={"_framework_run_info": {"user_id": "forged"}},
            run_info={"user_id": "host"},
        )


def test_grid_param_named_run_info_cannot_replace_callers_run_info() -> None:
    """非严格模式下参数组合里的 run_info 不会被拒: 它也不能顶掉调用方传入的."""
    denied = aq.run_grid_search(
        strategy=_Guarded,
        param_grid={"dummy": [1], "run_info": [{"user_id": "u-1"}]},
        data=_data(5),
        symbols=SYMBOL,
        max_workers=1,
        return_df=False,
        raise_on_all_failed=False,
        show_progress=False,
        strict_strategy_params=False,
        run_info={"user_id": "u-2"},
    )
    assert isinstance(denied, list) and len(denied) == 1
    assert "'u-2' is not allowed" in str(denied[0].error)


def _functional_guard_run(**kwargs: Any) -> List[str]:
    """函数式策略: initialize / on_start 里各记一次读到的 user_id, 返回调用记录."""
    calls: List[str] = []

    def initialize(ctx: Any) -> None:
        calls.append("initialize:" + str(ctx.run_info.get("user_id")))

    def on_start(ctx: Any) -> None:
        calls.append("on_start:" + str(ctx.run_info.get("user_id")))

    def on_bar(ctx: Any, bar: Bar) -> None:
        if "bar" not in calls:
            calls.append("bar")

    context = kwargs.get("context")
    result = _run(on_bar, initialize=initialize, on_start=on_start, **kwargs)
    if context is not None:
        assert result.strategy._context == context
    return calls


@pytest.mark.parametrize(
    "key",
    ["_on_start_func", "_initialize", "_on_bar_func", "_context"],
)
def test_functional_context_cannot_switch_off_hooks(key: str) -> None:
    """函数式策略的钩子存在实例属性里: context 里带同名键不能把钩子换掉或关掉."""
    calls = _functional_guard_run(context={key: None}, run_info={"user_id": "host"})
    assert calls == ["initialize:host", "on_start:host", "bar"]


def test_functional_initialize_reads_callers_run_info_without_key_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """键名检查之外的第二道: initialize(ctx) 读到的也是调用方的值."""
    import akquant.backtest.engine as engine_module

    monkeypatch.setattr(
        engine_module, "reject_framework_context_keys", lambda context: None
    )
    monkeypatch.setattr(
        engine_module,
        "reject_context_keys_shadowing_internals",
        lambda strategy, context: None,
    )
    calls = _functional_guard_run(
        context={"_framework_run_info": {"user_id": "forged"}},
        run_info={"user_id": "host"},
    )
    assert calls == ["initialize:host", "on_start:host", "bar"]


def test_walk_forward_param_named_run_info_cannot_replace_callers_run_info() -> None:
    """walk-forward 的样本外回测同样不能被参数组合里的 run_info 顶掉."""
    _WfoRecorder.seen = []
    aq.run_walk_forward(
        strategy=_WfoRecorder,
        param_grid={"dummy": [1], "run_info": [{"user_id": "forged"}]},
        data=_data(40),
        symbols=SYMBOL,
        train_period=10,
        test_period=10,
        metric="total_return",
        show_progress=False,
        strict_strategy_params=False,
        run_info={"user_id": "host"},
    )
    assert _WfoRecorder.seen
    assert all(dict(m) == {"user_id": "host"} for m in _WfoRecorder.seen)


def test_context_cannot_mark_on_start_as_already_run() -> None:
    """``_start_initialized`` 是标记 on_start 已执行的内部开关: context 不能把它置真."""
    forged = {"_start_initialized": True}
    with pytest.raises(ValueError, match="_start_initialized"):
        _run(_Guarded, context=forged, run_info={"user_id": "u-2"})
    with pytest.raises(ValueError, match="_start_initialized"):
        _functional_guard_run(context=forged, run_info={"user_id": "u-2"})
    with pytest.raises(ValueError, match="_start_initialized"):
        _run_paper(_Guarded(), context=forged, run_info={"user_id": "u-2"})


def test_context_may_add_new_private_attributes() -> None:
    """策略自己新起的下划线属性(实例上原本没有)照常注入."""
    result = _run(_Recorder, context={"_my_flag": 7}, run_info={"user_id": "host"})
    assert result.strategy._my_flag == 7


def test_grid_param_cannot_mark_functional_on_start_as_already_run() -> None:
    """函数式策略的网格参数经 context 注入: 同样不能跳过 on_start."""

    def on_start(ctx: Any) -> None:
        raise PermissionError("denied")

    def on_bar(ctx: Any, bar: Bar) -> None:
        return

    results = aq.run_grid_search(
        strategy=on_bar,  # type: ignore[arg-type]
        on_start=on_start,
        param_grid={"n": [1], "_start_initialized": [True]},
        data=_data(5),
        symbols=SYMBOL,
        max_workers=1,
        return_df=False,
        raise_on_all_failed=False,
        show_progress=False,
        run_info={"user_id": "u-2"},
    )
    assert isinstance(results, list) and len(results) == 1
    assert results[0].error
