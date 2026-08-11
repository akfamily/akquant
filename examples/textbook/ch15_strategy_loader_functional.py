"""
第 15 章：动态策略加载（Strategy Loader）（函数式写法）.

本章演示如何在不直接导入策略类的情况下，通过 strategy_source + strategy_loader
在运行时加载策略实现。

函数式改写说明：
本示例与 ch15_strategy_loader.py 的两个场景、数据构造、加载参数完全一致，
只把**被加载的策略**从类风格换成函数式：

1. 场景一源码字符串里的 class ChapterPlainStrategy(Strategy) -> 模块级 on_bar 函数
2. 场景二 decrypt_and_load 的返回值从策略类 -> 函数式 on_bar 回调
3. 两个场景仍然都传 strategy=None——策略来自 loader，而不是 strategy 形参

`python/akquant/strategy_loader.py` 的 ``StrategyLike`` 是
``Union[type[Strategy], Strategy, Callable[[Any, Bar], None]]``，
``_is_strategy_like`` 对任意 callable 都返回 True，所以**函数式策略是 loader
的一等受支持输入**，两个内置 loader（python_plain / encrypted_external）都能返回它。

两处必须注意的加载器约束（实测得出，函数式下才会暴露）：

- **python_plain 只解析出一个属性。** 它按 ``strategy_attr`` 从模块里取**单个**
  对象返回；源码文件里另外定义的 ``initialize`` 不会被自动配对使用。类风格版不
  受影响，因为状态初始化写在被加载类的 ``__init__`` 里，随类一起被加载；函数式
  则必须让状态初始化跟着 ``on_bar`` 一起走——本示例在 on_bar 里用
  ``getattr(ctx, "calls", 0)`` 自带初值，见场景一的注释。
- **不传 strategy_attr 时 python_plain 会直接失败。** 它的兜底扫描只找
  ``Strategy`` 子类，函数式源码里没有子类，会抛
  ``ValueError: no Strategy subclass found in module``。所以函数式源码必须显式
  指定 ``strategy_attr``（这里指向 ``"on_bar"``）。

场景二没有这个限制：自定义 loader 与调用方在同一模块，可以照常把
``initialize=`` 直接传给 run_backtest，这也正是两个场景在函数式下的真实差异。

两份示例的 calls 计数输出应完全一致（3 与 2），
这说明函数式与类风格在 loader 链路上等价。
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import akquant as aq
from akquant import Bar


def make_bars(symbol: str, count: int) -> list[Bar]:
    """构造示例 K 线（与类风格版完全一致）."""
    start = datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc)
    bars: list[Bar] = []
    for i in range(count):
        dt = start + timedelta(minutes=i)
        ts_ns = int(dt.timestamp() * 1_000_000_000)
        price = 50.0 + float(i)
        bars.append(
            Bar(
                timestamp=ts_ns,
                open=price,
                high=price + 0.8,
                low=price - 0.8,
                close=price + 0.3,
                volume=2000.0 + float(i),
                symbol=symbol,
            )
        )
    return bars


def chapter_plain_loader() -> None:
    """场景一：python_plain 从源码文件加载函数式 on_bar."""
    bars = make_bars("CH15_PLAIN", 3)
    with TemporaryDirectory() as tmp_dir:
        strategy_path = Path(tmp_dir) / "chapter_strategy.py"
        # 源码里定义的是模块级函数而不是 Strategy 子类。
        # 计数初值写成 getattr(ctx, "calls", 0)，而不是另开一个 initialize 函数：
        # python_plain 只按 strategy_attr 取回单个对象，源码里的 initialize
        # 不会被自动使用（见模块 docstring）。
        strategy_path.write_text(
            "\n".join(
                [
                    "from typing import Any",
                    "",
                    "",
                    "def on_bar(ctx: Any, bar: Any) -> None:",
                    '    """动态加载的函数式策略主回调."""',
                    "    _ = bar",
                    '    ctx.calls = getattr(ctx, "calls", 0) + 1',
                ]
            ),
            encoding="utf-8",
        )
        # 先单独解析一次，确认 loader 交回来的是 callable 而不是类。
        loaded = aq.resolve_strategy_input(
            strategy=None,
            strategy_source=str(strategy_path),
            strategy_loader="python_plain",
            strategy_loader_options={"strategy_attr": "on_bar"},
        )
        assert callable(loaded), "函数式策略应被解析为 callable"
        assert not isinstance(loaded, type), "函数式策略不应被解析为类"
        print(f"chapter15_plain_loaded_callable={callable(loaded)}")

        result = aq.run_backtest(
            data=bars,
            strategy=None,
            strategy_source=str(strategy_path),
            strategy_loader="python_plain",
            # 函数式源码里没有 Strategy 子类，strategy_attr 必须显式给出，
            # 否则 python_plain 的兜底扫描会抛 ValueError。
            strategy_loader_options={"strategy_attr": "on_bar"},
            symbols="CH15_PLAIN",
            show_progress=False,
        )
    strategy = result.strategy
    calls = getattr(strategy, "calls", -1) if strategy is not None else -1
    print(f"chapter15_plain_calls={calls}")


def chapter_encrypted_loader() -> None:
    """场景二：encrypted_external 通过外部回调加载函数式 on_bar."""
    bars = make_bars("CH15_ENC", 2)

    def chapter_encrypted_initialize(ctx: Any) -> None:
        """被加载策略的状态初始化，等价于类风格版的 __init__."""
        ctx.calls = 0

    def chapter_encrypted_on_bar(ctx: Any, bar: Bar) -> None:
        """被加载策略的主回调，等价于类风格版的 on_bar."""
        _ = bar
        ctx.calls += 1

    def decrypt_and_load(
        source: Any, options: dict[str, Any]
    ) -> Any:  # 返回 callable 而非策略类
        """解密并交回策略入口；这里返回的是函数式 on_bar."""
        _ = source
        _ = options
        return chapter_encrypted_on_bar

    result = aq.run_backtest(
        data=bars,
        strategy=None,
        strategy_source=b"chapter15_encrypted_payload",
        strategy_loader="encrypted_external",
        strategy_loader_options={"decrypt_and_load": decrypt_and_load},
        # 自定义 loader 与调用方同处一个模块，initialize 可以照常直接传，
        # 不必像场景一那样在 on_bar 里自带初值。
        initialize=chapter_encrypted_initialize,
        symbols="CH15_ENC",
        show_progress=False,
    )
    strategy = result.strategy
    calls = getattr(strategy, "calls", -1) if strategy is not None else -1
    print(f"chapter15_encrypted_calls={calls}")


def main() -> None:
    """运行第 15 章示例（函数式写法）."""
    chapter_plain_loader()
    chapter_encrypted_loader()
    print("done_ch15_strategy_loader")


if __name__ == "__main__":
    main()
