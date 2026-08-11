"""
第 15 章：实盘交易系统 (Live Trading)（函数式写法）.

本示例展示了如何将策略部署到实盘环境。
AKQuant 支持通过内置 broker 网关（ctp/miniqmt/ptrade）接入行情与交易链路。

注意：
1. 实盘交易涉及真实资金，请务必在模拟盘充分测试。
2. 本代码仅为配置演示，无法直接运行，因为需要有效账户信息。

配置流程：
1. 准备对应 broker 的账户凭证与连接参数。
2. 获取行情与交易前置地址（若 broker 需要）。
3. 配置 run_live 并启动。

函数式改写说明：
本示例与 ch15_live_trading.py 的交易逻辑、参数、连接配置完全一致，
只把策略入口从类风格换成函数式：

1. Strategy.on_bar 方法 -> 模块级 on_bar(ctx, bar) 函数（self 全部换成 ctx）
2. 类风格版没有 __init__，本示例的 initialize 只把 on_bar 里硬编码的两个
   均线长度（5 / 20）提到 ctx 上；类风格版没有设置 warmup_period，
   本示例也不设置（实盘的预热由 on_bar 里 len(closes) < 20 的判断兜住）。
3. strategy_cls=LiveDemoStrategy -> strategy_cls=on_bar + initialize=initialize

**本章最容易踩的坑：run_live 的形参名叫 strategy_cls，但它并不只收类。**
`python/akquant/live/_facade.py` 的类型标注是
``Optional[Union[Type[Strategy], Strategy, Callable[[Any, Bar], None]]]``，
参数文档也写明 "Strategy class/instance, or function-style on_bar callback"。
也就是说函数式策略在实盘链路上是**受支持**的一等输入，只是形参名沿用了
历史命名，读起来像"只能传类"。不要因为这个名字就以为实盘不能用函数式。

本示例无回测统计可比对：没有可用的 CTP 环境时，run_live 会因缺少接口库或
连接失败而抛异常，被外层 try/except 兜住并打印与类风格版完全相同的提示，
进程仍以 exit 0 结束。
"""

from typing import Any

import akquant as aq
from akquant import Bar, Instrument, run_live


def initialize(ctx: Any) -> None:
    """
    策略状态初始化，等价于类风格的 __init__.

    类风格版 LiveDemoStrategy 没有 __init__，均线长度直接写在 on_bar 里；
    这里把它们提到 ctx 上，数值与类风格版一致。
    注意：类风格版没有设置 warmup_period，所以这里也不设置，不凭空发明。
    """
    ctx.short_window = 5
    ctx.long_window = 20


def on_bar(ctx: Any, bar: Bar) -> None:
    """收到 Bar 事件的回调，等价于类风格的 on_bar；self 全部换成 ctx."""
    ctx.log(f"[Live] Received Bar: {bar.symbol} @ {bar.close}")

    closes = ctx.get_history(ctx.long_window, bar.symbol, "close")
    if len(closes) < ctx.long_window:
        return

    ma5 = closes[-ctx.short_window :].mean()
    ma20 = closes[-ctx.long_window :].mean()

    pos = ctx.get_position(bar.symbol)

    if ma5 > ma20 and pos == 0:
        ctx.log("金叉 -> 买入开仓")
        ctx.buy(bar.symbol, 1)
    elif ma5 < ma20 and pos > 0:
        ctx.log("死叉 -> 卖出平仓")
        ctx.close_position(bar.symbol)


if __name__ == "__main__":
    print("正在配置实盘环境...")

    rb2310 = Instrument(
        symbol="rb2310",
        asset_type=aq.AssetType.Futures,
        multiplier=10,
        margin_ratio=0.1,
    )

    CTP_CONFIG = {
        "md_front": "tcp://180.168.146.187:10131",
        "td_front": "tcp://180.168.146.187:10130",
        "broker_id": "9999",
        "user_id": "YOUR_USER_ID",
        "password": "YOUR_PASSWORD",
        "app_id": "simnow_client_test",
        "auth_code": "0000000000000000",
    }

    try:
        print("启动 CTP 接口...")
        run_live(
            # 注意：形参名是 strategy_cls，但它同时接受函数式 on_bar 回调
            # （见 python/akquant/live/_facade.py 的参数标注与文档）。
            # 传函数式策略时这个形参名容易误导——这里传的是模块级函数而非类。
            strategy_cls=on_bar,
            initialize=initialize,
            instruments=[rb2310],
            md_front=CTP_CONFIG["md_front"],
            td_front=CTP_CONFIG["td_front"],
            broker_id=CTP_CONFIG["broker_id"],
            user_id=CTP_CONFIG["user_id"],
            password=CTP_CONFIG["password"],
            app_id=CTP_CONFIG["app_id"],
            auth_code=CTP_CONFIG["auth_code"],
            cash=500_000,
        )

    except ImportError:
        print(
            "错误: 未找到 CTP 接口库。请确保已安装 akquant[ctp] 或手动配置 "
            "thosttraderapi。"
        )
    except Exception as e:
        print(f"实盘启动失败: {e}")
