"""
第 15 章：实盘交易系统 (Live Trading).

本示例展示了如何将策略部署到实盘环境。
AKQuant 支持通过内置 broker 网关（ctp/miniqmt/ptrade）接入行情与交易链路。

注意：
1. 实盘交易涉及真实资金，请务必在模拟盘充分测试。
2. 本代码仅为配置演示，无法直接运行，因为需要有效账户信息。

配置流程：
1. 准备对应 broker 的账户凭证与连接参数。
2. 获取行情与交易前置地址（若 broker 需要）。
3. 配置 run_live 并启动。
"""

import akquant as aq
from akquant import Bar, Instrument, Strategy, run_live


class LiveDemoStrategy(Strategy):
    """实盘演示策略."""

    def on_bar(self, bar: Bar) -> None:
        """收到 Bar 事件的回调."""
        self.log(f"[Live] Received Bar: {bar.symbol} @ {bar.close}")

        closes = self.get_history(20, bar.symbol, "close")
        if len(closes) < 20:
            return

        ma5 = closes[-5:].mean()
        ma20 = closes[-20:].mean()

        pos = self.get_position(bar.symbol)

        if ma5 > ma20 and pos == 0:
            self.log("金叉 -> 买入开仓")
            self.buy(bar.symbol, 1)
        elif ma5 < ma20 and pos > 0:
            self.log("死叉 -> 卖出平仓")
            self.close_position(bar.symbol)


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
            strategy_cls=LiveDemoStrategy,
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
