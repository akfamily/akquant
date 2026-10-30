# 下单与仓位

下单用 `self.buy` / `self.sell`（期货开空、平空用 `self.short` / `self.cover`），按目标调仓用 `self.order_target_percent` 等，持仓用 `self.get_position` 查询。

## 成交时机

默认的成交方式是 `NextOpen()`：在 `on_bar` 里下的市价单，按**下一根 bar 的开盘价**成交，不是当前 bar 的收盘价。所以下单后立刻调用 `get_position` 看不到变化，要等成交回报（`on_trade`）之后才会更新。

## 常用接口

| 接口 | 说明 |
|---|---|
| `self.buy(symbol, quantity, price=None)` | 买入；不传 `price` 是市价单，传了是限价单 |
| `self.sell(symbol, quantity, price=None)` | 卖出 |
| `self.order_target_percent(target_percent=, symbol=)` | 调到占权益的目标比例，0 表示清仓 |
| `self.order_target_value(target_value=, symbol=)` | 调到目标市值 |
| `self.order_target(target=, symbol=)` | 调到目标数量 |
| `self.get_position(symbol)` | 当前持仓数量（float） |
| `self.get_account()` | 账户快照 dict |
| `self.equity` | 当前权益 |
| `self.cancel_order(order_id)` / `self.cancel_all_orders(symbol)` | 撤单 |

`buy`/`sell` 不传 `quantity` 时，数量交给 sizer 决定。各接口的完整参数用 `get_api("Strategy.buy")` 等查询；传了签名里没有的关键字参数（比如 `qty=`），`check_strategy` 会报 `unknown-kwarg`。

## 示例

```python
from typing import Any

from akquant import Bar, Strategy


class BuyDipStrategy(Strategy):
    """收盘跌超 2% 挂限价买单, 持有 5 根 bar 后市价卖出."""

    def on_start(self) -> None:
        self.prev_close: dict[str, float] = {}
        self.held: dict[str, int] = {}

    def on_bar(self, bar: Bar) -> None:
        prev = self.prev_close.get(bar.symbol)
        self.prev_close[bar.symbol] = bar.close
        position = self.get_position(bar.symbol)
        if position > 0:
            self.held[bar.symbol] = self.held.get(bar.symbol, 0) + 1
            if self.held[bar.symbol] >= 5:
                self.sell(symbol=bar.symbol, quantity=position)
                self.held[bar.symbol] = 0
        elif prev is not None and bar.close < prev * 0.98:
            self.cancel_all_orders(bar.symbol)
            limit = round(bar.close * 0.99, 2)
            self.buy(symbol=bar.symbol, quantity=100, price=limit)

    def on_trade(self, trade: Any) -> None:
        self.log(f"成交: {trade}")
```
