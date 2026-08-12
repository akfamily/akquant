# Tick Size 对齐指南

委托价必须是标的最小变动价位（tick size）的整数倍——这是交易所层面的硬约束，
不是 AKQuant 自造的规则。本页说明框架如何校验它、为什么**只校验不自动取整**，
以及如何用 `AssetType.Fund` 在回测里近似可转债。

## 1. 为什么框架不自动取整

框架在委托路径上的立场是：**校验，拒单，报出可用价格；绝不悄悄改价**。原因有四层：

1. **回测侧本来就是 reject 语义。** `FuturesMatcher::validate_order`
   （[`src/execution/futures.rs`](https://github.com/akfamily/akquant/blob/main/src/execution/futures.rs)）
   一直是"价格不对齐就拒单"，而不是"round 到最近的合法价再成交"。如果实盘路径
   改成自动取整，同一笔委托就会出现"回测拒单、实盘却按取整后的价格成交"的语义
   分裂——用户在回测里学到的规则，在实盘里不成立。
2. **静默改价是隐性成本。** 用户没算过的价格成交了，滑点从哪来、成本对不对，
   都变得不可审计。报错但不下单，用户能立刻看到问题并自己决定用哪个方向取整；
   静默改价把这个决定权拿走了。
3. **CCXT 的前车之鉴。** CCXT 对接近百家交易所时会按各交易所的 tick/precision
   规则隐式取整下单价格与数量，社区里有相当一批 issue 反映过取整方向、四舍五入
   模式与交易所实际规则不一致，用户既没有主动做这个决定，出问题时也难以定位
   到底是自己的价格错了还是库悄悄改了价。
4. **QuantConnect Lean 的真实事故。** Lean 的价格取整发生在 `RoundPrice`
   / brokerage 模型这一层，会对委托价按 `PriceVariationModel` 自动取整；当某个
   标的的 tick size 元数据缺失时会回退到一个较粗的默认值。已知的一个后果是：
   止损单在自动取整后被挪到了入场价之上，等于这笔仓位实际上没有止损保护。
   AKQuant 把"元数据缺失"和"自动取整"这两个开关都拿掉——tick 缺省值按资产类型
   分流（见第 4 节），且无论如何都不做自动取整，就不会重现这条事故链路。

结论：**AKQuant 只提供取整工具，取整这个动作必须由策略代码显式调用。**

## 2. 怎么取整——`round_to_tick`

策略里最方便的入口是 `Strategy.round_to_tick(symbol, price, direction="nearest")`：
它会从标的的 `instrument` 快照里读 `tick_size`，再委托给底层的
[`akquant.utils.price.round_to_tick`](https://github.com/akfamily/akquant/blob/main/python/akquant/utils/price.py)。

```python
class MyStrategy(Strategy):
    def on_bar(self, bar):
        symbol = bar.symbol
        raw_price = bar.close * 1.001  # 比如按信号算出来的一个"生价"

        # 买单用 "down"：取整后价格不会比你算出来的价格更贵
        buy_price = self.round_to_tick(symbol, raw_price, "down")
        self.buy(symbol, 100, price=buy_price)

        # 卖单用 "up"：取整后价格不会比你算出来的价格更便宜
        sell_price = self.round_to_tick(symbol, raw_price, "up")
        self.sell(symbol, 100, price=sell_price)
```

规则速记：

| `direction` | 取整方式 | 什么时候用 |
|---|---|---|
| `"down"` | 向下取整（`ROUND_FLOOR`） | 买单——不把价格推得比你想付的更高 |
| `"up"` | 向上取整（`ROUND_CEILING`） | 卖单——不把价格推得比你想收的更低 |
| `"nearest"`（默认） | 四舍五入（`ROUND_HALF_UP`） | 只是想展示/记录一个"看起来合理"的价格，不进委托 |

两个边界行为需要知道：

- `symbol` 未在 `instruments_config` / `get_instrument` 里登记时，
  `Strategy.round_to_tick` 会抛 `KeyError`——**不会**偷偷用某个缺省 tick 顶上去
  （猜错 tick 比报错更危险，见第 1 节 Lean 的事故）。
- 底层的 `round_to_tick(price, tick, direction)` 函数在 `tick <= 0` 时原样返回
  `price`（部分柜台/模拟环境会给出 0），`direction` 拼错则抛 `ValueError`。

如果你什么都不做，直接把一个未对齐的价格传给 `buy`/`sell`：

- **实盘**：`python/akquant/gateway/order_submitter.py::_validate_price_tick`
  会在报单前本地拒单，错误信息里同时给出委托价、tick 和两个方向的建议价，例如：

  ```text
  委托价 10.567 不是 000001.SZ 最小变动价位 0.01 的整数倍, 已本地拒单;
  买入可用 10.56, 卖出可用 10.57;
  或调用 self.round_to_tick('000001.SZ', price, 'down'/'up') 自行对齐
  ```

- **回测**：对应的 matcher（股票/基金走 `StockMatcher`，期货走 `FuturesMatcher`）
  会把这笔委托判定为无效并拒单，不会静默成交在别的价格上。

## 3. A 股 tick 速查表

| 品种 | tick size | 依据 |
|---|---|---|
| A 股股票 | 0.01 | 交易所通行规则 |
| ETF / 基金 | 0.001 | 深圳证券交易所交易规则第 3.3.13 条 |
| 债券（含可转债） | 0.001 | 上海证券交易所可转换公司债券交易实施细则第六条；**沪深两市口径一致**，均为 0.001 |

> 网上部分资料称上交所可转债 tick 为 0.01，那是 2022 年之前已被替代的旧规则，
> 引用时请以现行细则为准。

## 4. 缺省值规则

`InstrumentConfig` / Rust 侧 `Instrument::new` 在没有显式传 `tick_size` 时，按
`asset_type` 分流缺省值（见
[`src/model/instrument.rs`](https://github.com/akfamily/akquant/blob/main/src/model/instrument.rs)）：

- `AssetType.Fund` → `0.001`
- 其余（`Stock` / `Futures` / `Option` / ...） → `0.01`

**这不是"猜"出来的兼容行为，而是设计意图**：A 股股票统一 0.01，基金/债券统一
0.001，这两条规则是确定的，缺省值就该按它们分流，而不是像 Lean 那样在元数据缺失
时统一回退到一个粗粒度的值。

需要你自己注意的坑：**如果 ETF 或可转债在 `InstrumentConfig` 里被配成
`asset_type="STOCK"`（而不是 `"FUND"`），缺省 tick 会变成 0.01，与它们实际的
0.001 规则不符**，必须显式传 `tick_size=0.001` 覆盖，否则会出现合法委托被误拒、
或者—如果你手工设的价格恰好是 0.01 的整数倍—悄悄跳过了本该更细的校验粒度。

## 5. 关闭校验

Tick 校验默认开启，两条独立开关（分别对应股票/基金与期货撮合器）：

```python
from akquant import BacktestConfig, ChinaStockConfig, StrategyConfig, run_backtest

config = BacktestConfig(
    strategy_config=StrategyConfig(initial_cash=100000.0),
    china_stock=ChinaStockConfig(enforce_tick_size=False),  # 关闭股票/基金的 tick 校验
)
```

期货侧对应 `ChinaFuturesValidationConfig(symbol_prefix=..., enforce_tick_size=False)`
（可整体或按品种前缀覆盖）。实盘侧目前没有单独的关闭开关——`_validate_price_tick`
只有在标的未登记或 `tick_size <= 0` 时才会跳过校验。

**不建议关闭**，因为：

- 关闭后，未对齐的委托价会被交易所/柜台在下单之后拒绝或按柜台自己的口径处理，
  排障成本从"本地立刻看到一行错误"变成"跑到柜台/交易所那一层才发现"；
- 关闭校验不会让"提交一个不合规的价格"这件事本身消失，只是把发现它的时间点
  推迟、推给了更贵的环节。

## 6. 用 `AssetType.Fund` 回测可转债

框架目前**没有** `AssetType::Bond`，可转债通过把它建模成
`AssetType.Fund` 来近似：

- `tick_size=0.001`——沪深口径一致（见第 3 节）
- `sellable_after_days=0`——可转债 T+0，当天买入当天可卖
- `multiplier=1`——面值 100 元，1 手 = 10 张，因此下单数量按"张"计，一次买 10
  张即 1 手
- 免印花税——`AssetType::Fund` 本来就不收印花税
  （对比 [`src/market/fund.rs`](https://github.com/akfamily/akquant/blob/main/src/market/fund.rs)
  与收印花税的 [`src/market/stock.rs`](https://github.com/akfamily/akquant/blob/main/src/market/stock.rs)），
  这恰好匹配可转债的真实费用结构

```python
import pandas as pd
from akquant import (
    BacktestConfig,
    ChinaStockConfig,
    InstrumentConfig,
    Strategy,
    StrategyConfig,
    run_backtest,
)

dates = pd.date_range("2024-01-02", periods=5, freq="D")
df = pd.DataFrame(
    {
        "timestamp": dates,
        "open": [100.123, 100.5, 101.2, 100.8, 101.5],
        "high": [100.5, 101.0, 101.5, 101.2, 102.0],
        "low": [99.8, 100.2, 100.5, 100.3, 101.0],
        "close": [100.2, 100.8, 101.0, 100.9, 101.8],
        "volume": [1000] * 5,
        "symbol": "113050.SH",
    }
)


class ConvertibleBondStrategy(Strategy):
    """最小示例：用 AssetType.Fund 近似可转债，买单显式向下取整."""

    def on_bar(self, bar):
        symbol = bar.symbol
        if self.get_position(symbol) == 0:
            raw_price = bar.close * 1.001  # 故意构造一个未对齐 tick 的价格
            price = self.round_to_tick(symbol, raw_price, "down")
            self.buy(symbol, 10, price=price)  # 1 手 = 10 张


result = run_backtest(
    data=df,
    strategy=ConvertibleBondStrategy,
    show_progress=False,
    config=BacktestConfig(
        strategy_config=StrategyConfig(initial_cash=100000.0),
        instruments_config=[
            InstrumentConfig(
                symbol="113050.SH",
                asset_type="FUND",
                tick_size=0.001,
                sellable_after_days=0,
                multiplier=1,
            )
        ],
        china_stock=ChinaStockConfig(enforce_tick_size=True),
    ),
)

print(result)
```

### 明确不支持的部分

这个近似**只覆盖了行情撮合和费用结构这一层**，以下可转债特有的业务规则框架
里都没有实现，也不在本计划的范围内：

- **转股**——不能把持仓换成对应股票的仓位
- **回售**——不支持触发条件后按约定价格回售给发行人
- **强赎**——不支持发行人提前赎回
- **上市首日临时停牌**——不支持

另外——**框架目前完全没有涨跌停（价格限制）机制**，这是独立于 tick size 的另一
套东西。因此可转债交易所规定的 ±20%（首日 ±57.3%/±44.5%，非首日 ±20%）价格波动
限制在这里**不生效**：回测里一个超出该区间的价格只会按 tick 规则校验，不会因为
"涨跌停"被拒。不要因为本节的例子能跑起来，就认为可转债的交易规则已经被完整
建模——它没有。

## 相关链接

- [`round_to_tick` 工具函数](https://github.com/akfamily/akquant/blob/main/python/akquant/utils/price.py)
- [实盘报单前校验](https://github.com/akfamily/akquant/blob/main/python/akquant/gateway/order_submitter.py)
- [回测侧 tick 校验（股票/基金）](https://github.com/akfamily/akquant/blob/main/src/execution/stock.rs)
- [回测侧 tick 校验（期货）](https://github.com/akfamily/akquant/blob/main/src/execution/futures.rs)
- [共享校验 helper](https://github.com/akfamily/akquant/blob/main/src/execution/validation.rs)
- [Broker Capability Matrix 的统一错误规范](broker_capability_matrix.md)
