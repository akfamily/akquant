from typing import Callable

from akquant.gateway.broker_models import UnifiedAccount, UnifiedPosition
from akquant.gateway.broker_state_cache import BrokerStateCache


class _Gw:
    """Fake gateway counting query calls."""

    def __init__(self) -> None:
        """Init counters."""
        self.pos_calls = 0
        self.acct_calls = 0

    def query_positions(self) -> list[UnifiedPosition]:
        """Return one position, counting calls."""
        self.pos_calls += 1
        return [
            UnifiedPosition(symbol="600000.SH", quantity=1000, available_quantity=800)
        ]

    def query_account(self) -> UnifiedAccount:
        """Return an account, counting calls."""
        self.acct_calls += 1
        return UnifiedAccount(account_id="a", equity=2.0, cash=1.0, available_cash=0.5)

    def sync_open_orders(self) -> list[object]:
        """Return no open orders."""
        return []


def test_positions_cached_until_invalidate() -> None:
    """positions() caches the gateway query until invalidate()."""
    gw = _Gw()
    cache = BrokerStateCache(gw)
    assert cache.positions()["600000.SH"] == 1000.0
    assert cache.available_positions()["600000.SH"] == 800.0
    cache.positions()
    assert gw.pos_calls == 1  # cached
    cache.invalidate()
    cache.positions()
    assert gw.pos_calls == 2  # refreshed


def test_account_and_error_fallback() -> None:
    """account() caches; a raising query returns the last cache (or None)."""
    gw = _Gw()
    cache = BrokerStateCache(gw)
    assert cache.account().equity == 2.0
    assert gw.acct_calls == 1

    def _boom() -> UnifiedAccount:
        raise RuntimeError("down")

    boom_query: Callable[[], UnifiedAccount] = _boom
    setattr(gw, "query_account", boom_query)
    cache.invalidate()
    # 异常 → 返回上次缓存(不抛)
    assert cache.account().equity == 2.0


class _RowsGw:
    """假柜台: 持仓查询返回给定的若干行."""

    def __init__(self, rows: list[UnifiedPosition]) -> None:
        """Keep the rows to return."""
        self.rows = rows

    def query_positions(self) -> list[UnifiedPosition]:
        """Return the configured rows."""
        return list(self.rows)


def test_short_row_with_positive_quantity_reads_as_negative() -> None:
    """柜台用 direction 表示空头、数量给正数时, 持仓与可用都读成负数.

    中间件的空头行就是这个形状(数量恒正, 方向在 position_side 里)。读成正数会让
    策略把空头当多头: 平仓指令变成加仓。回测里空头的总持仓和可用持仓都是负数。
    """
    cache = BrokerStateCache(
        _RowsGw(
            [
                UnifiedPosition(
                    symbol="IF2512.CFFEX",
                    quantity=2,
                    available_quantity=2,
                    direction="short",
                )
            ]
        )
    )
    assert cache.positions()["IF2512.CFFEX"] == -2.0
    assert cache.available_positions()["IF2512.CFFEX"] == -2.0


def test_short_row_already_negative_is_not_negated_twice() -> None:
    """柜台适配器已把空头数量取负(CTP)时不再翻一次; 可用量仍按方向取负."""
    cache = BrokerStateCache(
        _RowsGw(
            [
                UnifiedPosition(
                    symbol="rb2601.SHFE",
                    quantity=-2,
                    available_quantity=2,
                    direction="short",
                )
            ]
        )
    )
    assert cache.positions()["rb2601.SHFE"] == -2.0
    assert cache.available_positions()["rb2601.SHFE"] == -2.0


def test_long_and_short_rows_of_one_symbol_are_netted() -> None:
    """同一合约多空各一行(锁仓)时按净持仓合并, 与行的先后顺序无关."""
    long_row = UnifiedPosition(
        symbol="rb2601.SHFE", quantity=3, available_quantity=3, direction="long"
    )
    short_row = UnifiedPosition(
        symbol="rb2601.SHFE", quantity=-1, available_quantity=1, direction="short"
    )
    for rows in ([long_row, short_row], [short_row, long_row]):
        cache = BrokerStateCache(_RowsGw(rows))
        assert cache.positions()["rb2601.SHFE"] == 2.0
        assert cache.available_positions()["rb2601.SHFE"] == 2.0


def test_available_refresh_after_fill_is_netted_too() -> None:
    """成交后只重查可用的那条路径同样按方向合并."""
    gw = _RowsGw(
        [
            UnifiedPosition(
                symbol="rb2601.SHFE", quantity=3, available_quantity=3, direction="long"
            ),
            UnifiedPosition(
                symbol="rb2601.SHFE",
                quantity=1,
                available_quantity=1,
                direction="short",
            ),
        ]
    )
    cache = BrokerStateCache(gw)
    assert cache.positions()["rb2601.SHFE"] == 2.0
    gw.rows[1] = UnifiedPosition(
        symbol="rb2601.SHFE", quantity=2, available_quantity=2, direction="short"
    )
    cache.invalidate_available()
    assert cache.available_positions()["rb2601.SHFE"] == 1.0
    assert cache.positions()["rb2601.SHFE"] == 2.0  # 总持仓不随可用重查而变
