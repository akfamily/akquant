"""Tests for the trade/order stream bridge (frontend-facing message helper).

Mirrors ``test_indicator_recording.py``'s bridge tests: the engine already emits
``order`` / ``trade`` stream events; these helpers normalize them into the same
frontend-friendly envelope shape as the indicator bridge so a single consumer can
draw entry/exit markers alongside indicators.
"""

import akquant
import pandas as pd
from akquant import (
    Bar,
    Strategy,
    is_trade_stream_event,
    run_backtest,
    to_trade_message,
    to_trade_messages,
)


def _build_data() -> list[Bar]:
    closes = [10.0, 10.5, 11.0, 10.8]
    bars: list[Bar] = []
    for i, close in enumerate(closes):
        bars.append(
            Bar(
                timestamp=pd.Timestamp(f"2024-01-0{i + 1} 10:00:00").value,
                open=close - 0.1,
                high=close + 0.2,
                low=close - 0.2,
                close=close,
                volume=1000.0,
                symbol="TRD",
            )
        )
    return bars


class _TradingStrategy(Strategy):
    """Buy on the first bar and sell on the third to generate fills."""

    def on_bar(self, bar: Bar) -> None:
        """Trade so the engine emits order/trade stream events."""
        if bar.close == 10.0:
            self.buy(symbol=bar.symbol, quantity=10)
        elif bar.close == 11.0:
            self.sell(symbol=bar.symbol, quantity=10)


def _run_events() -> list[akquant.BacktestStreamEvent]:
    events: list[akquant.BacktestStreamEvent] = []
    run_backtest(
        data=_build_data(),
        strategy=_TradingStrategy,
        symbols="TRD",
        initial_cash=100000.0,
        show_progress=False,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        on_event=events.append,
        stream_batch_size=1,
        stream_max_buffer=256,
    )
    return events


def test_is_trade_stream_event_discriminates() -> None:
    """Only order/trade events are recognized by the trade bridge."""
    events = _run_events()
    trade_events = [e for e in events if is_trade_stream_event(e)]
    assert trade_events, "expected at least one order/trade event"
    assert all(e["event_type"] in {"order", "trade"} for e in trade_events)
    # indicator/progress/equity events are not trade events
    assert any(not is_trade_stream_event(e) for e in events)


def test_to_trade_message_normalizes_trade_fill() -> None:
    """A trade fill maps to a frontend marker message with numeric fields."""
    events = _run_events()
    trade_event = next(e for e in events if e["event_type"] == "trade")

    message = to_trade_message(trade_event)

    assert message is not None
    assert message["channel"] == "trade"
    assert message["type"] == "trade"
    assert message["symbol"] == "TRD"
    fill = message["fill"]
    assert fill["trade_id"]
    assert fill["order_id"]
    assert isinstance(fill["price"], float)
    assert isinstance(fill["quantity"], float)
    assert fill["price"] > 0.0
    # side is carried through so the frontend can pick buy/sell marker direction
    assert fill["side"] in {"Buy", "Sell"}
    # a chart marker anchors on the event's second-precision time
    assert isinstance(message["time"], int)
    assert message["time"] > 0


def test_trade_fills_carry_buy_and_sell_sides() -> None:
    """Entry and exit fills expose distinct Buy/Sell sides for marker direction."""
    events = _run_events()
    fills = [
        to_trade_message(e)["fill"]  # type: ignore[index]
        for e in events
        if e["event_type"] == "trade"
    ]
    sides = {f["side"] for f in fills}
    # the strategy buys on the first bar and sells on the third
    assert "Buy" in sides
    assert "Sell" in sides


def test_to_trade_message_normalizes_order() -> None:
    """An order event maps to an order-type message carrying status."""
    events = _run_events()
    order_event = next(e for e in events if e["event_type"] == "order")

    message = to_trade_message(order_event)

    assert message is not None
    assert message["type"] == "order"
    order = message["order"]
    assert order["order_id"]
    assert order["symbol"] == "TRD"
    assert order["status"]


def test_to_trade_message_ignores_non_trade_events() -> None:
    """Indicator and lifecycle events are not converted."""
    started = akquant.BacktestStreamEvent(
        run_id="demo",
        seq=1,
        ts=0,
        event_type="started",
        symbol=None,
        level="info",
        payload={"status": "started"},
    )
    assert is_trade_stream_event(started) is False
    assert to_trade_message(started) is None


def test_to_trade_messages_filters_and_preserves_order() -> None:
    """The batch helper keeps only trade/order messages in sequence order."""
    events = _run_events()
    messages = to_trade_messages(events)

    assert messages
    assert all(m["channel"] == "trade" for m in messages)
    assert {m["type"] for m in messages} <= {"order", "trade"}
    seqs = [m["seq"] for m in messages]
    assert seqs == sorted(seqs)


def test_trade_message_envelope_matches_indicator_bridge() -> None:
    """Trade and indicator bridges share the same envelope keys (one consumer)."""
    from akquant import to_indicator_message

    events = _run_events()
    trade_msg = to_trade_message(next(e for e in events if e["event_type"] == "trade"))

    ind_events: list[akquant.BacktestStreamEvent] = []

    class _IndStrat(Strategy):
        def on_bar(self, bar: Bar) -> None:
            self.record_indicator(name="c", value=bar.close, pane=0)

    run_backtest(
        data=_build_data(),
        strategy=_IndStrat,
        symbols="TRD",
        initial_cash=100000.0,
        show_progress=False,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        on_event=ind_events.append,
        stream_batch_size=1,
        stream_max_buffer=256,
    )
    ind_msg = to_indicator_message(
        next(e for e in ind_events if e["event_type"] == "indicator_point")
    )

    assert trade_msg is not None and ind_msg is not None
    envelope = {
        "channel",
        "type",
        "run_id",
        "seq",
        "ts",
        "symbol",
        "level",
        "schema_version",
    }
    assert envelope <= set(trade_msg.keys())
    assert envelope <= set(ind_msg.keys())


def test_trade_message_carries_schema_version() -> None:
    """Every trade message stamps the shared stream schema version."""
    from akquant import STREAM_SCHEMA_VERSION

    events = _run_events()
    message = to_trade_message(next(e for e in events if e["event_type"] == "trade"))

    assert message is not None
    assert message["schema_version"] == STREAM_SCHEMA_VERSION
