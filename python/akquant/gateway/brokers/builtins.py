"""内置 broker builder 及注册入口（把内置 broker 变成 registry 插件）."""

from __future__ import annotations

from typing import Any, Sequence

from ...akquant import DataFeed
from ..protocols import GatewayBundle, MarketGateway, TraderGateway
from ..registry import register_broker
from .ctp.adapter import CTPMarketAdapter, CTPTraderAdapter
from .miniqmt.stub import MiniQMTMarketGateway, MiniQMTTraderGateway
from .ptrade.stub import PTradeMarketGateway, PTradeTraderGateway
from .replay.gateway import build_replay_bundle


def _resolve_trader_capabilities(trader_gateway: TraderGateway | None) -> Any:
    if trader_gateway is None:
        return None
    get_capabilities = getattr(trader_gateway, "get_capabilities", None)
    return get_capabilities() if callable(get_capabilities) else None


def _build_ctp(
    feed: DataFeed, symbols: Sequence[str], use_aggregator: bool, **kwargs: Any
) -> GatewayBundle:
    md_front = kwargs.get("md_front", "")
    if not md_front:
        raise ValueError("md_front is required when broker='ctp'")
    # emit_ticks/emit_bars 经 gateway_options 透传给 CTPMarketAdapter ->
    # CTPMarketGateway, 与 klinedata 的 _builder.py 同构(见其 96-97 行)。
    # 这里只是把 gateway_options 里的键原样转发, 不做任何回退/校验——那些
    # 逐参数回退(None 时按 use_aggregator 别名推导)与"两者都 False 报错"的
    # 规则都在 CTPMarketGateway.__init__ 里实现一次, 避免两处语义漂移。
    market_gateway: MarketGateway = CTPMarketAdapter(
        feed=feed,
        front_url=md_front,
        symbols=list(symbols),
        use_aggregator=use_aggregator,
        emit_ticks=kwargs.get("emit_ticks"),
        emit_bars=kwargs.get("emit_bars"),
    )
    trader_gateway: TraderGateway | None = None
    td_front = kwargs.get("td_front")
    user_id = kwargs.get("user_id")
    if td_front and user_id:
        trader_gateway = CTPTraderAdapter(
            front_url=td_front,
            broker_id=kwargs.get("broker_id", "9999"),
            user_id=user_id,
            password=kwargs.get("password", ""),
            auth_code=kwargs.get("auth_code", "0000000000000000"),
            app_id=kwargs.get("app_id", "simnow_client_test"),
            execution_semantics_mode=kwargs.get("execution_semantics_mode", "strict"),
        )
    return GatewayBundle(
        market_gateway=market_gateway,
        trader_gateway=trader_gateway,
        trader_capabilities=_resolve_trader_capabilities(trader_gateway),
        metadata={"broker": "ctp"},
    )


def _build_miniqmt(
    feed: DataFeed, symbols: Sequence[str], use_aggregator: bool, **kwargs: Any
) -> GatewayBundle:
    market_gateway = MiniQMTMarketGateway(feed=feed, symbols=list(symbols), **kwargs)
    trader_gateway: TraderGateway | None = MiniQMTTraderGateway(**kwargs)
    return GatewayBundle(
        market_gateway=market_gateway,
        trader_gateway=trader_gateway,
        trader_capabilities=_resolve_trader_capabilities(trader_gateway),
        metadata={"broker": "miniqmt"},
    )


def _build_ptrade(
    feed: DataFeed, symbols: Sequence[str], use_aggregator: bool, **kwargs: Any
) -> GatewayBundle:
    market_gateway = PTradeMarketGateway(feed=feed, symbols=list(symbols), **kwargs)
    trader_gateway: TraderGateway | None = PTradeTraderGateway(**kwargs)
    return GatewayBundle(
        market_gateway=market_gateway,
        trader_gateway=trader_gateway,
        trader_capabilities=_resolve_trader_capabilities(trader_gateway),
        metadata={"broker": "ptrade"},
    )


def register_builtin_brokers() -> None:
    """Register all built-in brokers into the shared registry (idempotent)."""
    register_broker("ctp", _build_ctp)
    register_broker("miniqmt", _build_miniqmt)
    register_broker("ptrade", _build_ptrade)
    register_broker("replay", build_replay_bundle)
