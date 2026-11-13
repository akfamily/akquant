"""内置 broker builder 及注册入口（把内置 broker 变成 registry 插件）."""

from __future__ import annotations

from typing import Any, Sequence

from ...akquant import DataFeed
from ..protocols import GatewayBundle, MarketGateway, TraderGateway
from ..registry import register_broker
from .ctp.adapter import CTPMarketAdapter, CTPTraderAdapter
from .replay.gateway import build_replay_bundle

# CTP 交易通道的连接参数(经 gateway_options 传入), 六项必须同时给出。
_CTP_TRADER_KEYS = (
    "td_front",
    "broker_id",
    "user_id",
    "password",
    "app_id",
    "auth_code",
)


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
        raise ValueError("broker='ctp' 需要行情前置地址: gateway_options 缺少 md_front")
    # 交易通道: 六项连接参数要么都不给(只接行情), 要么全给。不替用户补
    # SimNow 的仿真默认值, 也不因缺一项就悄悄不建交易通道。先于建网关校验,
    # 配置写错时不必等连上行情前置才发现。
    trader_values = {key: kwargs.get(key) for key in _CTP_TRADER_KEYS}
    given = [key for key, value in trader_values.items() if value not in (None, "")]
    missing = [key for key in _CTP_TRADER_KEYS if key not in given]
    if given and missing:
        raise ValueError(
            "broker='ctp' 的交易通道参数不完整, gateway_options 缺少: "
            + ", ".join(missing)
            + "（只接行情时这六项都不要传）"
        )
    # emit_ticks/emit_bars 经 gateway_options 透传给 CTPMarketAdapter ->
    # CTPMarketGateway, 与 klinedata 的 _builder.py 同构。
    # 这里只是把 gateway_options 里的键原样转发, 不做任何回退/校验——那些
    # 逐参数回退(None 时按 use_aggregator 别名推导)与"两者都 False 报错"的
    # 规则都在 CTPMarketGateway.__init__ 里实现一次, 避免两处语义漂移。
    market_adapter = CTPMarketAdapter(
        feed=feed,
        front_url=md_front,
        symbols=list(symbols),
        use_aggregator=use_aggregator,
        emit_ticks=kwargs.get("emit_ticks"),
        emit_bars=kwargs.get("emit_bars"),
    )
    market_gateway: MarketGateway = market_adapter
    trader_gateway: TraderGateway | None = None
    if given:
        trader_gateway = CTPTraderAdapter(
            front_url=str(trader_values["td_front"]),
            broker_id=str(trader_values["broker_id"]),
            user_id=str(trader_values["user_id"]),
            password=str(trader_values["password"]),
            auth_code=str(trader_values["auth_code"]),
            app_id=str(trader_values["app_id"]),
            execution_semantics_mode=kwargs.get("execution_semantics_mode", "strict"),
        )
    return GatewayBundle(
        market_gateway=market_gateway,
        trader_gateway=trader_gateway,
        trader_capabilities=_resolve_trader_capabilities(trader_gateway),
        metadata={
            "broker": "ctp",
            # 同时推 tick 与合成 bar 时为双流(见 LiveRunner._inject_dual_stream)
            "dual_stream": bool(
                market_adapter.gateway.emit_ticks and market_adapter.gateway.emit_bars
            ),
        },
    )


def register_builtin_brokers() -> None:
    """Register all built-in brokers into the shared registry (idempotent)."""
    register_broker("ctp", _build_ctp)
    register_broker("replay", build_replay_bundle)
