"""CTP 期权合约表: 合约行转换、中金所调整系数反推、按时段缓存、运行中刷新的差异判定.

查询本身需要 CTP 前置(SimNow / 期货公司), 这里用与 CTP 回报同字段的合成数据覆盖
转换与缓存逻辑。
"""

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pytest
from akquant import AssetType, OptionMarginModel, OptionType
from akquant.gateway import CTPOptionUniverse
from akquant.gateway.brokers.ctp import instruments as ctp_instruments
from akquant.gateway.brokers.ctp.instruments import (
    build_instruments,
    index_margin_coefficient,
    select_option_rows,
)
from akquant.live._instrument_refresh import InstrumentRefresher

TZ = ZoneInfo("Asia/Shanghai")
CONNECTION = {
    "td_front": "tcp://test",
    "broker_id": "9999",
    "user_id": "u",
    "password": "p",
    "app_id": "app",
    "auth_code": "0",
}
INDEX = {"IO": "000300.SH"}


def _futures(symbol: str, product: str) -> Dict[str, Any]:
    return {
        "InstrumentID": symbol,
        "ProductID": product,
        "ProductClass": "1",
        "VolumeMultiple": 10,
        "PriceTick": 1.0,
        "ExpireDate": "20270115",
    }


def _option(
    symbol: str,
    underlying: str,
    kind: str = "1",
    strike: float = 3000.0,
    product: str = "m_o",
    product_class: str = "2",
    multiple: int = 10,
) -> Dict[str, Any]:
    return {
        "InstrumentID": symbol,
        "ExchangeID": "CFFEX" if product_class == "6" else "DCE",
        "ProductID": product,
        "ProductClass": product_class,
        "OptionsType": kind,
        "StrikePrice": strike,
        "ExpireDate": "20261207",
        "UnderlyingInstrID": underlying,
        "VolumeMultiple": multiple,
        "PriceTick": 0.5,
    }


ROWS = [
    _futures("m2701", "m"),
    _futures("m2705", "m"),
    _futures("IF2612", "IF"),
    _option("m2701-C-3000", "m2701"),
    _option("m2701-P-3000", "m2701", kind="2"),
    _option("m2705-C-3100", "m2705", strike=3100.0),
    _option(
        "IO2612-C-4400",
        "IO2612",
        strike=4400.0,
        product="IO",
        product_class="6",
        multiple=100,
    ),
    _option("IO2612-P-4400", "IO2612", "2", 4400.0, "IO", "6", 100),
]


def _rate(by_money: float, by_volume: float = 0.0) -> List[Dict[str, Any]]:
    return [
        {"ShortMarginRatioByMoney": by_money, "ShortMarginRatioByVolume": by_volume}
    ]


def _probes(itm: float = 0.12, otm: float = 0.12) -> List[Dict[str, Any]]:
    """按 CHINA_INDEX 公式构造实值、虚值两次交易成本(行权价 4400, 乘数 100)."""
    return [
        {
            "label": "itm",
            "underlying_price": 5280.0,
            "multiplier": 100,
            "fixed_margin": 100 * 5280.0 * itm,
        },
        {
            "label": "otm",
            "underlying_price": 3520.0,
            "multiplier": 100,
            "fixed_margin": 100 * 0.5 * otm * 3520.0,
        },
    ]


def _snapshot(underlyings: List[str], **override: Any) -> Dict[str, Any]:
    options, futures = select_option_rows(ROWS, underlyings)
    snapshot = {
        "version": ctp_instruments.CACHE_VERSION,
        "trading_day": "20261123",
        "underlyings": underlyings,
        "options": options,
        "futures": futures,
        "margin_rates": {
            str(f["InstrumentID"]): _rate(0.14000000000000004) for f in futures
        },
        "index_probes": {"IO": _probes()}
        if any(o["ProductClass"] == "6" for o in options)
        else {},
    }
    snapshot.update(override)
    return snapshot


def test_select_by_product_or_underlying_contract() -> None:
    """标的品种代码选出全部月份, 标的合约代码只选该月; 中金所按期权品种选."""
    options, futures = select_option_rows(ROWS, ["m"])
    assert [o["InstrumentID"] for o in options] == [
        "m2701-C-3000",
        "m2701-P-3000",
        "m2705-C-3100",
    ]
    assert [f["InstrumentID"] for f in futures] == ["m2701", "m2705"]

    options, futures = select_option_rows(ROWS, ["M2705", "io"])
    assert [o["InstrumentID"] for o in options] == [
        "m2705-C-3100",
        "IO2612-C-4400",
        "IO2612-P-4400",
    ]
    assert [f["InstrumentID"] for f in futures] == ["m2705"]

    with pytest.raises(ValueError, match="CTP 合约表里没有以下标的的期权: rb"):
        select_option_rows(ROWS, ["m", "rb"])


def test_build_commodity_and_index_instruments() -> None:
    """商品期权跟随标的期货(CHINA_COMMODITY); 中金所期权标的为指定指数(CHINA_INDEX)."""
    instruments = {
        i.symbol: i for i in build_instruments(_snapshot(["m2701", "IO"]), INDEX)
    }
    assert sorted(instruments) == [
        "IO2612-C-4400",
        "IO2612-P-4400",
        "m2701",
        "m2701-C-3000",
        "m2701-P-3000",
    ]

    futures = instruments["m2701"]
    assert futures.asset_type == AssetType.Futures
    assert (futures.multiplier, futures.tick_size) == (10.0, 1.0)
    assert futures.margin_ratio == pytest.approx(0.14)
    assert futures.expiry_date == 20270115

    put = instruments["m2701-P-3000"]
    assert put.asset_type == AssetType.Option
    assert put.option_type == OptionType.Put
    assert (put.strike_price, put.expiry_date, put.underlying_symbol) == (
        3000.0,
        20261207,
        "m2701",
    )
    assert put.option_margin_model == OptionMarginModel.ChinaCommodity

    index_call = instruments["IO2612-C-4400"]
    assert index_call.underlying_symbol == "000300.SH"
    assert index_call.option_margin_model == OptionMarginModel.ChinaIndex
    assert index_call.margin_ratio == pytest.approx(0.12)
    assert index_call.multiplier == 100.0


def test_build_reports_exact_causes() -> None:
    """缺指数映射、按手数收保证金、调整系数两分支不一致时逐项报出原因."""
    with pytest.raises(
        ValueError, match="index_underlyings 没有给出 IO 对应的标的指数代码"
    ):
        build_instruments(_snapshot(["IO"]), {})
    with pytest.raises(ValueError, match=r"m2701 的保证金按手数收取\(500 元/手\)"):
        build_instruments(
            _snapshot(["m2701"], margin_rates={"m2701": _rate(0.0, 500.0)}), INDEX
        )
    with pytest.raises(ValueError, match="期货公司没有返回 m2701 的保证金率"):
        build_instruments(_snapshot(["m2701"], margin_rates={"m2701": []}), INDEX)
    with pytest.raises(
        ValueError,
        match="IO 反推的保证金调整系数不一致: 实值分支 0.12000000, 虚值分支 0.15000000",
    ):
        index_margin_coefficient("IO", _probes(itm=0.12, otm=0.15))


def test_refresh_points_follow_session_times() -> None:
    """缓存键取最近一个已过的刷新时刻; 凌晨仍属前一晚 20:30 的夜盘时段."""
    universe = CTPOptionUniverse(CONNECTION, ["m"])

    def at(day: int, hour: int, minute: int = 0) -> dt.datetime:
        return dt.datetime(2026, 11, day, hour, minute, tzinfo=TZ)

    assert universe.refresh_point(at(23, 1)) == at(22, 20, 30)
    assert universe.refresh_point(at(23, 8, 30)) == at(23, 8, 30)
    assert universe.refresh_point(at(23, 15)) == at(23, 8, 30)
    assert universe.refresh_point(at(23, 21)) == at(23, 20, 30)
    assert universe.next_refresh(at(23, 1)) == at(23, 8, 30)
    assert universe.next_refresh(at(23, 21)) == at(24, 8, 30)


def test_load_reads_cache_within_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同一时段内 load 读缓存不再查询; 进入下一时段或跟踪标的变化时重新查询."""
    calls: List[List[str]] = []

    def fake_fetch(
        connection: Dict[str, str], underlyings: List[str], flow_dir: Path
    ) -> Dict[str, Any]:
        calls.append(list(underlyings))
        return _snapshot(list(underlyings))

    monkeypatch.setattr(ctp_instruments, "fetch_snapshot", fake_fetch)
    morning = dt.datetime(2026, 11, 23, 9, 0, tzinfo=TZ)
    universe = CTPOptionUniverse(CONNECTION, ["m"], cache_dir=tmp_path)

    first = universe.load(morning)
    again = universe.load(morning + dt.timedelta(hours=5))
    assert len(calls) == 1
    assert [i.symbol for i in again] == [i.symbol for i in first]
    assert (tmp_path / "ctp_options_20261123_0830.json").exists()

    universe.load(morning + dt.timedelta(hours=12))  # 21:00, 夜盘时段
    CTPOptionUniverse(CONNECTION, ["m2705"], cache_dir=tmp_path).load(morning)
    assert calls == [["m"], ["m"], ["m2705"]]


class _Source:
    def __init__(self, batches: List[List[Any]]) -> None:
        self.batches = batches

    def refresh(self, now: Optional[dt.datetime] = None) -> List[Any]:
        return self.batches.pop(0)


def test_refresher_applies_only_new_and_changed() -> None:
    """刷新只交出新增与参数变化的合约, 未变化的不重复登记."""
    old = build_instruments(_snapshot(["m2701"]), INDEX)
    changed = _snapshot(
        ["m"], margin_rates={"m2701": _rate(0.16), "m2705": _rate(0.14)}
    )
    fresh = build_instruments(changed, INDEX)
    applied: List[List[str]] = []
    refresher = InstrumentRefresher(
        _Source([fresh, fresh]),
        old,
        lambda insts: applied.append([i.symbol for i in insts]),
    )

    refresher.refresh_once()
    refresher.refresh_once()

    assert applied == [["m2701", "m2705", "m2705-C-3100"]]
