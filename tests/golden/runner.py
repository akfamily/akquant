"""
Golden Test Runner.

Executes a suite of regression tests against baseline results to ensure
backtesting engine stability and correctness.
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import akquant as aq
import numpy as np
import pandas as pd
from akquant import (
    BacktestConfig,
    BacktestStreamEvent,
    InstrumentConfig,
    StrategyConfig,
    run_backtest,
)
from akquant.backtest.result import BacktestResult

sys.path.append(str(Path(__file__).parent / "strategies"))

from futures_margin import FutureMarginStrategy  # noqa: E402
from futures_night_weekend import FuturesNightWeekendStrategy  # noqa: E402
from option_basic import OptionBasicStrategy  # noqa: E402
from order_cancel import OrderCancelStrategy  # noqa: E402
from stock_minute_auction import StockMinuteAuctionStrategy  # noqa: E402
from stock_t1 import StockT1Strategy  # noqa: E402

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
BASELINE_DIR = BASE_DIR / "baselines"
CURRENT_DIR = BASE_DIR / "current"

# Ensure directories exist
BASELINE_DIR.mkdir(parents=True, exist_ok=True)
CURRENT_DIR.mkdir(parents=True, exist_ok=True)

TOLERANCE = {
    "equity_cosine": 1e-6,
    "metrics_abs_diff": 1e-4,  # Generic tolerance
    "metrics_pct_diff": 0.01,  # 1% relative
}


# 规则 2.0.0 新增的两个分钟线时段场景(数据见 gen_data.py 同名函数)。
STOCK_MINUTE_AUCTION_CONFIG: Dict[str, Any] = {
    "t_plus_one": True,
    "lot_size": 100,
    "commission_rate": 0.0003,
    "min_commission": 5.0,
    "stamp_tax_rate": 0.001,
    "transfer_fee_rate": 0.00002,
    "run_kwargs": {"freq": "1min"},
}


def futures_night_weekend_config() -> Dict[str, Any]:
    """Config for the rb night-session weekend scenario."""
    rb_config = InstrumentConfig(
        symbol="rb2405",
        asset_type="FUTURES",
        multiplier=10.0,
        margin_ratio=0.1,
        tick_size=1.0,
    )
    return {
        "initial_cash": 100000.0,
        "instruments_config": [rb_config],
        "commission_rate": 0.0001,
        "run_kwargs": {"freq": "1min", "sessions": {"rb": "CN_FUTURES_NIGHT_23"}},
    }


def get_file_hash(path: Path) -> str:
    """Calculate SHA256 hash of a file.

    Args:
        path: Path to the file.

    Returns:
        Hex digest of the hash.
    """
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def save_result(result: BacktestResult, out_dir: Path) -> None:
    """Save backtest results to disk.

    Args:
        result: The backtest result object.
        out_dir: Directory to save results.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save Equity Curve
    if not result.equity_curve.empty:
        # Convert Series to DataFrame with timestamp column
        df_eq = result.equity_curve.reset_index()
        df_eq.columns = ["timestamp", "equity"]
        df_eq.to_parquet(out_dir / "equity_curve.parquet", index=False)

    # Save Orders

    if not result.orders_df.empty:
        orders = result.orders_df.sort_values("id").reset_index(drop=True)
        orders.to_parquet(out_dir / "orders.parquet", index=False)

    # Save Trades
    if not result.trades_df.empty:
        trades = result.trades_df.sort_values("entry_time").reset_index(drop=True)
        trades.to_parquet(out_dir / "trades.parquet", index=False)

    # Save Metrics
    metrics = {
        "total_return_pct": result.metrics.total_return_pct,
        "sharpe_ratio": result.metrics.sharpe_ratio,
        "max_drawdown_pct": result.metrics.max_drawdown_pct,
        "total_commission": result.trade_metrics.total_commission,
        "net_pnl": result.trade_metrics.net_pnl,
        "end_market_value": result.metrics.end_market_value,
    }

    # 交易日与收盘（日终结算）时刻: 锁住夜盘归属与结算时点。
    trading_day_ends = [
        {
            "trading_day": day.isoformat(),
            "close_ns": end_ns,
            "close_local": pd.Timestamp(end_ns, tz="Asia/Shanghai").isoformat(),
        }
        for day, end_ns in result.trading_day_ends
    ]
    with open(out_dir / "trading_day_ends.json", "w") as f:
        json.dump(trading_day_ends, f, indent=2)

    # Add metadata
    metrics["engine_rule_version"] = getattr(aq, "__engine_rule_version__", "unknown")
    metrics["akquant_version"] = aq.__version__

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def compare_results(baseline_dir: Path, current_dir: Path) -> List[str]:
    """Compare current results with baseline.

    Args:
        baseline_dir: Directory containing baseline results.
        current_dir: Directory containing current results.

    Returns:
        List of error messages.
    """
    errors = []

    # 1. Compare Equity Curve (Cosine Similarity)
    base_eq_path = baseline_dir / "equity_curve.parquet"
    curr_eq_path = current_dir / "equity_curve.parquet"

    if base_eq_path.exists() and curr_eq_path.exists():
        base_eq = pd.read_parquet(base_eq_path)
        curr_eq = pd.read_parquet(curr_eq_path)

        # Align length if needed (though golden test should be deterministic length)
        min_len = min(len(base_eq), len(curr_eq))
        if len(base_eq) != len(curr_eq):
            errors.append(
                f"Equity Curve Length Mismatch: Base {len(base_eq)} "
                f"vs Curr {len(curr_eq)}"
            )

        # Calculate cosine similarity on common part
        v1 = base_eq["equity"].iloc[:min_len].values
        v2 = curr_eq["equity"].iloc[:min_len].values

        dot = np.dot(np.asarray(v1), np.asarray(v2))
        norm = np.linalg.norm(np.asarray(v1)) * np.linalg.norm(np.asarray(v2))
        similarity = dot / norm if norm > 0 else 0

        if 1.0 - similarity > TOLERANCE["equity_cosine"]:
            errors.append(f"Equity Curve Cosine Similarity Too Low: {similarity:.8f}")

    elif base_eq_path.exists() != curr_eq_path.exists():
        errors.append("Equity Curve Existence Mismatch")

    # 2. Compare Metrics
    base_metrics_path = baseline_dir / "metrics.json"
    curr_metrics_path = current_dir / "metrics.json"

    if base_metrics_path.exists() and curr_metrics_path.exists():
        with open(base_metrics_path) as f:
            base_m = json.load(f)
        with open(curr_metrics_path) as f:
            curr_m = json.load(f)

        for key in [
            "total_return_pct",
            "sharpe_ratio",
            "max_drawdown_pct",
            "total_commission",
            "net_pnl",
        ]:
            val_base = base_m.get(key, 0.0)
            val_curr = curr_m.get(key, 0.0)

            diff = abs(val_curr - val_base)

            # Absolute check
            if diff > TOLERANCE["metrics_abs_diff"]:
                # Relative check
                if abs(val_base) > 1e-9:
                    rel_diff = diff / abs(val_base)
                    if rel_diff > TOLERANCE["metrics_pct_diff"]:
                        errors.append(
                            f"Metric {key} Mismatch: Base {val_base} "
                            f"vs Curr {val_curr} (Diff {diff})"
                        )
                else:
                    errors.append(
                        f"Metric {key} Mismatch: Base {val_base} "
                        f"vs Curr {val_curr} (Diff {diff})"
                    )

    # 3. Compare Trades Count
    base_trades_path = baseline_dir / "trades.parquet"
    curr_trades_path = current_dir / "trades.parquet"

    base_trades_count = (
        len(pd.read_parquet(base_trades_path)) if base_trades_path.exists() else 0
    )
    curr_trades_count = (
        len(pd.read_parquet(curr_trades_path)) if curr_trades_path.exists() else 0
    )

    if base_trades_count != curr_trades_count:
        errors.append(
            f"Trades Count Mismatch: Base {base_trades_count} "
            f"vs Curr {curr_trades_count}"
        )

    # 4. Compare Order Status Distribution
    # orders.parquet was already being written but never compared; order ids are
    # random UUIDs so we compare the per-status counts, which are deterministic.
    # This is what pins cancel/reject semantics — aggregates alone cannot.
    errors.extend(_compare_order_statuses(baseline_dir, current_dir))

    # 5. Compare Order Details (fill price / fill time / TIF)
    # 净值余弦与汇总指标对"同一笔单换了撮合时点"不敏感(如收盘竞价成交改成午休后
    # 13:01 连续竞价成交, 价差只有几分钱); 逐笔比对下单/成交时刻与成交均价才能
    # 锁住集合竞价、Day 单过期、夜盘归属这类时段语义。
    errors.extend(_compare_order_details(baseline_dir, current_dir))

    # 6. Compare Trading Day Ends (trading days and their close/settlement time)
    errors.extend(_compare_trading_day_ends(baseline_dir, current_dir))

    return errors


# 逐笔比对的订单列; 订单 UUID 每次随机, 按下单时刻等确定性字段排序后比对。
ORDER_DETAIL_COLUMNS = [
    "created_at",
    "updated_at",
    "side",
    "order_type",
    "quantity",
    "filled_quantity",
    "limit_price",
    "avg_price",
    "commission",
    "status",
    "time_in_force",
]
ORDER_SORT_COLUMNS = ["created_at", "side", "order_type", "quantity", "limit_price"]


def _order_details(orders_path: Path) -> List[Dict[str, Any]]:
    """Return deterministic order rows for one results dir (empty if absent)."""
    if not orders_path.exists():
        return []
    orders = pd.read_parquet(orders_path)
    columns = [c for c in ORDER_DETAIL_COLUMNS if c in orders.columns]
    sort_columns = [c for c in ORDER_SORT_COLUMNS if c in orders.columns]
    orders = orders.sort_values(sort_columns, na_position="last", kind="stable")
    return [
        {key: row[key] for key in columns}
        for row in orders[columns].to_dict(orient="records")
    ]


def _detail_value_equal(base: Any, curr: Any) -> bool:
    """Compare one order field: floats within 1e-9 (NaN equals NaN)."""
    if isinstance(base, float) and isinstance(curr, float):
        if np.isnan(base) and np.isnan(curr):
            return True
        return bool(abs(base - curr) <= 1e-9)
    return str(base) == str(curr)


def _compare_order_details(baseline_dir: Path, current_dir: Path) -> List[str]:
    """Compare per-order fields (times, prices, status) between runs."""
    base_rows = _order_details(baseline_dir / "orders.parquet")
    curr_rows = _order_details(current_dir / "orders.parquet")
    if len(base_rows) != len(curr_rows):
        return [f"Order Count Mismatch: Base {len(base_rows)} vs Curr {len(curr_rows)}"]
    errors: List[str] = []
    for index, (base_row, curr_row) in enumerate(zip(base_rows, curr_rows)):
        for key, base_value in base_row.items():
            curr_value = curr_row.get(key)
            if not _detail_value_equal(base_value, curr_value):
                errors.append(
                    f"Order[{index}] {key} Mismatch: Base {base_value} "
                    f"vs Curr {curr_value}"
                )
    return errors


def _compare_trading_day_ends(baseline_dir: Path, current_dir: Path) -> List[str]:
    """Compare trading days and close times (skipped if baseline lacks the file)."""
    base_path = baseline_dir / "trading_day_ends.json"
    curr_path = current_dir / "trading_day_ends.json"
    if not base_path.exists():
        return []
    if not curr_path.exists():
        return ["Trading Day Ends Missing In Current"]
    with open(base_path) as f:
        base_ends = [(e["trading_day"], e["close_ns"]) for e in json.load(f)]
    with open(curr_path) as f:
        curr_ends = [(e["trading_day"], e["close_ns"]) for e in json.load(f)]
    if base_ends == curr_ends:
        return []
    return [f"Trading Day Ends Mismatch: Base {base_ends} vs Curr {curr_ends}"]


def _order_status_counts(orders_path: Path) -> Dict[str, int]:
    """Return per-status order counts for one results dir (empty if absent)."""
    if not orders_path.exists():
        return {}
    orders = pd.read_parquet(orders_path)
    if "status" not in orders.columns:
        return {}
    counts = orders["status"].astype(str).value_counts().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def _compare_order_statuses(baseline_dir: Path, current_dir: Path) -> List[str]:
    """Compare the order status distribution between baseline and current."""
    base_counts = _order_status_counts(baseline_dir / "orders.parquet")
    curr_counts = _order_status_counts(current_dir / "orders.parquet")

    if base_counts == curr_counts:
        return []
    return [
        f"Order Status Distribution Mismatch: Base {dict(sorted(base_counts.items()))} "
        f"vs Curr {dict(sorted(curr_counts.items()))}"
    ]


def run_test(
    test_name: str,
    strategy_cls: Any,
    data_file: str,
    config: Dict[str, Any],
    generate_baseline: bool = False,
) -> List[str]:
    """Run a single golden test.

    Args:
        test_name: Name of the test.
        strategy_cls: Strategy class to run.
        data_file: Name of the data file.
        config: Test configuration.
        generate_baseline: Whether to generate baseline instead of comparing.

    Returns:
        List of error messages.
    """
    print(f"Running Golden Test: {test_name}...")

    # Load Data
    data_path = DATA_DIR / data_file
    df = pd.read_parquet(data_path)

    # Setup Config
    initial_cash = config.get("initial_cash", 100000.0)

    # Run Backtest
    # Need to handle instruments_config properly
    instr_configs = config.get("instruments_config", [])

    # Create BacktestConfig
    bc_config = BacktestConfig(
        strategy_config=StrategyConfig(initial_cash=initial_cash),
        instruments_config=instr_configs,
    )

    # Handle T+1 via kwarg if supported by run_backtest, or via market config
    t_plus_one = config.get("t_plus_one", False)

    # Run
    try:
        result = run_backtest(
            data=df,
            strategy=strategy_cls,
            config=bc_config,
            t_plus_one=t_plus_one,
            fill_policy=aq.NextOpen(),
            lot_size=config.get("lot_size", 1),
            commission_rate=config.get("commission_rate", 0.0),
            min_commission=config.get("min_commission", 0.0),
            stamp_tax_rate=config.get("stamp_tax_rate", 0.0),
            transfer_fee_rate=config.get("transfer_fee_rate", 0.0),
            **config.get("run_kwargs", {}),
        )
    except Exception as e:
        print(f"Backtest Failed: {e}")
        raise e

    # Determine Output Directory
    out_dir = BASELINE_DIR / test_name if generate_baseline else CURRENT_DIR / test_name

    # Save Results
    save_result(result, out_dir)

    # Compare if not generating baseline
    if not generate_baseline:
        base_dir = BASELINE_DIR / test_name
        if not base_dir.exists():
            print(
                f"Baseline not found for {test_name}. "
                "Please run with generate_baseline=True first."
            )
            return ["Baseline Missing"]

        errors = compare_results(base_dir, out_dir)
        if errors:
            print(f"FAILED: {test_name}")
            for err in errors:
                print(f"  - {err}")
            return errors
        else:
            print(f"PASSED: {test_name}")
            return []
    else:
        print(f"Baseline generated for {test_name}")
        return []


def run_stream_consistency_test(
    test_name: str,
    strategy_cls: Any,
    data_file: str,
    config: Dict[str, Any],
) -> List[str]:
    """Run one golden scenario in stream and non-stream modes and compare."""
    data_path = DATA_DIR / data_file
    df = pd.read_parquet(data_path)

    initial_cash = config.get("initial_cash", 100000.0)
    instr_configs = config.get("instruments_config", [])
    bc_config = BacktestConfig(
        strategy_config=StrategyConfig(initial_cash=initial_cash),
        instruments_config=instr_configs,
    )
    t_plus_one = config.get("t_plus_one", False)
    run_kwargs: Dict[str, Any] = {
        "data": df,
        "strategy": strategy_cls,
        "config": bc_config,
        "t_plus_one": t_plus_one,
        "fill_policy": aq.NextOpen(),
        "lot_size": config.get("lot_size", 1),
        "commission_rate": config.get("commission_rate", 0.0),
        "min_commission": config.get("min_commission", 0.0),
        "stamp_tax_rate": config.get("stamp_tax_rate", 0.0),
        "transfer_fee_rate": config.get("transfer_fee_rate", 0.0),
        "show_progress": False,
        **config.get("run_kwargs", {}),
    }

    normal_result = run_backtest(**run_kwargs)
    events: List[BacktestStreamEvent] = []
    stream_result = run_backtest(
        **run_kwargs,
        on_event=events.append,
        stream_progress_interval=8,
        stream_equity_interval=8,
        stream_batch_size=16,
        stream_max_buffer=256,
    )

    errors: List[str] = []
    if not events:
        errors.append(f"{test_name}: no stream events emitted")
        return errors
    if events[0].get("event_type") != "started":
        errors.append(f"{test_name}: first stream event is not started")
    if events[-1].get("event_type") != "finished":
        errors.append(f"{test_name}: last stream event is not finished")
    seq_values = [event["seq"] for event in events]
    if seq_values != sorted(seq_values):
        errors.append(f"{test_name}: stream seq is not monotonic")
    if len(stream_result.equity_curve) != len(normal_result.equity_curve):
        normal_equity_len = len(normal_result.equity_curve)
        stream_equity_len = len(stream_result.equity_curve)
        errors.append(
            f"{test_name}: equity length mismatch "
            f"{normal_equity_len} vs {stream_equity_len}"
        )
    if len(stream_result.trades) != len(normal_result.trades):
        normal_trades_len = len(normal_result.trades)
        stream_trades_len = len(stream_result.trades)
        errors.append(
            f"{test_name}: trades count mismatch "
            f"{normal_trades_len} vs {stream_trades_len}"
        )
    total_return_pct_diff = abs(
        stream_result.metrics.total_return_pct - normal_result.metrics.total_return_pct
    )
    if total_return_pct_diff > 1e-9:
        normal_total_return_pct = normal_result.metrics.total_return_pct
        stream_total_return_pct = stream_result.metrics.total_return_pct
        errors.append(
            f"{test_name}: total_return_pct mismatch "
            f"{normal_total_return_pct} vs {stream_total_return_pct}"
        )
    max_drawdown_pct_diff = abs(
        stream_result.metrics.max_drawdown_pct - normal_result.metrics.max_drawdown_pct
    )
    if max_drawdown_pct_diff > 1e-9:
        normal_max_drawdown_pct = normal_result.metrics.max_drawdown_pct
        stream_max_drawdown_pct = stream_result.metrics.max_drawdown_pct
        errors.append(
            f"{test_name}: max_drawdown_pct mismatch "
            f"{normal_max_drawdown_pct} vs {stream_max_drawdown_pct}"
        )
    return errors


def run_stream_consistency_suite() -> List[str]:
    """Run stream consistency checks for all golden scenarios."""
    failures: List[str] = []

    stock_errors = run_stream_consistency_test(
        "stock_t1",
        StockT1Strategy,
        "stock_t1.parquet",
        {
            "t_plus_one": True,
            "lot_size": 100,
            "commission_rate": 0.0003,
            "min_commission": 5.0,
            "stamp_tax_rate": 0.001,
            "transfer_fee_rate": 0.00002,
        },
    )
    failures.extend(stock_errors)

    future_config = InstrumentConfig(
        symbol="FUTURE_A",
        asset_type="FUTURES",
        multiplier=300.0,
        margin_ratio=0.1,
        tick_size=0.2,
    )
    future_errors = run_stream_consistency_test(
        "futures_margin",
        FutureMarginStrategy,
        "future_margin.parquet",
        {
            "initial_cash": 200000.0,
            "instruments_config": [future_config],
            "commission_rate": 0.0001,
        },
    )
    failures.extend(future_errors)

    option_config = InstrumentConfig(
        symbol="OPTION_A",
        asset_type="OPTION",
        multiplier=100.0,
        margin_ratio=1.0,
        tick_size=0.0001,
        option_type="CALL",
        strike_price=100.0,
        expiry_date=None,
        underlying_symbol="STOCK_A",
    )
    option_errors = run_stream_consistency_test(
        "option_basic",
        OptionBasicStrategy,
        "option_basic.parquet",
        {
            "initial_cash": 10000.0,
            "instruments_config": [option_config],
            "commission_rate": 0.0,
        },
    )
    failures.extend(option_errors)

    cancel_errors = run_stream_consistency_test(
        "order_cancel",
        OrderCancelStrategy,
        "order_cancel.parquet",
        {
            "initial_cash": 100000.0,
            "commission_rate": 0.0,
        },
    )
    failures.extend(cancel_errors)

    failures.extend(
        run_stream_consistency_test(
            "stock_minute_auction",
            StockMinuteAuctionStrategy,
            "stock_minute_auction.parquet",
            STOCK_MINUTE_AUCTION_CONFIG,
        )
    )
    failures.extend(
        run_stream_consistency_test(
            "futures_night_weekend",
            FuturesNightWeekendStrategy,
            "futures_night_weekend.parquet",
            futures_night_weekend_config(),
        )
    )

    return failures


def main(generate_baseline: bool = False) -> None:
    """Run all golden tests.

    Args:
        generate_baseline: Whether to generate new baselines.
    """
    failures = []

    # 1. Stock T+1 Test
    # Stock T+1: Buy Day 1, Sell Day 2 (OK), Buy Day 3, Sell Day 3 (Fail),
    # Sell Day 4 (OK)
    errs = run_test(
        "stock_t1",
        StockT1Strategy,
        "stock_t1.parquet",
        {
            "t_plus_one": True,
            "lot_size": 100,
            "commission_rate": 0.0003,
            "min_commission": 5.0,
            "stamp_tax_rate": 0.001,
            "transfer_fee_rate": 0.00002,
        },
        generate_baseline,
    )
    if errs:
        failures.append("stock_t1")

    # 2. Futures Margin Test
    # Buy 1 lot (margin OK), Buy 2nd lot (margin fail)
    # Multiplier 300, Margin 0.1, Price ~3500. Margin ~105,000.
    # Initial Cash 200,000.
    future_config = InstrumentConfig(
        symbol="FUTURE_A",
        asset_type="FUTURES",
        multiplier=300.0,
        margin_ratio=0.1,
        tick_size=0.2,
    )
    errs = run_test(
        "futures_margin",
        FutureMarginStrategy,
        "future_margin.parquet",
        {
            "initial_cash": 200000.0,
            "instruments_config": [future_config],
            "commission_rate": 0.0001,  # Futures commission
        },
        generate_baseline,
    )
    if errs:
        failures.append("futures_margin")

    # 3. Option Basic Test
    # Buy 1 Call, Price 50 -> 60. Multiplier 100.
    # Commission 5 per contract.
    option_config = InstrumentConfig(
        symbol="OPTION_A",
        asset_type="OPTION",
        multiplier=100.0,
        margin_ratio=1.0,  # Options usually full premium paid if buying
        tick_size=0.0001,
        option_type="CALL",
        strike_price=100.0,
        expiry_date=None,  # "20250101", # Integer/Timestamp conversion fix required
        underlying_symbol="STOCK_A",
    )

    # Note: akquant run_backtest doesn't support 'commission_per_contract'
    # easily via kwargs yet unless we pass a custom config object or specific
    # kwargs handled by run_backtest.
    # We will rely on default or pass it via config if supported.
    # Checking run_backtest signature... it takes **kwargs.
    # But let's assume standard commission rate for now or implement per-contract
    # support in runner if needed.
    # For now, let's use 0 commission to verify PnL cleanly.
    errs = run_test(
        "option_basic",
        OptionBasicStrategy,
        "option_basic.parquet",
        {
            "initial_cash": 10000.0,
            "instruments_config": [option_config],
            "commission_rate": 0.0,
        },
        generate_baseline,
    )
    if errs:
        failures.append("option_basic")

    # 4. Order Cancel Lifecycle Test
    # Rest 3 limit orders + 1 market buy; cancel one, query it across bars,
    # then cancel_all_orders. Pins the撤单 bookkeeping no other scenario covers.
    errs = run_test(
        "order_cancel",
        OrderCancelStrategy,
        "order_cancel.parquet",
        {
            "initial_cash": 100000.0,
            "commission_rate": 0.0,
        },
        generate_baseline,
    )
    if errs:
        failures.append("order_cancel")

    # 5. A-share 1-minute auction test (rule 2.0.0)
    # 600000.SH two days: day-1 continuous buy + Day limit order expiring at the
    # close; day-2 09:20 timer buy filled at the 09:25 opening-auction price and
    # 14:58 sell filled at the 15:00 closing-auction price.
    errs = run_test(
        "stock_minute_auction",
        StockMinuteAuctionStrategy,
        "stock_minute_auction.parquet",
        STOCK_MINUTE_AUCTION_CONFIG,
        generate_baseline,
    )
    if errs:
        failures.append("stock_minute_auction")

    # 6. Futures night session across the weekend (rule 2.0.0)
    # rb with CN_FUTURES_NIGHT_23: Friday night belongs to Monday; open on the
    # Friday night session, close on the Monday day session; settlements at
    # Friday 15:00 and Monday 15:00.
    errs = run_test(
        "futures_night_weekend",
        FuturesNightWeekendStrategy,
        "futures_night_weekend.parquet",
        futures_night_weekend_config(),
        generate_baseline,
    )
    if errs:
        failures.append("futures_night_weekend")

    if failures:
        print(f"\nGolden Suite FAILED: {failures}")
        exit(1)
    else:
        print("\nGolden Suite PASSED")
        exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate-baseline", action="store_true", help="Generate new baselines"
    )
    args = parser.parse_args()

    main(args.generate_baseline)
