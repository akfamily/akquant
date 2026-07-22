import json
import re
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    cast,
)

import pandas as pd

from ..akquant import BacktestResult as RustBacktestResult
from ..akquant import ClosedTrade, Order, Trade


class BacktestResult:
    """
    Backtest Result Wrapper.

    Wraps the underlying Rust BacktestResult to provide Python-friendly properties
    like DataFrames.
    """

    def __init__(
        self,
        raw_result: RustBacktestResult,
        timezone: str = "Asia/Shanghai",
        initial_cash: float = 0.0,
        strategy: Optional[Any] = None,
        engine: Optional[Any] = None,
        indicator_outputs: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ):
        """
        Initialize the BacktestResult wrapper.

        :param raw_result: The raw Rust BacktestResult object.
        :param timezone: The timezone string for datetime conversion.
        :param initial_cash: Initial capital used in backtest.
        :param strategy: The strategy instance used in backtest.
        :param engine: The engine instance used in backtest.
        """
        self._raw = raw_result
        self._timezone = timezone
        self.initial_cash = initial_cash
        self.strategy = strategy
        self.engine = engine
        self.analyzer_outputs: dict[str, dict[str, Any]] = {}
        self.resolved_execution_policy: Optional[dict[str, Any]] = None
        # 已解析运行时配置，供 save_checkpoint 持久化以便续跑继承 (issue #282)
        self.resolved_config: Optional[dict[str, Any]] = None
        self.stream_run_id: Optional[str] = None
        self.indicator_outputs: Dict[str, List[Dict[str, Any]]] = (
            indicator_outputs
            if indicator_outputs is not None
            else {"definitions": [], "instances": [], "points": []}
        )

    @property
    def equity_curve(self) -> pd.Series:
        """
        Get the equity curve as a Pandas Series.

        Index: Datetime (Timezone-aware)
        Values: Total equity
        """
        series = pd.Series(dtype=float)

        if self._raw.equity_curve:
            df = pd.DataFrame(self._raw.equity_curve, columns=["timestamp", "equity"])
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], unit="ns", utc=True
            ).dt.tz_convert(self._timezone)
            df.set_index("timestamp", inplace=True)
            series = df["equity"]

        # Fallback: Try to derive from snapshots via positions_df
        if series.empty:
            try:
                pos_df = self.positions_df
                if (
                    not pos_df.empty
                    and "equity" in pos_df.columns
                    and "date" in pos_df.columns
                ):
                    # WARNING: pos_df["equity"] might be Position Value,
                    # not Account Equity.
                    # We only use it if it looks like Account Equity
                    # (>= initial_cash * 0.5) or if we can't determine otherwise.
                    # Equity is account-level, unique per timestamp
                    df = pos_df[["date", "equity"]].drop_duplicates(subset=["date"])
                    df.set_index("date", inplace=True)
                    df.sort_index(inplace=True)
                    series = df["equity"]
            except Exception:
                pass

        # Post-processing: Handle PnL-only curve (starts at 0)
        if not series.empty and self.initial_cash > 0:
            # If the curve starts near 0 (e.g., < 10% of initial cash), assume it's PnL
            # and add initial_cash to convert to Equity.
            if abs(series.iloc[0]) < self.initial_cash * 0.1:
                series += self.initial_cash

            # Special case: If the curve is significantly below initial cash
            # (e.g., just position value), and we are in the fallback scenario,
            # we might want to warn or fix.
            # But adding initial_cash blindly to Position Value is also wrong.
            # For now, the PnL check handles the "0 to 250" case.

        return series

    @property
    def daily_returns(self) -> pd.Series:
        """
        Get daily returns of the strategy equity.

        Returns:
            pd.Series: Daily returns (percentage).
        """
        equity = self.equity_curve
        if equity.empty:
            return pd.Series(dtype=float)

        # Resample to daily and calculate percentage change
        # Assuming equity curve might have intraday data,
        # take the last value of each day
        # For intraday data, we need to be careful. 'D' means Calendar Day.
        # If we have multiple days, this works.
        daily_equity = equity.resample("D").last().ffill()
        returns = daily_equity.pct_change().fillna(0.0)
        return cast(pd.Series, returns)

    @property
    def cash_curve(self) -> pd.Series:
        """
        Get the cash curve as a Pandas Series.

        Index: Datetime (Timezone-aware)
        Values: Available cash
        """
        if not hasattr(self._raw, "cash_curve") or not self._raw.cash_curve:
            return pd.Series(dtype=float)

        df = pd.DataFrame(self._raw.cash_curve, columns=["timestamp", "cash"])
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], unit="ns", utc=True
        ).dt.tz_convert(self._timezone)
        df.set_index("timestamp", inplace=True)
        return df["cash"]

    @property
    def margin_curve(self) -> pd.Series:
        """
        Get the margin curve as a Pandas Series.

        Index: Datetime (Timezone-aware)
        Values: Used margin
        """
        if not hasattr(self._raw, "margin_curve") or not self._raw.margin_curve:
            return pd.Series(dtype=float)

        df = pd.DataFrame(self._raw.margin_curve, columns=["timestamp", "margin"])
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], unit="ns", utc=True
        ).dt.tz_convert(self._timezone)
        df.set_index("timestamp", inplace=True)
        return df["margin"]

    @staticmethod
    def _to_daily_curve(series: pd.Series) -> pd.Series:
        """
        Convert a curve series to daily end-of-day values.

        :param series: Input curve series with DatetimeIndex.
        :return: Daily curve using last value per day.
        """
        if series.empty:
            return pd.Series(dtype=float)
        daily_series = series.resample("D").last().dropna()
        return cast(pd.Series, daily_series)

    @property
    def equity_curve_daily(self) -> pd.Series:
        """
        Get daily equity curve as a Pandas Series.

        Index: Datetime (Timezone-aware)
        Values: Daily end-of-day equity
        """
        return self._to_daily_curve(self.equity_curve)

    @property
    def cash_curve_daily(self) -> pd.Series:
        """
        Get daily cash curve as a Pandas Series.

        Index: Datetime (Timezone-aware)
        Values: Daily end-of-day cash
        """
        return self._to_daily_curve(self.cash_curve)

    @property
    def margin_curve_daily(self) -> pd.Series:
        """
        Get daily margin curve as a Pandas Series.

        Index: Datetime (Timezone-aware)
        Values: Daily end-of-day used margin
        """
        return self._to_daily_curve(self.margin_curve)

    @property
    def trades(self) -> List[ClosedTrade]:
        """
        Get closed trades as a list of raw objects (Raw Access).

        These are the raw Rust objects, useful for iteration and accessing complex
        fields. For statistical analysis, use `trades_df`.
        """
        return cast(List[ClosedTrade], self._raw.trades)

    @property
    def orders(self) -> List[Order]:
        """
        Get orders as a list of raw objects (Raw Access).

        These are the raw Rust objects, useful for iteration and debugging.
        For statistical analysis, use `orders_df`.
        """
        if hasattr(self._raw, "orders"):
            return cast(List[Order], self._raw.orders)
        return []

    @property
    def metrics(self) -> Any:
        """Get metrics with timezone-aware datetime conversion."""
        metrics = self._raw.metrics

        class MetricsWrapper:
            def __init__(
                self, raw_metrics: Any, timezone: str, initial_cash: float
            ) -> None:
                self._raw = raw_metrics
                self._timezone = timezone
                self._initial_cash = initial_cash

            def __getattr__(self, name: str) -> Any:
                val = getattr(self._raw, name)
                if name in ["start_time", "end_time"]:
                    # Convert ns timestamp to datetime
                    if isinstance(val, int):
                        dt = pd.to_datetime(val, unit="ns", utc=True).tz_convert(
                            self._timezone
                        )
                        return dt
                return val

        return MetricsWrapper(metrics, self._timezone, self.initial_cash)

    @property
    def positions(self) -> pd.DataFrame:
        """
        Get positions history as a Pandas DataFrame.

        Index: Datetime (Timezone-aware)
        Columns: Symbols
        Values: Quantity.
        """
        if not self._raw.snapshots:
            return pd.DataFrame()

        # Extract data from snapshots
        data = []
        timestamps = []

        for ts, snapshots in self._raw.snapshots:
            timestamps.append(ts)
            # Create a dict for this timestamp: {symbol: quantity}
            row = {s.symbol: s.quantity for s in snapshots}
            data.append(row)

        df = pd.DataFrame(data, index=timestamps)

        # Convert nanosecond timestamp to datetime with timezone
        df.index = pd.to_datetime(df.index, unit="ns", utc=True).tz_convert(
            self._timezone
        )

        # Sort index just in case
        df = df.sort_index()

        # Fill missing values with 0.0
        df = df.fillna(0.0)

        return cast(pd.DataFrame, df)

    @property
    def positions_df(self) -> pd.DataFrame:
        """
        Get detailed positions history as a Pandas DataFrame (PyBroker style).

        Columns:
            - date (datetime): Snapshot time.
            - symbol (str): Trading symbol.
            - long_shares (float): Long position quantity.
            - short_shares (float): Short position quantity.
            - close (float): Closing price.
            - equity (float): Total account equity.
            - market_value (float): Market value of positions.
            - margin (float): Margin used.
            - unrealized_pnl (float): Floating PnL.
            - entry_price (float): Average entry price.
        """
        df = pd.DataFrame()

        # 1. Try IPC
        if hasattr(self._raw, "get_positions_ipc"):
            try:
                import importlib
                import io

                pa = importlib.import_module("pyarrow")

                ipc_bytes = self._raw.get_positions_ipc()
                if ipc_bytes and len(ipc_bytes) > 0:
                    reader = pa.ipc.open_stream(io.BytesIO(ipc_bytes))
                    df = reader.read_pandas()
            except (ImportError, Exception):
                pass

        # 2. Try Dict
        if df.empty and hasattr(self._raw, "get_positions_dict"):
            try:
                data = self._raw.get_positions_dict()
                if data and data.get("symbol"):  # Check if data exists
                    df = pd.DataFrame(data)
            except Exception:
                pass

        if df.empty:
            return pd.DataFrame()

        # Convert date to datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], unit="ns", utc=True).dt.tz_convert(
                self._timezone
            )

        # Reorder columns
        cols = [
            "long_shares",
            "short_shares",
            "close",
            "equity",
            "market_value",
            "margin",
            "unrealized_pnl",
            "entry_price",
            "symbol",
            "date",
        ]
        # Ensure all columns exist
        existing_cols = [c for c in cols if c in df.columns]
        df = df[existing_cols]

        # Sort
        if "symbol" in df.columns and "date" in df.columns:
            df = df.sort_values(by=["symbol", "date"])

        return cast(pd.DataFrame, df)

    @property
    def metrics_df(self) -> pd.DataFrame:
        """
        Get performance metrics as a Pandas DataFrame.

        Returns a DataFrame indexed by metric name with a single 'value' column,
        matching PyBroker's format.
        """
        df: pd.DataFrame = cast(pd.DataFrame, self._raw.metrics_df).copy()

        # Convert time fields to the configured timezone
        time_fields = ["start_time", "end_time"]
        for field in time_fields:
            if field in df.index:
                val = df.at[field, "value"]
                if val is not None:
                    try:
                        # Convert to pandas Timestamp for easy tz handling
                        ts = pd.Timestamp(cast(Any, val))
                        if ts.tz is not None:
                            df.at[field, "value"] = ts.tz_convert(self._timezone)
                    except Exception:
                        pass

        # Expose explicit user-facing trade state metrics.
        try:
            closed_trade_count = float(len(self.trades_df))
            df.loc["closed_trade_count", "value"] = closed_trade_count
        except Exception:
            pass

        try:
            execution_count = float(len(self.executions_df))
            df.loc["execution_count", "value"] = execution_count
        except Exception:
            pass

        try:
            open_position_count = 0.0
            pos_df = self.positions_df
            if not pos_df.empty and "date" in pos_df.columns:
                latest_date = pos_df["date"].max()
                latest_positions = pos_df[pos_df["date"] == latest_date].copy()
                if "quantity" in latest_positions.columns:
                    quantity = pd.to_numeric(
                        latest_positions["quantity"], errors="coerce"
                    ).fillna(0.0)
                    open_position_count = float(quantity.ne(0.0).sum())
                elif {
                    "long_shares",
                    "short_shares",
                }.issubset(latest_positions.columns):
                    long_shares = pd.to_numeric(
                        latest_positions["long_shares"], errors="coerce"
                    ).fillna(0.0)
                    short_shares = pd.to_numeric(
                        latest_positions["short_shares"], errors="coerce"
                    ).fillna(0.0)
                    open_position_count = float(
                        (long_shares.ne(0.0) | short_shares.ne(0.0)).sum()
                    )
            df.loc["open_position_count", "value"] = open_position_count
        except Exception:
            pass

        # Calculate additional margin/leverage metrics using snapshots
        try:
            pos_df = self.positions_df
            if (
                not pos_df.empty
                and "margin" in pos_df.columns
                and "equity" in pos_df.columns
            ):
                # Group by date to get account-level snapshot
                # Sum margin (used margin) and market_value
                # Equity should be the same for all symbols on the same date
                # (it's account level)
                # But to be safe, take the first one or mean (should be identical)
                daily_agg = pos_df.groupby("date").agg(
                    {
                        "margin": "sum",
                        "market_value": lambda x: x.abs().sum(),  # type: ignore # Gross
                        "equity": "first",
                    }
                )

                # Calculate Max Leverage
                # Leverage = Gross Market Value / Equity
                daily_agg["leverage"] = daily_agg["market_value"] / daily_agg["equity"]
                max_leverage = daily_agg["leverage"].max()

                # Calculate Min Margin Level (Safety)
                # Margin Level = Equity / Used Margin
                # Avoid division by zero
                daily_agg["margin_level"] = daily_agg.apply(
                    lambda row: (
                        row["equity"] / row["margin"]
                        if row["margin"] > 0
                        else float("inf")
                    ),
                    axis=1,
                )
                # Filter out inf (no margin used)
                valid_levels = daily_agg[daily_agg["margin_level"] != float("inf")][
                    "margin_level"
                ]
                min_margin_level = (
                    valid_levels.min() if not valid_levels.empty else float("inf")
                )

                df.loc["max_leverage", "value"] = max_leverage
                df.loc["min_margin_level", "value"] = (
                    min_margin_level if min_margin_level != float("inf") else 0.0
                )
        except Exception:
            # Fallback or ignore if calculation fails
            pass

        return cast(pd.DataFrame, df)

    @cached_property
    def orders_df(self) -> pd.DataFrame:
        """
        Get orders history as a Pandas DataFrame.

        Columns:
            - id (str): Order ID.
            - symbol (str): Trading symbol.
            - side (str): 'buy' or 'sell'.
            - order_type (str): 'market', 'limit', 'stop'.
            - quantity (float): Order quantity.
            - filled_quantity (float): Executed quantity.
            - limit_price (float): Price for limit orders.
            - stop_price (float): Trigger price for stop orders.
            - avg_price (float): Average execution price.
            - commission (float): Commission paid.
            - status (str): 'filled', 'cancelled', 'rejected', etc.
            - time_in_force (str): 'gtc', 'day', 'ioc', etc.
            - created_at (datetime): Creation time.
        """
        if not hasattr(self._raw, "orders_df"):
            return pd.DataFrame()

        df = cast(pd.DataFrame, self._raw.orders_df.copy())

        if df.empty:
            return df

        if "created_at" in df.columns:
            # Rust returns int64 timestamp (ns since epoch)
            if pd.api.types.is_numeric_dtype(df["created_at"]):
                df["created_at"] = pd.to_datetime(
                    df["created_at"], unit="ns", utc=True
                ).dt.tz_convert(self._timezone)
            elif hasattr(df["created_at"], "dt"):
                if df["created_at"].dt.tz is None:
                    df["created_at"] = (
                        df["created_at"]
                        .dt.tz_localize("UTC")
                        .dt.tz_convert(self._timezone)
                    )
                else:
                    df["created_at"] = df["created_at"].dt.tz_convert(self._timezone)

        if "updated_at" in df.columns:
            # Rust returns int64 timestamp (ns since epoch)
            if pd.api.types.is_numeric_dtype(df["updated_at"]):
                df["updated_at"] = pd.to_datetime(
                    df["updated_at"], unit="ns", utc=True
                ).dt.tz_convert(self._timezone)
            elif hasattr(df["updated_at"], "dt"):
                if df["updated_at"].dt.tz is None:
                    df["updated_at"] = (
                        df["updated_at"]
                        .dt.tz_localize("UTC")
                        .dt.tz_convert(self._timezone)
                    )
                else:
                    df["updated_at"] = df["updated_at"].dt.tz_convert(self._timezone)

        # Calculate derivative columns
        if "filled_quantity" in df.columns and "avg_price" in df.columns:
            # Calculate filled value (成交金额)
            df["filled_value"] = df["filled_quantity"] * df["avg_price"].fillna(0.0)

        if "created_at" in df.columns and "updated_at" in df.columns:
            # Calculate duration (存续时长)
            df["duration"] = df["updated_at"] - df["created_at"]

        if "owner_strategy_id" not in df.columns:
            owner_strategy_id = getattr(self, "_owner_strategy_id", None)
            if owner_strategy_id is not None:
                df["owner_strategy_id"] = owner_strategy_id

        # Sort by creation time for better readability
        if "created_at" in df.columns:
            df.sort_values(by="created_at", inplace=True)
            df.reset_index(drop=True, inplace=True)

        return df

    @cached_property
    def trades_df(self) -> pd.DataFrame:
        """
        Get closed trades as a Pandas DataFrame.

        Columns:
            - symbol (str): Trading symbol.
            - entry_time (datetime): Time of entry.
            - exit_time (datetime): Time of exit.
            - entry_price (float): Average entry price.
            - exit_price (float): Average exit price.
            - quantity (float): Traded quantity.
            - side (str): 'long' or 'short'.
            - pnl (float): Gross PnL.
            - net_pnl (float): Net PnL (after commission).
            - return_pct (float): Trade return (decimal).
            - commission (float): Commission paid.
            - duration_bars (int): Number of bars held.
            - duration (timedelta): Duration of trade.
            - mae (float): Maximum Adverse Excursion (%).
            - mfe (float): Maximum Favorable Excursion (%).
            - entry_tag (str): Tag of the entry order.
            - exit_tag (str): Tag of the exit order.
            - entry_portfolio_value (float): Portfolio value at entry.
            - max_drawdown_pct (float): Max drawdown % during trade.
        """
        if not self._raw.trades:
            return pd.DataFrame()

        df = pd.DataFrame()

        # 1. Try IPC (Zero-Copy-ish via Arrow)
        if hasattr(self._raw, "get_trades_ipc"):
            try:
                import importlib
                import io

                pa = importlib.import_module("pyarrow")

                ipc_bytes = self._raw.get_trades_ipc()
                if ipc_bytes and len(ipc_bytes) > 0:
                    reader = pa.ipc.open_stream(io.BytesIO(ipc_bytes))
                    df = reader.read_pandas()
            except (ImportError, Exception):
                # Fallback to other methods if pyarrow missing or IPC fails
                pass

        # 2. Try Dict (Fast)
        if df.empty and hasattr(self._raw, "get_trades_dict"):
            try:
                data_dict = self._raw.get_trades_dict()
                if data_dict:
                    df = pd.DataFrame(data_dict)
            except Exception:
                pass

        # 3. Fallback to List of Objects (Slow)
        if df.empty:
            data_list = []
            for t in self._raw.trades:
                data_list.append(
                    {
                        "symbol": t.symbol,
                        "entry_time": t.entry_time,
                        "exit_time": t.exit_time,
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "quantity": t.quantity,
                        "side": t.side,
                        "pnl": t.pnl,
                        "net_pnl": t.net_pnl,
                        "return_pct": t.return_pct,
                        "commission": t.commission,
                        "duration_bars": t.duration_bars,
                        "duration": t.duration,
                        "mae": t.mae,
                        "mfe": t.mfe,
                        "entry_tag": t.entry_tag,
                        "exit_tag": t.exit_tag,
                        "entry_portfolio_value": getattr(
                            t, "entry_portfolio_value", 0.0
                        ),
                        "max_drawdown_pct": getattr(t, "max_drawdown_pct", 0.0),
                    }
                )
            df = pd.DataFrame(data_list)

        if df.empty:
            return df

        # Convert timestamps
        # Check columns exist. IPC schema may vary across versions;
        # here we control the schema.
        if "entry_time" in df.columns:
            df["entry_time"] = pd.to_datetime(
                df["entry_time"], unit="ns", utc=True
            ).dt.tz_convert(self._timezone)

        if "exit_time" in df.columns:
            df["exit_time"] = pd.to_datetime(
                df["exit_time"], unit="ns", utc=True
            ).dt.tz_convert(self._timezone)

        # Convert duration to Timedelta
        if "duration" in df.columns:
            df["duration"] = pd.to_timedelta(df["duration"], unit="ns")

        return df

    @cached_property
    def executions_df(self) -> pd.DataFrame:
        """Get execution reports (fills) as a Pandas DataFrame."""
        executions = cast(List[Trade], getattr(self._raw, "executions", []))
        if not executions:
            return pd.DataFrame()

        df = pd.DataFrame()
        if hasattr(self._raw, "get_executions_ipc"):
            try:
                import importlib
                import io

                pa = importlib.import_module("pyarrow")
                ipc_bytes = self._raw.get_executions_ipc()
                if ipc_bytes and len(ipc_bytes) > 0:
                    reader = pa.ipc.open_stream(io.BytesIO(ipc_bytes))
                    df = reader.read_pandas()
            except (ImportError, Exception):
                pass

        if df.empty and hasattr(self._raw, "get_executions_dict"):
            try:
                data_dict = self._raw.get_executions_dict()
                if data_dict:
                    df = pd.DataFrame(data_dict)
            except Exception:
                pass

        if df.empty:
            rows: list[dict[str, Any]] = []
            fallback_owner_strategy_id = getattr(self, "_owner_strategy_id", None)
            for t in executions:
                rows.append(
                    {
                        "id": t.id,
                        "order_id": t.order_id,
                        "symbol": t.symbol,
                        "side": str(t.side).lower(),
                        "quantity": float(t.quantity),
                        "price": float(t.price),
                        "commission": float(t.commission),
                        "timestamp": t.timestamp,
                        "bar_index": t.bar_index,
                        "owner_strategy_id": getattr(
                            t, "owner_strategy_id", fallback_owner_strategy_id
                        ),
                    }
                )
            df = pd.DataFrame(rows)

        if "timestamp" in df.columns and pd.api.types.is_numeric_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], unit="ns", utc=True
            ).dt.tz_convert(self._timezone)
        return df

    @cached_property
    def liquidation_audit_df(self) -> pd.DataFrame:
        """Get margin forced-liquidation audit records as a Pandas DataFrame."""
        if not hasattr(self._raw, "get_liquidation_audits_dict"):
            return pd.DataFrame()

        try:
            data = self._raw.get_liquidation_audits_dict()
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data)
        except Exception:
            return pd.DataFrame()

        if df.empty:
            return df
        if "timestamp" in df.columns and pd.api.types.is_numeric_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], unit="ns", utc=True
            ).dt.tz_convert(self._timezone)
        if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(
            df["date"]
        ):
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)
        return cast(pd.DataFrame, df)

    def exposure_df(self, freq: Optional[str] = "D") -> pd.DataFrame:
        """Get portfolio exposure decomposition time series."""
        positions = self.positions_df.copy()
        columns = [
            "date",
            "equity",
            "long_exposure",
            "short_exposure",
            "net_exposure",
            "gross_exposure",
            "net_exposure_pct",
            "gross_exposure_pct",
            "leverage",
        ]
        if positions.empty or "date" not in positions.columns:
            return pd.DataFrame(columns=columns)

        numeric_cols = ["long_shares", "short_shares", "close", "equity"]
        for col in numeric_cols:
            if col not in positions.columns:
                positions[col] = 0.0
            positions[col] = pd.to_numeric(positions[col], errors="coerce").fillna(0.0)

        long_shares = positions["long_shares"].clip(lower=0.0)
        short_shares = positions["short_shares"].clip(lower=0.0)
        closes = positions["close"]
        positions["long_exposure"] = long_shares * closes
        positions["short_exposure"] = short_shares * closes

        exposure = positions.groupby("date", as_index=True).agg(
            {
                "long_exposure": "sum",
                "short_exposure": "sum",
                "equity": "first",
            }
        )
        exposure["net_exposure"] = (
            exposure["long_exposure"] - exposure["short_exposure"]
        )
        exposure["gross_exposure"] = (
            exposure["long_exposure"] + exposure["short_exposure"]
        )

        safe_equity = exposure["equity"].replace(0.0, pd.NA)
        exposure["net_exposure_pct"] = (exposure["net_exposure"] / safe_equity).fillna(
            0.0
        )
        exposure["gross_exposure_pct"] = (
            exposure["gross_exposure"] / safe_equity
        ).fillna(0.0)
        exposure["leverage"] = (exposure["gross_exposure"] / safe_equity).fillna(0.0)

        if freq:
            exposure = exposure.resample(freq).last().dropna(how="all")

        exposure = exposure.reset_index()
        return cast(pd.DataFrame, exposure[columns])

    def attribution_df(
        self, by: str = "symbol", use_net: bool = True, top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """Get grouped contribution analysis by symbol or tag."""
        trades = self.trades_df.copy()
        columns = [
            "group",
            "trade_count",
            "total_pnl",
            "avg_return_pct",
            "total_commission",
            "contribution_pct",
            "abs_contribution_pct",
        ]
        if trades.empty:
            return pd.DataFrame(columns=columns)

        value_col = "net_pnl" if use_net and "net_pnl" in trades.columns else "pnl"
        if value_col not in trades.columns:
            return pd.DataFrame(columns=columns)

        by_key = by.lower().strip()
        if by_key == "symbol":
            group_values = trades["symbol"].astype(str)
        elif by_key == "entry_tag":
            entry_fallback = pd.Series(index=trades.index, dtype=object)
            group_values = trades.get("entry_tag", entry_fallback).fillna("")
        elif by_key == "exit_tag":
            exit_fallback = pd.Series(index=trades.index, dtype=object)
            group_values = trades.get("exit_tag", exit_fallback).fillna("")
        elif by_key == "tag":
            entry_fallback = pd.Series(index=trades.index, dtype=object)
            exit_fallback = pd.Series(index=trades.index, dtype=object)
            entry_tags = trades.get("entry_tag", entry_fallback).fillna("")
            exit_tags = trades.get("exit_tag", exit_fallback).fillna("")
            group_values = entry_tags.where(entry_tags.astype(str) != "", exit_tags)
        else:
            raise ValueError("`by` must be one of: symbol, entry_tag, exit_tag, tag")

        trades = trades.copy()
        trades["group"] = group_values.astype(str).replace("", "_untagged")
        if "return_pct" not in trades.columns:
            trades["return_pct"] = 0.0
        if "commission" not in trades.columns:
            trades["commission"] = 0.0

        grouped = trades.groupby("group", as_index=False).agg(
            trade_count=(value_col, "count"),
            total_pnl=(value_col, "sum"),
            avg_return_pct=("return_pct", "mean"),
            total_commission=("commission", "sum"),
        )

        total_pnl = float(pd.to_numeric(grouped["total_pnl"], errors="coerce").sum())
        if total_pnl == 0.0:
            grouped["contribution_pct"] = 0.0
        else:
            grouped["contribution_pct"] = grouped["total_pnl"] / total_pnl
        grouped["abs_contribution_pct"] = grouped["contribution_pct"].abs()
        grouped = grouped.sort_values("total_pnl", ascending=False)
        grouped = grouped.reset_index(drop=True)

        if top_n is not None and top_n > 0:
            grouped = grouped.head(top_n).reset_index(drop=True)

        return cast(pd.DataFrame, grouped[columns])

    def capacity_df(self, freq: str = "D") -> pd.DataFrame:
        """Get execution capacity proxy metrics from orders and equity."""
        orders = self.orders_df.copy()
        columns = [
            "date",
            "order_count",
            "filled_order_count",
            "ordered_quantity",
            "filled_quantity",
            "ordered_value",
            "filled_value",
            "fill_rate_qty",
            "fill_rate_value",
            "equity",
            "turnover",
        ]
        if orders.empty:
            return pd.DataFrame(columns=columns)

        ts_col = "updated_at" if "updated_at" in orders.columns else "created_at"
        if ts_col not in orders.columns:
            return pd.DataFrame(columns=columns)

        numeric_cols = [
            "quantity",
            "filled_quantity",
            "filled_value",
            "avg_price",
            "limit_price",
        ]
        for col in numeric_cols:
            if col not in orders.columns:
                orders[col] = 0.0
            orders[col] = pd.to_numeric(orders[col], errors="coerce").fillna(0.0)

        price_proxy = orders["limit_price"].where(
            orders["limit_price"] > 0.0, orders["avg_price"]
        )
        orders["ordered_value"] = orders["quantity"] * price_proxy.fillna(0.0)
        orders["is_filled"] = (orders["filled_quantity"] > 0.0).astype(int)

        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(orders[ts_col]),
                "order_count": 1,
                "filled_order_count": orders["is_filled"],
                "ordered_quantity": orders["quantity"],
                "filled_quantity": orders["filled_quantity"],
                "ordered_value": orders["ordered_value"],
                "filled_value": orders["filled_value"],
            }
        ).set_index("date")

        grouped = frame.resample(freq).sum()

        equity = self.equity_curve
        if not equity.empty:
            equity_resampled = equity.resample(freq).last().ffill()
            grouped["equity"] = equity_resampled.reindex(grouped.index).ffill()
        else:
            grouped["equity"] = 0.0

        safe_qty = grouped["ordered_quantity"].replace(0.0, pd.NA)
        safe_value = grouped["ordered_value"].replace(0.0, pd.NA)
        safe_equity = grouped["equity"].replace(0.0, pd.NA)
        grouped["fill_rate_qty"] = (grouped["filled_quantity"] / safe_qty).fillna(0.0)
        grouped["fill_rate_value"] = (grouped["filled_value"] / safe_value).fillna(0.0)
        grouped["turnover"] = (grouped["filled_value"] / safe_equity).fillna(0.0)

        grouped = grouped.reset_index()
        return cast(pd.DataFrame, grouped[columns])

    def orders_by_strategy(self) -> pd.DataFrame:
        """Get strategy-level order aggregation table."""
        orders = self.orders_df.copy()
        columns = [
            "owner_strategy_id",
            "order_count",
            "filled_order_count",
            "ordered_quantity",
            "filled_quantity",
            "ordered_value",
            "filled_value",
            "fill_rate_qty",
            "fill_rate_value",
        ]
        if orders.empty:
            return pd.DataFrame(columns=columns)

        if "owner_strategy_id" not in orders.columns:
            orders["owner_strategy_id"] = "_default"
        orders["owner_strategy_id"] = (
            orders["owner_strategy_id"].fillna("_default").astype(str)
        )

        for col in [
            "quantity",
            "filled_quantity",
            "avg_price",
            "limit_price",
            "filled_value",
        ]:
            if col not in orders.columns:
                orders[col] = 0.0
        quantity = pd.to_numeric(orders["quantity"], errors="coerce").fillna(0.0)
        filled_quantity = pd.to_numeric(
            orders["filled_quantity"], errors="coerce"
        ).fillna(0.0)
        avg_price = pd.to_numeric(orders["avg_price"], errors="coerce").fillna(0.0)
        limit_price = pd.to_numeric(orders["limit_price"], errors="coerce").fillna(0.0)
        filled_value = pd.to_numeric(orders["filled_value"], errors="coerce").fillna(
            0.0
        )
        ordered_price = limit_price.where(limit_price > 0.0, avg_price)
        ordered_value = quantity * ordered_price.fillna(0.0)
        is_filled = (filled_quantity > 0.0).astype(int)

        frame = pd.DataFrame(
            {
                "owner_strategy_id": orders["owner_strategy_id"],
                "order_count": 1,
                "filled_order_count": is_filled,
                "ordered_quantity": quantity,
                "filled_quantity": filled_quantity,
                "ordered_value": ordered_value,
                "filled_value": filled_value,
            }
        )
        grouped = frame.groupby("owner_strategy_id", as_index=False).sum()
        safe_ordered_qty = grouped["ordered_quantity"].replace(0.0, pd.NA)
        safe_ordered_value = grouped["ordered_value"].replace(0.0, pd.NA)
        grouped["fill_rate_qty"] = (
            grouped["filled_quantity"] / safe_ordered_qty
        ).fillna(0.0)
        grouped["fill_rate_value"] = (
            grouped["filled_value"] / safe_ordered_value
        ).fillna(0.0)
        grouped = grouped.sort_values("owner_strategy_id").reset_index(drop=True)
        return cast(pd.DataFrame, grouped[columns])

    def executions_by_strategy(self) -> pd.DataFrame:
        """Get strategy-level execution aggregation table."""
        executions = self.executions_df.copy()
        columns = [
            "owner_strategy_id",
            "execution_count",
            "total_quantity",
            "total_notional",
            "total_commission",
            "avg_fill_price",
        ]
        if executions.empty:
            return pd.DataFrame(columns=columns)

        if "owner_strategy_id" not in executions.columns:
            executions["owner_strategy_id"] = "_default"
        executions["owner_strategy_id"] = (
            executions["owner_strategy_id"].fillna("_default").astype(str)
        )

        for col in ["quantity", "price", "commission"]:
            if col not in executions.columns:
                executions[col] = 0.0
        quantity = pd.to_numeric(executions["quantity"], errors="coerce").fillna(0.0)
        price = pd.to_numeric(executions["price"], errors="coerce").fillna(0.0)
        commission = pd.to_numeric(executions["commission"], errors="coerce").fillna(
            0.0
        )
        notional = quantity * price

        frame = pd.DataFrame(
            {
                "owner_strategy_id": executions["owner_strategy_id"],
                "execution_count": 1,
                "total_quantity": quantity,
                "total_notional": notional,
                "total_commission": commission,
            }
        )
        grouped = frame.groupby("owner_strategy_id", as_index=False).sum()
        safe_quantity = grouped["total_quantity"].replace(0.0, pd.NA)
        grouped["avg_fill_price"] = (grouped["total_notional"] / safe_quantity).fillna(
            0.0
        )
        grouped = grouped.sort_values("owner_strategy_id").reset_index(drop=True)
        return cast(pd.DataFrame, grouped[columns])

    def get_event_stats(self) -> dict[str, Any]:
        """Return normalized event stream statistics from payload and run summary."""
        keys = {
            "processed_events",
            "dropped_event_count",
            "callback_error_count",
            "backpressure_policy",
            "stream_mode",
            "sampling_enabled",
            "sampling_rate",
            "reason",
        }
        stats: dict[str, Any] = {}
        direct_stats = getattr(self, "_event_stats", None)
        if isinstance(direct_stats, dict):
            stats.update(direct_stats)
        try:
            metrics_df = self.metrics_df
            if not metrics_df.empty:
                value_col = "value" if "value" in metrics_df.columns else None
                for key in keys:
                    if key not in metrics_df.index:
                        continue
                    raw_value: Any
                    if value_col is not None:
                        raw_value = metrics_df.at[key, value_col]
                    else:
                        row = metrics_df.loc[key]
                        if isinstance(row, pd.Series):
                            raw_value = row.iloc[0]
                        else:
                            raw_value = row
                    if pd.isna(raw_value):
                        continue
                    if hasattr(raw_value, "item"):
                        try:
                            stats[key] = raw_value.item()
                            continue
                        except Exception:
                            pass
                    stats[key] = raw_value
        except Exception:
            pass

        summary = getattr(self, "_engine_summary", None)
        if isinstance(summary, str) and summary:
            parsed: dict[str, Any] = {}
            for part in summary.split(","):
                token = part.strip()
                if "=" not in token:
                    continue
                raw_key, raw_value = token.split("=", 1)
                key = raw_key.strip()
                value_text = raw_value.strip()
                lowered = value_text.lower()
                if lowered in {"true", "false"}:
                    parsed[key] = lowered == "true"
                else:
                    try:
                        parsed[key] = int(value_text)
                    except ValueError:
                        try:
                            parsed[key] = float(value_text)
                        except ValueError:
                            parsed[key] = value_text
            matched = re.search(r"Processed\s+(\d+)\s+events", summary, re.IGNORECASE)
            if matched and "processed_events" not in parsed:
                parsed["processed_events"] = int(matched.group(1))
            for key in keys:
                if key in parsed and key not in stats:
                    stats[key] = parsed[key]
        return stats

    def _risk_reason_masks(self, rejected: pd.DataFrame) -> dict[str, pd.Series]:
        reason_lower = rejected["reject_reason"].fillna("").astype(str).str.lower()
        daily_loss_mask = reason_lower.str.contains(
            "daily loss", regex=False
        ) | reason_lower.str.contains("stop-loss threshold", regex=False)
        drawdown_mask = reason_lower.str.contains("drawdown", regex=False)
        reduce_only_mask = reason_lower.str.contains(
            "reduce_only mode", regex=False
        ) | reason_lower.str.contains("cooldown", regex=False)
        position_limit_mask = reason_lower.str.contains(
            "projected position", regex=False
        ) | reason_lower.str.contains("available position", regex=False)
        position_limit_mask = position_limit_mask | reason_lower.str.contains(
            "is restricted", regex=False
        )
        order_size_mask = reason_lower.str.contains(
            "order quantity", regex=False
        ) | reason_lower.str.contains("lot size", regex=False)
        order_value_mask = reason_lower.str.contains("order value", regex=False)
        strategy_budget_mask = reason_lower.str.contains(
            "risk budget", regex=False
        ) & reason_lower.str.contains("strategy", regex=False)
        portfolio_budget_mask = reason_lower.str.contains(
            "portfolio risk budget", regex=False
        )
        known_mask = (
            daily_loss_mask
            | drawdown_mask
            | reduce_only_mask
            | position_limit_mask
            | order_size_mask
            | order_value_mask
            | strategy_budget_mask
            | portfolio_budget_mask
        )
        return {
            "daily_loss_reject_count": daily_loss_mask.astype(int),
            "drawdown_reject_count": drawdown_mask.astype(int),
            "reduce_only_reject_count": reduce_only_mask.astype(int),
            "position_limit_reject_count": position_limit_mask.astype(int),
            "order_size_limit_reject_count": order_size_mask.astype(int),
            "order_value_limit_reject_count": order_value_mask.astype(int),
            "strategy_risk_budget_reject_count": strategy_budget_mask.astype(int),
            "portfolio_risk_budget_reject_count": portfolio_budget_mask.astype(int),
            "other_risk_reject_count": (~known_mask).astype(int),
        }

    @staticmethod
    def _classify_reject_reason(reason: Any) -> str:
        """Map raw reject messages to stable report-friendly categories."""
        reason_text = str(reason).strip()
        if not reason_text:
            return "Empty Reject Reason"
        reason_lower = reason_text.lower()
        if "insufficient margin" in reason_lower:
            return "Insufficient Margin"
        if "daily loss" in reason_lower or "stop-loss threshold" in reason_lower:
            return "Daily Loss Limit"
        if "drawdown" in reason_lower:
            return "Drawdown Limit"
        if "reduce_only mode" in reason_lower or "cooldown" in reason_lower:
            return "Reduce-Only / Cooldown"
        if (
            "projected position" in reason_lower
            or "available position" in reason_lower
            or "is restricted" in reason_lower
        ):
            return "Position Limit"
        if "order quantity" in reason_lower or "lot size" in reason_lower:
            return "Order Size Limit"
        if "order value" in reason_lower:
            return "Order Value Limit"
        if "portfolio risk budget" in reason_lower:
            return "Portfolio Risk Budget"
        if "risk budget" in reason_lower and "strategy" in reason_lower:
            return "Strategy Risk Budget"
        return "Other"

    def top_reject_reasons(self, top_n: int = 10) -> pd.DataFrame:
        """Get Top-N rejection reasons from orders."""
        columns = ["reject_reason", "count", "ratio"]
        if top_n <= 0:
            return pd.DataFrame(columns=columns)

        orders = self.orders_df.copy()
        if orders.empty:
            return pd.DataFrame(columns=columns)

        if "reject_reason" not in orders.columns:
            orders["reject_reason"] = ""
        if "status" not in orders.columns:
            orders["status"] = ""
        reject_reason = orders["reject_reason"].fillna("").astype(str)
        status = orders["status"].fillna("").astype(str)
        rejected_mask = (reject_reason.str.len() > 0) | status.str.contains(
            "Rejected", case=False, regex=False
        )
        rejected = orders.loc[rejected_mask].copy()
        if rejected.empty:
            return pd.DataFrame(columns=columns)

        cleaned_reason = (
            rejected["reject_reason"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", "(empty reject reason)")
        )
        all_counts = cleaned_reason.value_counts()
        counts = all_counts.head(top_n)
        total = int(all_counts.sum())
        if total <= 0:
            return pd.DataFrame(columns=columns)

        frame = pd.DataFrame(
            {
                "reject_reason": counts.index.astype(str),
                "count": counts.values.astype(int),
            }
        )
        frame["ratio"] = frame["count"] / float(total)
        return cast(pd.DataFrame, frame.reset_index(drop=True))

    def top_reject_reason_types(self, top_n: int = 10) -> pd.DataFrame:
        """Get Top-N rejection categories with a sample raw message."""
        columns = ["reject_reason_type", "sample_reject_reason", "count", "ratio"]
        if top_n <= 0:
            return pd.DataFrame(columns=columns)

        orders = self.orders_df.copy()
        if orders.empty:
            return pd.DataFrame(columns=columns)

        if "reject_reason" not in orders.columns:
            orders["reject_reason"] = ""
        if "status" not in orders.columns:
            orders["status"] = ""
        reject_reason = orders["reject_reason"].fillna("").astype(str)
        status = orders["status"].fillna("").astype(str)
        rejected_mask = (reject_reason.str.len() > 0) | status.str.contains(
            "Rejected", case=False, regex=False
        )
        rejected = orders.loc[rejected_mask].copy()
        if rejected.empty:
            return pd.DataFrame(columns=columns)

        cleaned_reason = (
            rejected["reject_reason"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", "(empty reject reason)")
        )
        classified = cleaned_reason.map(self._classify_reject_reason)
        total = int(len(classified))
        if total <= 0:
            return pd.DataFrame(columns=columns)

        frame = pd.DataFrame(
            {
                "reject_reason_type": classified.astype(str),
                "sample_reject_reason": cleaned_reason.astype(str),
            }
        )
        grouped = (
            frame.groupby("reject_reason_type", sort=False)
            .agg(
                count=("reject_reason_type", "size"),
                sample_reject_reason=(
                    "sample_reject_reason",
                    lambda series: cast(str, series.value_counts().index[0]),
                ),
            )
            .reset_index()
        )
        grouped = grouped.sort_values(
            ["count", "reject_reason_type"], ascending=[False, True]
        ).head(top_n)
        grouped["ratio"] = grouped["count"] / float(total)
        return cast(pd.DataFrame, grouped[columns].reset_index(drop=True))

    def risk_rejections_by_strategy(self) -> pd.DataFrame:
        """Get strategy-level risk rejection breakdown table."""
        orders = self.orders_df.copy()
        columns = [
            "owner_strategy_id",
            "risk_reject_count",
            "daily_loss_reject_count",
            "drawdown_reject_count",
            "reduce_only_reject_count",
            "position_limit_reject_count",
            "order_size_limit_reject_count",
            "order_value_limit_reject_count",
            "strategy_risk_budget_reject_count",
            "portfolio_risk_budget_reject_count",
            "other_risk_reject_count",
        ]
        if orders.empty:
            return pd.DataFrame(columns=columns)

        if "owner_strategy_id" not in orders.columns:
            orders["owner_strategy_id"] = "_default"
        orders["owner_strategy_id"] = (
            orders["owner_strategy_id"].fillna("_default").astype(str)
        )
        if "reject_reason" not in orders.columns:
            orders["reject_reason"] = ""
        if "status" not in orders.columns:
            orders["status"] = ""
        reject_reason = orders["reject_reason"].fillna("").astype(str)
        status = orders["status"].fillna("").astype(str)
        rejected_mask = (reject_reason.str.len() > 0) | status.str.contains(
            "Rejected", case=False, regex=False
        )
        rejected = orders.loc[rejected_mask].copy()
        if rejected.empty:
            return pd.DataFrame(columns=columns)
        masks = self._risk_reason_masks(rejected)

        frame = pd.DataFrame(
            {
                "owner_strategy_id": rejected["owner_strategy_id"],
                "risk_reject_count": 1,
                "daily_loss_reject_count": masks["daily_loss_reject_count"],
                "drawdown_reject_count": masks["drawdown_reject_count"],
                "reduce_only_reject_count": masks["reduce_only_reject_count"],
                "position_limit_reject_count": masks["position_limit_reject_count"],
                "order_size_limit_reject_count": masks["order_size_limit_reject_count"],
                "order_value_limit_reject_count": masks[
                    "order_value_limit_reject_count"
                ],
                "strategy_risk_budget_reject_count": masks[
                    "strategy_risk_budget_reject_count"
                ],
                "portfolio_risk_budget_reject_count": masks[
                    "portfolio_risk_budget_reject_count"
                ],
                "other_risk_reject_count": masks["other_risk_reject_count"],
            }
        )
        grouped = frame.groupby("owner_strategy_id", as_index=False).sum()
        grouped = grouped.sort_values("owner_strategy_id").reset_index(drop=True)
        return cast(pd.DataFrame, grouped[columns])

    def risk_rejections_trend(self, freq: str = "D") -> pd.DataFrame:
        """Get time-series trend of risk rejections."""
        orders = self.orders_df.copy()
        columns = [
            "date",
            "risk_reject_count",
            "daily_loss_reject_count",
            "drawdown_reject_count",
            "reduce_only_reject_count",
            "position_limit_reject_count",
            "order_size_limit_reject_count",
            "order_value_limit_reject_count",
            "strategy_risk_budget_reject_count",
            "portfolio_risk_budget_reject_count",
            "other_risk_reject_count",
        ]
        if orders.empty:
            return pd.DataFrame(columns=columns)

        ts_col = "updated_at" if "updated_at" in orders.columns else "created_at"
        if ts_col not in orders.columns:
            return pd.DataFrame(columns=columns)
        if "reject_reason" not in orders.columns:
            orders["reject_reason"] = ""
        if "status" not in orders.columns:
            orders["status"] = ""
        reject_reason = orders["reject_reason"].fillna("").astype(str)
        status = orders["status"].fillna("").astype(str)
        rejected_mask = (reject_reason.str.len() > 0) | status.str.contains(
            "Rejected", case=False, regex=False
        )
        rejected = orders.loc[rejected_mask].copy()
        if rejected.empty:
            return pd.DataFrame(columns=columns)

        ts = pd.to_datetime(rejected[ts_col], errors="coerce")
        rejected = rejected.loc[ts.notna()].copy()
        if rejected.empty:
            return pd.DataFrame(columns=columns)
        rejected["date"] = cast(pd.Series, ts.loc[ts.notna()]).dt.floor(freq)

        masks = self._risk_reason_masks(rejected)
        frame = pd.DataFrame(
            {
                "date": rejected["date"],
                "risk_reject_count": 1,
                "daily_loss_reject_count": masks["daily_loss_reject_count"],
                "drawdown_reject_count": masks["drawdown_reject_count"],
                "reduce_only_reject_count": masks["reduce_only_reject_count"],
                "position_limit_reject_count": masks["position_limit_reject_count"],
                "order_size_limit_reject_count": masks["order_size_limit_reject_count"],
                "order_value_limit_reject_count": masks[
                    "order_value_limit_reject_count"
                ],
                "strategy_risk_budget_reject_count": masks[
                    "strategy_risk_budget_reject_count"
                ],
                "portfolio_risk_budget_reject_count": masks[
                    "portfolio_risk_budget_reject_count"
                ],
                "other_risk_reject_count": masks["other_risk_reject_count"],
            }
        )
        grouped = frame.groupby("date", as_index=False).sum()
        grouped = grouped.sort_values("date").reset_index(drop=True)
        return cast(pd.DataFrame, grouped[columns])

    def risk_rejections_trend_by_strategy(self, freq: str = "D") -> pd.DataFrame:
        """Get time-series trend of risk rejections by strategy."""
        orders = self.orders_df.copy()
        columns = [
            "date",
            "owner_strategy_id",
            "risk_reject_count",
            "daily_loss_reject_count",
            "drawdown_reject_count",
            "reduce_only_reject_count",
            "position_limit_reject_count",
            "order_size_limit_reject_count",
            "order_value_limit_reject_count",
            "strategy_risk_budget_reject_count",
            "portfolio_risk_budget_reject_count",
            "other_risk_reject_count",
        ]
        if orders.empty:
            return pd.DataFrame(columns=columns)

        ts_col = "updated_at" if "updated_at" in orders.columns else "created_at"
        if ts_col not in orders.columns:
            return pd.DataFrame(columns=columns)
        if "owner_strategy_id" not in orders.columns:
            orders["owner_strategy_id"] = "_default"
        orders["owner_strategy_id"] = (
            orders["owner_strategy_id"].fillna("_default").astype(str)
        )
        if "reject_reason" not in orders.columns:
            orders["reject_reason"] = ""
        if "status" not in orders.columns:
            orders["status"] = ""
        reject_reason = orders["reject_reason"].fillna("").astype(str)
        status = orders["status"].fillna("").astype(str)
        rejected_mask = (reject_reason.str.len() > 0) | status.str.contains(
            "Rejected", case=False, regex=False
        )
        rejected = orders.loc[rejected_mask].copy()
        if rejected.empty:
            return pd.DataFrame(columns=columns)

        ts = pd.to_datetime(rejected[ts_col], errors="coerce")
        rejected = rejected.loc[ts.notna()].copy()
        if rejected.empty:
            return pd.DataFrame(columns=columns)
        rejected["date"] = cast(pd.Series, ts.loc[ts.notna()]).dt.floor(freq)

        masks = self._risk_reason_masks(rejected)
        frame = pd.DataFrame(
            {
                "date": rejected["date"],
                "owner_strategy_id": rejected["owner_strategy_id"],
                "risk_reject_count": 1,
                "daily_loss_reject_count": masks["daily_loss_reject_count"],
                "drawdown_reject_count": masks["drawdown_reject_count"],
                "reduce_only_reject_count": masks["reduce_only_reject_count"],
                "position_limit_reject_count": masks["position_limit_reject_count"],
                "order_size_limit_reject_count": masks["order_size_limit_reject_count"],
                "order_value_limit_reject_count": masks[
                    "order_value_limit_reject_count"
                ],
                "strategy_risk_budget_reject_count": masks[
                    "strategy_risk_budget_reject_count"
                ],
                "portfolio_risk_budget_reject_count": masks[
                    "portfolio_risk_budget_reject_count"
                ],
                "other_risk_reject_count": masks["other_risk_reject_count"],
            }
        )
        grouped = frame.groupby(["date", "owner_strategy_id"], as_index=False).sum()
        grouped = grouped.sort_values(["date", "owner_strategy_id"]).reset_index(
            drop=True
        )
        return cast(pd.DataFrame, grouped[columns])

    @cached_property
    def indicator_definitions(self) -> pd.DataFrame:
        """Get normalized indicator definition metadata."""
        rows = list(self.indicator_outputs.get("definitions", []))
        columns = [
            "indicator_key",
            "display_name",
            "pane",
            "render_type",
            "unit",
            "precision",
            "color",
            "reference_lines",
            "scale_group",
        ]
        if not rows:
            return pd.DataFrame(columns=columns)
        frame = pd.DataFrame(rows)
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        frame = frame[columns].sort_values("indicator_key").reset_index(drop=True)
        return cast(pd.DataFrame, frame)

    @cached_property
    def indicator_instances(self) -> pd.DataFrame:
        """Get normalized indicator instance metadata."""
        rows = list(self.indicator_outputs.get("instances", []))
        columns = [
            "instance_id",
            "owner_strategy_id",
            "symbol",
            "indicator_key",
            "meta_json",
        ]
        if not rows:
            return pd.DataFrame(columns=columns)
        frame = pd.DataFrame(rows)
        for column in columns:
            if column not in frame.columns:
                frame[column] = ""
        frame = frame[columns].sort_values(
            ["owner_strategy_id", "symbol", "indicator_key", "instance_id"]
        )
        return cast(pd.DataFrame, frame.reset_index(drop=True))

    @cached_property
    def _indicator_points_df(self) -> pd.DataFrame:
        rows = list(self.indicator_outputs.get("points", []))
        columns = [
            "instance_id",
            "owner_strategy_id",
            "symbol",
            "indicator_key",
            "timestamp",
            "timestamp_ms",
            "value",
            "warmup",
        ]
        if not rows:
            return pd.DataFrame(columns=columns + ["datetime"])
        frame = pd.DataFrame(rows)
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce").astype(
            "Int64"
        )
        # ``timestamp_ms`` mirrors ``timestamp`` for frontend charting libraries
        # that expect epoch milliseconds; derive it when a legacy payload omits it.
        frame["timestamp_ms"] = pd.to_numeric(
            frame["timestamp_ms"], errors="coerce"
        ).astype("Int64")
        frame["timestamp_ms"] = frame["timestamp_ms"].fillna(
            frame["timestamp"] // 1_000_000
        )
        frame["datetime"] = pd.to_datetime(
            frame["timestamp"],
            unit="ns",
            utc=True,
            errors="coerce",
        ).dt.tz_convert(self._timezone)
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame["warmup"] = frame["warmup"].fillna(False).astype(bool)
        frame = frame.sort_values(
            ["owner_strategy_id", "symbol", "indicator_key", "timestamp"]
        ).reset_index(drop=True)
        return cast(pd.DataFrame, frame[columns + ["datetime"]])

    def indicator_df(
        self,
        name: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get recorded indicator points as a DataFrame."""
        frame = cast(pd.DataFrame, self._indicator_points_df.copy())
        if frame.empty:
            return cast(pd.DataFrame, frame)
        if name is not None:
            frame = frame.loc[frame["indicator_key"] == str(name)]
        if symbol is not None:
            frame = frame.loc[frame["symbol"] == str(symbol)]
        return cast(pd.DataFrame, frame.reset_index(drop=True))

    def export_indicators(self, path: str, format: str = "json") -> None:
        """Export indicator outputs to json or a parquet directory bundle."""
        output_format = str(format).strip().lower()
        output_path = Path(path)
        if output_format == "json":
            payload = {
                "run_id": self.stream_run_id,
                "definitions": self._json_ready_records(self.indicator_definitions),
                "instances": self._json_ready_records(self.indicator_instances),
                "points": self._json_ready_records(self._indicator_points_df),
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # ``ensure_ascii=False`` keeps CJK metadata readable and matches the
            # indicator collection layer, so the export and stream paths agree.
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return
        if output_format == "parquet":
            output_path.mkdir(parents=True, exist_ok=True)
            definitions_frame = self.indicator_definitions.copy()
            if "reference_lines" in definitions_frame.columns:
                definitions_frame["reference_lines"] = definitions_frame[
                    "reference_lines"
                ].map(
                    lambda v: (
                        json.dumps(v, ensure_ascii=False)
                        if isinstance(v, (list, dict))
                        else json.dumps([])
                    )
                )
            definitions_frame.to_parquet(output_path / "definitions.parquet")
            self.indicator_instances.to_parquet(output_path / "instances.parquet")
            self._indicator_points_df.to_parquet(output_path / "points.parquet")
            return
        raise ValueError("format must be 'json' or 'parquet'")

    def benchmark_analysis(
        self,
        benchmark: Optional[Union[str, pd.Series]] = None,
        curve_freq: str = "raw",
    ) -> Dict[str, Any]:
        """Build a structured benchmark analysis payload for APIs and reports."""
        from ..analysis.benchmark import benchmark_analysis_from_result

        return cast(
            Dict[str, Any],
            benchmark_analysis_from_result(
                result=self,
                benchmark=benchmark,
                curve_freq=curve_freq,
            ),
        )

    def export_benchmark_analysis(
        self,
        path: str,
        benchmark: Optional[Union[str, pd.Series]] = None,
        format: str = "json",
        curve_freq: str = "raw",
    ) -> None:
        """Export structured benchmark analysis to json or a parquet bundle."""
        output_format = str(format).strip().lower()
        output_path = Path(path)
        payload = self.benchmark_analysis(
            benchmark=benchmark,
            curve_freq=curve_freq,
        )
        if output_format == "json":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # ``ensure_ascii=False`` keeps CJK benchmark labels readable, matching
            # export_indicators so all structured exports share one encoding.
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return
        if output_format == "parquet":
            output_path.mkdir(parents=True, exist_ok=True)
            series_frame = pd.DataFrame(payload.get("series", []))
            series_frame.to_parquet(output_path / "series.parquet")
            metadata = dict(payload)
            metadata.pop("series", None)
            (output_path / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return
        raise ValueError("format must be 'json' or 'parquet'")

    @staticmethod
    def _json_ready_records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for row in frame.to_dict(orient="records"):
            normalized: Dict[str, Any] = {}
            for key, value in row.items():
                key_str = str(key)
                if isinstance(value, (list, dict)):
                    normalized[key_str] = value
                elif pd.isna(value):
                    normalized[key_str] = None
                elif isinstance(value, pd.Timestamp):
                    normalized[key_str] = value.isoformat()
                else:
                    normalized[key_str] = value
            records.append(normalized)
        return records

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the raw result."""
        return getattr(self._raw, name)

    def __repr__(self) -> str:
        """Return the string representation of the result (Vertical Metrics)."""
        metrics = self.metrics_df
        metrics.columns = ["Value"]
        return f"BacktestResult:\n{metrics.to_string()}"

    def __dir__(self) -> List[str]:
        """Return the list of attributes including raw result attributes."""
        return list(set(dir(self._raw) + list(self.__dict__.keys()) + ["positions"]))

    def plot(
        self,
        symbol: Optional[str] = None,
        show: bool = True,
        title: str = "Backtest Result",
    ) -> Any:
        """
        Plot the backtest results using Plotly dashboard.

        :param symbol: Reserved for backward compatibility.
        :param show: Whether to display the plot immediately.
        :param title: Title of the plot.
        :return: Plotly Figure object.
        """
        _ = symbol
        try:
            from ..plot import plot_dashboard
        except ImportError:
            print(
                "Plotly is not installed. Please install it using `pip install plotly` "
                "or `pip install akquant[plot]`."
            )
            return None

        return plot_dashboard(result=self, show=show, title=title)

    def plot_indicators(
        self,
        name: Optional[str] = None,
        symbol: Optional[str] = None,
        include_warmup: bool = True,
        show: bool = True,
        title: str = "Indicator History",
        theme: str = "light",
        filename: Optional[str] = None,
    ) -> Any:
        """
        Plot recorded indicator history in a lightweight multi-pane figure.

        :param name: Optional indicator key filter.
        :param symbol: Optional symbol filter.
        :param include_warmup: Whether to keep warmup points.
        :param show: Whether to display the plot immediately.
        :param title: Figure title.
        :param theme: Plot theme key.
        :param filename: Optional HTML output path.
        :return: Plotly Figure object.
        """
        try:
            from ..plot import plot_indicators
        except ImportError:
            print(
                "Plotly is not installed. Please install it using `pip install plotly` "
                "or `pip install akquant[plot]`."
            )
            return None

        return plot_indicators(
            result=self,
            name=name,
            symbol=symbol,
            include_warmup=include_warmup,
            show=show,
            title=title,
            theme=theme,
            filename=filename,
        )

    def to_quantstats(self) -> pd.Series:
        """
        Convert backtest results to QuantStats-compatible returns series.

        :return: pd.Series with DatetimeIndex and daily returns.
        """
        # Get daily returns (already calculated from equity curve)
        returns = self.daily_returns.copy()

        if returns.empty:
            return returns

        # QuantStats prefers timezone-naive index (or UTC) usually, but it handles
        # datetime index well.
        # However, to be safe and compatible with most data sources QS uses (Yahoo),
        # we might want to ensure it's timezone-naive (localized to None)
        # or keep it as is if QS handles it.
        # AKQuant returns are timezone-aware (user configured).
        # Let's strip timezone to avoid warnings/errors in some QS versions if they
        # compare with naive benchmarks.
        # Use cast to help mypy understand it's a DatetimeIndex
        idx = cast(pd.DatetimeIndex, returns.index)
        if idx.tz is not None:
            returns.index = idx.tz_localize(None)

        return returns

    def report_quantstats(
        self,
        benchmark: Optional[Union[str, pd.Series]] = None,
        title: str = "Strategy Report",
        filename: str = "quantstats-report.html",
        **kwargs: Any,
    ) -> None:
        """
        Generate a QuantStats HTML report.

        :param benchmark: Benchmark ticker (e.g. "SPY") or pd.Series.
        :param title: Report title.
        :param filename: Output filename.
        :param kwargs: Additional arguments passed to qs.reports.html.
        """
        try:
            import quantstats as qs
        except ImportError:
            print(
                "QuantStats is not installed. Please install it using "
                "`pip install quantstats` or `pip install akquant[quantstats]`."
            )
            return

        # Extend pandas functionality (optional, but good practice for QS)
        qs.extend_pandas()

        returns = self.to_quantstats()

        if returns.empty:
            print("No returns data available to generate report.")
            return

        print(f"Generating QuantStats report to {filename}...")
        qs.reports.html(
            returns, benchmark=benchmark, title=title, output=filename, **kwargs
        )
        print("Done.")

    def report(
        self,
        title: str = "AKQuant 策略回测报告",
        filename: str = "akquant_report.html",
        show: bool = False,
        compact_currency: bool = True,
        market_data: Optional[Union[pd.DataFrame, dict[str, pd.DataFrame]]] = None,
        plot_symbol: Optional[str] = None,
        include_trade_kline: bool = True,
        include_indicators: bool = False,
        indicator_name: Optional[str] = None,
        indicator_symbol: Optional[str] = None,
        indicator_include_warmup: bool = True,
        benchmark: Optional[Union[str, pd.Series]] = None,
        curve_freq: str = "D",
    ) -> None:
        """
        生成 HTML 策略回测报告 (便捷方法).

        该方法是 akquant.plot.report.plot_report 的快捷入口。

        :param title: 报告标题
        :param filename: 保存的文件名
        :param show: 是否在浏览器中自动打开 (默认 False)
        :param compact_currency: 是否将金额用 K/M/B 紧凑显示 (默认 True)
        :param market_data: 可选行情数据，用于绘制 K 线买卖点图
        :param plot_symbol: 可选标的代码，指定 K 线复盘标的
        :param include_trade_kline: 是否在报告中包含 K 线复盘图
        :param include_indicators: 是否在报告中包含自定义指标预览区块
        :param indicator_name: 可选指标键过滤，仅展示指定指标
        :param indicator_symbol: 可选标的过滤，仅展示指定标的指标
        :param indicator_include_warmup: 是否在指标报告区块中保留预热点
        :param benchmark: 基准收益序列 (pd.Series) 或基准标识字符串
        :param curve_freq: 曲线频率，默认 "D" 为日频末值，"raw" 为原始频率
        """
        # 延迟导入，避免循环引用和非必要的 Plotly 依赖
        try:
            from ..plot.report import plot_report
        except ImportError:
            print("Plot module not found. Please install akquant[plot] or plotly.")
            return

        return plot_report(
            result=self,
            title=title,
            filename=filename,
            show=show,
            compact_currency=compact_currency,
            market_data=market_data,
            plot_symbol=plot_symbol,
            include_trade_kline=include_trade_kline,
            include_indicators=include_indicators,
            indicator_name=indicator_name,
            indicator_symbol=indicator_symbol,
            indicator_include_warmup=indicator_include_warmup,
            benchmark=benchmark,
            curve_freq=curve_freq,
        )

    def report_lwc(
        self,
        title: str = "AKQuant 策略回测报告 (Lightweight Charts)",
        filename: str = "akquant_lwc_report.html",
        show: bool = False,
        market_data: Optional[Union[pd.DataFrame, dict[str, pd.DataFrame]]] = None,
        symbols: Optional[list[str]] = None,
        plot_symbol: Optional[str] = None,
    ) -> str:
        """
        生成基于 TradingView Lightweight Charts 的 HTML 回测报告 (plotly 替代).

        报告为单文件 HTML（内嵌图表库，无 CDN 依赖），包含绩效指标、
        净值/回撤曲线与交易复盘（K线买卖点）。被复盘的股票可在网页中
        热切换：在输入框中键入代码即可切换复盘标的。

        :param title: 报告标题
        :param filename: 保存的文件名
        :param show: 是否在浏览器中自动打开 (默认 False)
        :param market_data: 可选行情数据；传入 ``{symbol: frame}`` 字典时
            所有标的都会被内嵌进页面，可在页面上互相热切换
        :param symbols: 可选标的子集，仅内嵌这些标的
        :param plot_symbol: 可选标的代码，指定初始复盘标的
        :return: 生成的 HTML 文件绝对路径
        """
        from ..lwc.report import plot_report as _plot_report_lwc

        return _plot_report_lwc(
            result=self,
            title=title,
            filename=filename,
            show=show,
            market_data=market_data,
            symbols=symbols,
            plot_symbol=plot_symbol,
        )

    def serve_review(
        self,
        market_data: Optional[Union[pd.DataFrame, dict[str, pd.DataFrame]]] = None,
        data_provider: Optional[Any] = None,
        host: str = "127.0.0.1",
        port: int = 8765,
        title: str = "AKQuant 交易复盘 (Lightweight Charts)",
        open_browser: bool = True,
    ) -> None:
        """
        启动交互式交易复盘 Web 服务 (阻塞, 基于 Lightweight Charts).

        页面按需加载各标的 K 线与买卖点：在网页输入框中键入任意可解析的
        股票代码即可热切换复盘标的，无需重新生成文件。

        :param market_data: 可选预加载行情数据 ``{symbol: frame}``
        :param data_provider: 可选回调 ``(code) -> DataFrame``，用于解析
            ``market_data`` 中不存在的标的代码
        :param host: 监听地址
        :param port: 监听端口，0 表示自动选择空闲端口
        :param title: 页面标题
        :param open_browser: 启动后是否自动打开浏览器
        """
        from ..lwc.server import serve_review as _serve_review

        return _serve_review(
            result=self,
            market_data=market_data,
            data_provider=data_provider,
            host=host,
            port=port,
            title=title,
            open_browser=open_browser,
        )
