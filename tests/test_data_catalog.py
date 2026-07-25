from pathlib import Path

import akquant
import pandas as pd
from akquant.data import ParquetDataCatalog


class CaptureCatalogBarsStrategy(akquant.Strategy):
    """Capture bar timestamps loaded from catalog-backed backtests."""

    def __init__(self) -> None:
        """Initialize the capture buffer."""
        super().__init__()
        self.seen_timestamps: list[str] = []

    def on_bar(self, bar: akquant.Bar) -> None:
        """Record each visited bar timestamp."""
        self.seen_timestamps.append(str(bar.timestamp_iso))


def test_parquet_catalog_read_naive_boundaries_follow_timezone(
    tmp_path: Path,
) -> None:
    """Catalog.read should interpret naive boundaries in the requested timezone."""
    catalog = ParquetDataCatalog(root_path=str(tmp_path))
    symbol = "CATALOG_BOUNDARY"
    frame = pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0],
            "high": [10.5, 11.5, 12.5],
            "low": [9.5, 10.5, 11.5],
            "close": [10.2, 11.2, 12.2],
            "volume": [1000.0, 1000.0, 1000.0],
        },
        index=pd.to_datetime(
            ["2024-01-02 20:00:00", "2024-01-02 21:00:00", "2024-01-02 22:00:00"]
        ),
    )
    catalog.write(symbol, frame)

    result = catalog.read(
        symbol,
        start_time="2024-01-02 13:00:00",
        end_time="2024-01-02 14:00:00",
        timezone="UTC",
    )

    assert list(result.index) == [
        pd.Timestamp("2024-01-02 21:00:00"),
        pd.Timestamp("2024-01-02 22:00:00"),
    ]


def test_run_backtest_catalog_path_naive_boundaries_follow_timezone(
    tmp_path: Path,
) -> None:
    """Catalog-backed backtests should forward timezone-aware runtime boundaries."""
    catalog = ParquetDataCatalog(root_path=str(tmp_path))
    symbol = "CATALOG_BACKTEST_BOUNDARY"
    frame = pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0],
            "high": [10.5, 11.5, 12.5],
            "low": [9.5, 10.5, 11.5],
            "close": [10.2, 11.2, 12.2],
            "volume": [1000.0, 1000.0, 1000.0],
        },
        index=pd.to_datetime(
            ["2024-01-02 20:00:00", "2024-01-02 21:00:00", "2024-01-02 22:00:00"]
        ),
    )
    catalog.write(symbol, frame)

    result = akquant.run_backtest(
        strategy=CaptureCatalogBarsStrategy,
        symbols=[symbol],
        catalog_path=str(tmp_path),
        start_time="2024-01-02 13:00:00",
        end_time="2024-01-02 14:00:00",
        timezone="UTC",
        fill_policy=akquant.CurrentClose(),
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        show_progress=False,
    )

    strategy = result.strategy
    assert strategy is not None
    assert strategy.seen_timestamps == [
        "2024-01-02T13:00:00Z",
        "2024-01-02T14:00:00Z",
    ]
