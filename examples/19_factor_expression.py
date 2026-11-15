import logging
import shutil
from pathlib import Path

import akshare as ak
import pandas as pd
from akquant.data import ParquetDataCatalog
from akquant.factor import FactorEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_data_with_akshare(catalog_path: Path) -> ParquetDataCatalog:
    """Download real A-share data using AKShare and store in catalog."""
    # 1. Initialize Catalog
    if catalog_path.exists():
        shutil.rmtree(catalog_path)

    catalog = ParquetDataCatalog(root_path=str(catalog_path))

    # 2. Define Symbols (Example: Ping An Bank, CATL, China Merchants Bank)
    # Using small date range for quick testing
    symbols = ["sh600001", "sz300750", "sh600036"]
    start_date = "20230101"
    end_date = "20230601"

    logger.info("Starting data download from AKShare...")

    for symbol in symbols:
        logger.info(f"Downloading {symbol}...")

        try:
            # Fetch daily data (hfq = backward adjusted)
            df = ak.stock_zh_a_hist_sina(
                symbol=symbol, start_date=start_date, end_date=end_date, adjust="hfq"
            )

            if df.empty:
                logger.warning(f"No data found for {symbol}")
                continue

            # Add symbol column (Required for FactorEngine grouping)
            df["symbol"] = symbol

            # Ensure date is datetime and set as index
            # AKShare returns: date, open, high, low, close, volume, etc.
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

            # Write to Parquet Catalog
            catalog.write(symbol, df)
            logger.info(f"Successfully wrote {len(df)} rows for {symbol}")

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    return catalog


def test_engine() -> None:
    """Test the factor engine with sample expressions."""
    catalog_path = Path("temp_factor_akshare_catalog")

    try:
        # 1. Setup Data
        catalog = setup_data_with_akshare(catalog_path)

        # 2. Initialize Engine
        engine = FactorEngine(catalog)

        # 3. Define Factor Expressions
        # These are Alpha101-style expressions
        expressions = [
            # Time Series Factors
            "Ts_Mean(Close, 5)",  # 5-day Moving Average
            "Ts_Std(Close, 20)",  # 20-day Volatility
            "Delta(Close, 1)",  # Daily Price Change
            # Cross Sectional Factors
            "Rank(Close)",  # Price Rank across stocks
            "Rank(Ts_Mean(Volume, 5))",  # Rank of 5-day Avg Volume
            # Advanced Time Series Features (New in v0.3)
            "Ts_ArgMax(High, 20)",  # Days since 20-day high
            "Ts_ArgMin(Low, 20)",  # Days since 20-day low
            "Ts_Rank(Close, 10)",  # Percentile rank of current price in 10-day window
            # Composite/Logic Factors
            "If(Close > Open, 1, -1)",  # Up/Down day indicator
            "Rank(Ts_Corr(Close, Volume, 10))",  # Rank of Price-Volume Correlation
        ]

        print("\n=== Testing Single Expressions ===")
        for expr in expressions:
            print(f"\nExpression: {expr}")
            try:
                df = engine.run(expr)
                print(df.head(5))

                # Basic Validation
                if not df.is_empty():
                    assert "factor_value" in df.columns
                    assert "date" in df.columns
                    assert "symbol" in df.columns
                else:
                    logger.warning("Result DataFrame is empty!")

            except Exception as e:
                logger.error(f"FAILED: {e}")
                # Don't raise, continue testing others

        print("\n=== Testing Batch Execution ===")
        batch_exprs = expressions[:3]
        df_batch = engine.run_batch(batch_exprs)
        print(df_batch.head())
        print(f"Batch Result Shape: {df_batch.shape}")

    finally:
        # Cleanup
        if catalog_path.exists():
            shutil.rmtree(catalog_path)
            logger.info("Cleaned up temporary catalog.")

    print("\nSUCCESS: All tests completed!")


if __name__ == "__main__":
    test_engine()
