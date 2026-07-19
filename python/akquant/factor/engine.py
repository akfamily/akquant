from typing import List, Optional

import polars as pl

from ..data import ParquetDataCatalog
from ..log import get_logger
from .parser import ExpressionParser

logger = get_logger("factor")


class FactorEngine:
    """Engine for calculating factors using Polars Lazy API."""

    def __init__(self, catalog: ParquetDataCatalog):
        """Initialize the FactorEngine with a data catalog."""
        self.catalog = catalog
        self.parser = ExpressionParser()

    def load_data(self) -> pl.LazyFrame:
        """
        Load all data from catalog as a LazyFrame.

        Assumes data is partitioned by symbol or simply stored in separate files.
        """
        # We look for all parquet files under the root
        # Pattern: root/**/data.parquet
        # This covers:
        # - root/AAPL/data.parquet (ParquetDataCatalog default)
        # - root/symbol=AAPL/data.parquet (Hive style)
        pattern = str(self.catalog.root / "**" / "*.parquet")

        try:
            lf = pl.scan_parquet(
                pattern,
                cast_options=pl.ScanCastOptions(integer_cast="allow-float"),
            )
            return lf
        except Exception as e:
            logger.warning(f"Failed to scan parquet files: {e}")
            return pl.DataFrame().lazy()

    def _ensure_date_column(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Ensure 'date' column exists, renaming from common variants if needed."""
        # Check schema to find date column
        # Note: collect_schema() is preferred in newer Polars, .schema in older
        try:
            schema = lf.collect_schema()
        except AttributeError:
            schema = lf.schema  # type: ignore

        cols = schema.names() if hasattr(schema, "names") else list(schema.keys())
        logger.info(f"Detected columns: {cols}")

        if "date" in cols:
            return lf
        elif "index" in cols:
            return lf.rename({"index": "date"})
        elif "datetime" in cols:
            return lf.rename({"datetime": "date"})
        elif "__index_level_0__" in cols:
            return lf.rename({"__index_level_0__": "date"})
        else:
            # Fallback: assume user knows what they are doing or it will fail later
            logger.warning(
                f"Could not identify 'date' column from {cols}. "
                "Expecting 'date', 'index', 'datetime', or '__index_level_0__'."
            )
            return lf

    def run(
        self,
        expr_str: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Execute a factor expression.

        :param expr_str: The factor expression string (e.g., "Rank(Ts_Mean(Close, 5))")
        :param start_date: Filter start date (inclusive).
        :param end_date: Filter end date (inclusive).
        :return: DataFrame with [date, symbol, factor] columns.
        """
        lf = self.load_data()
        return self.run_on_data(lf, expr_str, start_date, end_date)

    def run_on_data(
        self,
        data: pl.LazyFrame | pl.DataFrame,
        expr_str: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Execute a factor expression on provided data.

        :param data: Polars DataFrame or LazyFrame containing input data.
        :param expr_str: The factor expression string.
        :param start_date: Filter start date (inclusive).
        :param end_date: Filter end date (inclusive).
        :return: DataFrame with [date, symbol, factor] columns.
        """
        if isinstance(data, pl.DataFrame):
            lf = data.lazy()
        else:
            lf = data

        lf = self._ensure_date_column(lf)
        lf = lf.sort(["symbol", "date"])

        if start_date:
            lf = lf.filter(
                pl.col("date") >= pl.lit(start_date).str.to_datetime(strict=False)
            )
        if end_date:
            lf = lf.filter(
                pl.col("date") <= pl.lit(end_date).str.to_datetime(strict=False)
            )

        # Plan execution
        logger.info(f"Planning expression: {expr_str}")
        steps = self.parser.plan(expr_str)

        # Execute steps sequentially
        current_lf = lf
        for var_name, sub_expr_str in steps:
            logger.info(f"Executing step: {var_name} = {sub_expr_str}")
            sub_expr = self.parser.parse(sub_expr_str)

            # If it's the final result, select it
            if var_name == "result":
                final_expr = sub_expr.alias("factor_value")
                return current_lf.select(
                    [pl.col("date"), pl.col("symbol"), final_expr]
                ).collect()
            else:
                # Intermediate step: compute and materialize
                # We use with_columns to add the intermediate result
                # IMPORTANT: We MUST collect() to break the Lazy graph and force
                # materialization to avoid the nested window function issue
                # (Polars #25691). Polars treats nested `over()` as hierarchical,
                # but TS/CS ops are orthogonal. Explicit materialization ensures
                # the inner result is fully computed before the outer window
                # function runs.

                temp_df = current_lf.with_columns(sub_expr.alias(var_name)).collect()
                current_lf = temp_df.lazy()

        # Should not reach here
        return pl.DataFrame()

    def run_batch(self, exprs: List[str]) -> pl.DataFrame:
        """Run multiple expressions at once."""
        lf = self.load_data()
        lf = self._ensure_date_column(lf)

        # We need a base dataframe with all dates/symbols to join against
        # This can be expensive if data is huge.
        # Alternatively, we can use the first result as base, but outer join might be
        # needed if filters differ (here filters are global).

        # Optimization: Scan unique date/symbol only
        base_df = (
            lf.select(["date", "symbol"]).unique().collect().sort(["date", "symbol"])
        )

        for i, expr_str in enumerate(exprs):
            # Run each expression.
            # Note: run_on_data sorts internally, so order is consistent.
            res = self.run_on_data(lf, expr_str)
            res = res.rename({"factor_value": f"factor_{i}"})

            # Join back to base
            # Use left join if base has all keys
            base_df = base_df.join(res, on=["date", "symbol"], how="left")

        return base_df
