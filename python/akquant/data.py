import hashlib
from pathlib import Path
from typing import List, Optional, Union, cast

import pandas as pd

from .akquant import Bar
from .feed_adapter import DEFAULT_INPUT_TIMEZONE
from .log import get_logger
from .utils import load_bar_from_df

logger = get_logger("data")


def _parse_catalog_boundary_timestamp(
    value: Union[str, pd.Timestamp],
    timezone: Optional[str],
) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return cast(
            pd.Timestamp, timestamp.tz_localize(timezone or DEFAULT_INPUT_TIMEZONE)
        )
    return timestamp


def _filter_catalog_frame_by_time_range(
    df: pd.DataFrame,
    start_time: Optional[Union[str, pd.Timestamp]],
    end_time: Optional[Union[str, pd.Timestamp]],
    timezone: Optional[str],
) -> pd.DataFrame:
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    compare_index = df.index
    if compare_index.tz is None:
        compare_index = compare_index.tz_localize(DEFAULT_INPUT_TIMEZONE)
    mask = pd.Series(True, index=df.index)
    if start_time is not None:
        start_ts = _parse_catalog_boundary_timestamp(start_time, timezone)
        mask &= compare_index >= start_ts
    if end_time is not None:
        end_ts = _parse_catalog_boundary_timestamp(end_time, timezone)
        mask &= compare_index <= end_ts
    return cast(pd.DataFrame, df.loc[mask.to_numpy()])


class ParquetDataCatalog:
    """
    Data Catalog using Parquet files for storage.

    Optimized for performance using PyArrow/FastParquet.
    Structure: {root}/{symbol}.parquet (Simplest for now)
    """

    def __init__(self, root_path: Optional[str] = None):
        """
        Initialize the DataCatalog.

        :param root_path: Root directory for the catalog.
        """
        if root_path:
            self.root = Path(root_path)
        else:
            self.root = Path.home() / ".akquant" / "catalog"

        try:
            if not self.root.exists():
                self.root.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            self.root = Path.cwd() / ".akquant_catalog"
            self.root.mkdir(parents=True, exist_ok=True)

    def write(self, symbol: str, df: pd.DataFrame) -> Path:
        """
        Write DataFrame to Parquet catalog.

        :param symbol: Instrument symbol.
        :param df: DataFrame with DatetimeIndex.
        :return: Path to the written file.
        """
        symbol_path = self.root / symbol
        symbol_path.mkdir(exist_ok=True)
        file_path = symbol_path / "data.parquet"

        # Ensure index is standard
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to convert date column if exists
            if "date" in df.columns:
                df = df.set_index("date")
                df.index = pd.to_datetime(df.index)

        # Normalize common numeric columns so parquet schemas stay stable
        # across symbols even when some series happen to contain integer-only values.
        if "symbol" not in df.columns:
            df = df.copy()
            df["symbol"] = symbol
        else:
            df = df.copy()

        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        df.to_parquet(file_path, compression="snappy")
        return file_path

    def read(
        self,
        symbol: str,
        start_time: Optional[Union[str, pd.Timestamp]] = None,
        end_time: Optional[Union[str, pd.Timestamp]] = None,
        columns: Optional[List[str]] = None,
        timezone: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Read DataFrame from Parquet catalog.

        :param symbol: Instrument symbol.
        :param start_time: Filter start date/time.
        :param end_time: Filter end date/time.
        :param columns: Specific columns to read.
        :param timezone: Timezone used to interpret naive start/end boundaries.
        :return: DataFrame.
        """
        symbol_path = self.root / symbol
        file_path = symbol_path / "data.parquet"

        if not file_path.exists():
            return pd.DataFrame()

        try:
            df = pd.read_parquet(file_path, columns=columns)
        except Exception:
            df = pd.read_parquet(file_path, columns=columns)
        return _filter_catalog_frame_by_time_range(df, start_time, end_time, timezone)

    def list_symbols(self) -> List[str]:
        """List all symbols in the catalog."""
        return [p.name for p in self.root.iterdir() if p.is_dir()]


class DataLoader:
    """Data Loader with caching capabilities."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize DataLoader.

        Args:
            cache_dir (str, optional): Directory to store cache files.
                                     Defaults to ~/.akquant/cache.
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".akquant" / "cache"

        try:
            if not self.cache_dir.exists():
                self.cache_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.warning(
                f"Permission denied for {self.cache_dir}, "
                "falling back to local .akquant_cache"
            )
            self.cache_dir = Path.cwd() / ".akquant_cache"
            if not self.cache_dir.exists():
                self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Generate cache file path based on a unique key."""
        # Use a hash of the key to avoid filesystem issues with long/invalid filenames
        hashed_key = hashlib.md5(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{hashed_key}.pkl"

    def df_to_bars(self, df: pd.DataFrame, symbol: Optional[str] = None) -> List[Bar]:
        """Convert DataFrame to list of Bar objects."""
        return load_bar_from_df(df, symbol)
