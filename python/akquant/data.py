import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .akquant import Bar
from .utils import load_bar_from_df

logger = logging.getLogger(__name__)


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

        df.to_parquet(file_path, compression="snappy")
        return file_path

    def read(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Read DataFrame from Parquet catalog.

        :param symbol: Instrument symbol.
        :param start_date: Filter start date (YYYYMMDD or YYYY-MM-DD).
        :param end_date: Filter end date.
        :param columns: Specific columns to read.
        :return: DataFrame.
        """
        symbol_path = self.root / symbol
        file_path = symbol_path / "data.parquet"

        if not file_path.exists():
            return pd.DataFrame()

        # Read with projection (columns) and push-down filters
        filters = []
        if start_date:
            filters.append(("index", ">=", pd.to_datetime(start_date)))
        if end_date:
            filters.append(("index", "<=", pd.to_datetime(end_date)))

        # If filters is empty, pass None to read_parquet
        kwargs: Dict[str, Any] = {"columns": columns}
        if filters:
            kwargs["filters"] = filters

        try:
            df = pd.read_parquet(file_path, **kwargs)
        except Exception:
            # Fallback for engines that don't support filters or if index name mismatch
            df = pd.read_parquet(file_path, columns=columns)
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]

        return df

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
