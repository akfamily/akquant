import hashlib
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

try:
    import akshare as ak  # type: ignore
except ImportError:
    ak = None

from .akquant import Bar
from .utils import load_akshare_bar

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

        # Read with projection (columns)
        df = pd.read_parquet(file_path, columns=columns)

        # Filter by date
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        return df

    def list_symbols(self) -> List[str]:
        """List all symbols in the catalog."""
        return [p.name for p in self.root.iterdir() if p.is_dir()]


class DataLoader:
    """Data Loader with caching capabilities, inspired by PyBroker."""

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

    def load_akshare(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
        period: str = "daily",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Load A-share history data from AKShare with caching.

        Args:
            symbol (str): Stock symbol (e.g., "600000").
            start_date (str): Start date (YYYYMMDD).
            end_date (str): End date (YYYYMMDD).
            adjust (str): Adjustment factor ("qfq", "hfq", ""). Default "qfq".
            period (str): Period ("daily", "weekly", "monthly"). Default "daily".
            use_cache (bool): Whether to use cache. Default True.

        Returns:
            pd.DataFrame: Historical data.
        """
        if ak is None:
            raise ImportError(
                "akshare is not installed. Please run `pip install akshare`."
            )

        cache_key = (
            f"akshare_stock_zh_a_hist_{symbol}_{start_date}_"
            f"{end_date}_{adjust}_{period}"
        )
        cache_path = self._get_cache_path(cache_key)

        if use_cache and cache_path.exists():
            logger.info(f"Loading cached data for {symbol} from {cache_path}")
            try:
                df = pd.read_pickle(cache_path)
                return df  # type: ignore
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Reloading from source.")

        logger.info(f"Fetching data for {symbol} from AKShare...")
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )

            # Basic validation
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return df  # type: ignore

            # Cache the result
            if use_cache:
                df.to_pickle(cache_path)
                logger.info(f"Data cached to {cache_path}")

            return df  # type: ignore
        except Exception as e:
            logger.error(f"Error fetching data from AKShare: {e}")
            raise

    def df_to_bars(self, df: pd.DataFrame, symbol: Optional[str] = None) -> List[Bar]:
        """
        Convert DataFrame to list of Bar objects.

        Wrapper around utils.load_akshare_bar.
        """
        return load_akshare_bar(df, symbol)
