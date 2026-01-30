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
