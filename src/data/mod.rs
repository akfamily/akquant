pub mod aggregator;
pub mod batch;
pub mod client;
pub mod columns;
pub mod compute;
pub mod feed;
pub mod parquet_stream;

pub use aggregator::BarAggregator;
pub use batch::from_arrays;
pub use client::FeedAction;
pub use compute::{
    vec_cumsum, vec_ema, vec_log_returns, vec_returns, vec_rolling_max, vec_rolling_min,
    vec_rolling_std, vec_rolling_sum, vec_sma, vec_wma, vec_zscore,
};
pub use feed::DataFeed;
