use pyo3::prelude::*;

#[path = "volatility_candle.rs"]
mod volatility_candle;
#[path = "volatility_price.rs"]
mod volatility_price;
#[path = "volatility_stats.rs"]
mod volatility_stats;

pub use volatility_candle::HeikinAshi;
pub use volatility_price::{AVGPRICE, MEDPRICE, MIDPRICE, TYPPRICE, WCLPRICE};
pub use volatility_stats::{ATR, BollingerBands, NATR, STDDEV, TRANGE, VAR};

pub fn register_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    volatility_stats::register_classes(m)?;
    volatility_price::register_classes(m)?;
    volatility_candle::register_classes(m)?;
    Ok(())
}
