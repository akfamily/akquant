use pyo3::prelude::*;

#[path = "moving_average_compound.rs"]
mod moving_average_compound;
#[path = "moving_average_core.rs"]
mod moving_average_core;
#[path = "moving_average_hilbert.rs"]
mod moving_average_hilbert;
#[path = "moving_average_transforms.rs"]
mod moving_average_transforms;
#[path = "moving_average_windowed.rs"]
mod moving_average_windowed;

pub use moving_average_compound::{APO, DEMA, KAMA, MACD, PPO, T3, TEMA, TRIX};
pub use moving_average_core::{EMA, SMA, TRIMA, WMA};
pub use moving_average_hilbert::{HT_TRENDLINE, MAMA};
pub use moving_average_transforms::{
    ABS, ACOS, ADD, ASIN, ATAN, CEIL, CLAMP01, CLIP, COS, COSH, CUBE, DEG2RAD, DIV, EXP, EXPM1,
    FLOOR, INV_SQRT, LOG1P, LOG10, MAX2, MIN2, MOD, MULT, POW, RECIP, ROUND, SIGN, SIN, SINH, SQ,
    SQRT, SUB, TAN, TANH,
};
pub use moving_average_windowed::{
    AVGDEV, LN, MAX, MAXINDEX, MIDPOINT, MIN, MININDEX, MINMAX, MINMAXINDEX, RANGE, SUM,
};

pub fn register_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    moving_average_core::register_classes(m)?;
    moving_average_compound::register_classes(m)?;
    moving_average_hilbert::register_classes(m)?;
    moving_average_windowed::register_classes(m)?;
    moving_average_transforms::register_classes(m)?;
    Ok(())
}
