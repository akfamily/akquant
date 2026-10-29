use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

/// 任一输入非有限值(NaN/±inf, 例如停牌缺价)时本根不出值, 也不推进任何状态。
///
/// 每个内置指标 `update` 的第一行都调用它(设计 C2), 规则与 `HeikinAshi` 相同:
/// 调用方拿到 None, `value` 保持上一个有效值, 下一根有效数据接着上一根有效数据算。
macro_rules! skip_non_finite {
    ($($value:expr),+ $(,)?) => {
        if !($($value.is_finite())&&+) {
            return None;
        }
    };
}

/// TA-Lib `TA_EPSILON`。
pub(crate) const TA_EPSILON: f64 = 1e-14;

/// TA-Lib `TA_IS_ZERO`: 绝对值小于 1e-14。
#[inline]
pub(crate) fn is_zero(value: f64) -> bool {
    value > -TA_EPSILON && value < TA_EPSILON
}

/// TA-Lib `TA_IS_ZERO_SCALED`: 差值相对两个操作数的量级可忽略。
#[inline]
pub(crate) fn is_zero_scaled(value: f64, scale: f64) -> bool {
    value.abs() <= TA_EPSILON * scale
}

/// 真实波幅 `max(H-L, |H-前收|, |L-前收|)`。
#[inline]
pub(crate) fn true_range(high: f64, low: f64, prev_close: f64) -> f64 {
    (high - low)
        .max((high - prev_close).abs())
        .max((low - prev_close).abs())
}

mod momentum;
mod moving_average;
mod trend;
mod volatility;
mod volume;

pub use momentum::{CMO, MOM, ROC, ROCP, ROCR, ROCR100, RSI, WILLR};
pub use moving_average::{
    ABS, ACOS, ADD, APO, ASIN, ATAN, AVGDEV, CEIL, CLAMP01, CLIP, COS, COSH, CUBE, DEG2RAD, DEMA,
    DIV, EMA, EXP, EXPM1, FLOOR, HT_TRENDLINE, INV_SQRT, KAMA, LN, LOG1P, LOG10, MACD, MAMA, MAX,
    MAX2, MAXINDEX, MIDPOINT, MIN, MIN2, MININDEX, MINMAX, MINMAXINDEX, MOD, MULT, POW, PPO, RANGE,
    RECIP, ROUND, SIGN, SIN, SINH, SMA, SQ, SQRT, SUB, SUM, T3, TAN, TANH, TEMA, TRIMA, TRIX, WMA,
};
pub use trend::{
    ADX, ADXR, AROON, AROONOSC, BETA, CCI, CORREL, COVAR, DX, LINEARREG, LINEARREG_ANGLE,
    LINEARREG_INTERCEPT, LINEARREG_R2, LINEARREG_SLOPE, MINUS_DI, PLUS_DI, SAR, STOCH, TSF, ULTOSC,
};
pub use volatility::{
    ATR, AVGPRICE, BollingerBands, HeikinAshi, MEDPRICE, MIDPRICE, NATR, STDDEV, TRANGE, TYPPRICE,
    VAR, WCLPRICE,
};
pub use volume::{AD, ADOSC, BOP, MFI, OBV};

/// 复制一个内建增量指标的完整状态, 返回同类型的新对象.
///
/// 服务于"试算不提交"(intrabar peek): 用未闭合窗口的快照喂副本算出临时值, 原对象
/// 状态不受污染。所有内建指标都 `#[derive(Clone)]`, 但 pyo3 不会为此暴露
/// `__copy__`, Python 侧 `copy.copy` / `deepcopy` / `pickle` 一律 TypeError —— 这里是
/// 唯一的复制入口。非内建指标(用户自写的 Python 类)请走 `copy.deepcopy`。
///
/// :param indicator: 任一内建增量指标实例(``SMA`` / ``EMA`` / ``MACD`` …)
/// :return: 状态完全相同的新实例
/// :raises TypeError: 传入对象不是内建增量指标
#[gen_stub_pyfunction]
#[pyfunction]
pub fn clone_indicator(py: Python<'_>, indicator: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    macro_rules! try_clone {
        ($($t:ty),+ $(,)?) => {
            $(
                if let Ok(cell) = indicator.cast::<$t>() {
                    let cloned: $t = cell.borrow().clone();
                    return Ok(Py::new(py, cloned)?.into_any());
                }
            )+
        };
    }
    try_clone!(
        CMO,
        MOM,
        ROC,
        ROCP,
        ROCR,
        ROCR100,
        RSI,
        WILLR,
        ABS,
        ACOS,
        ADD,
        APO,
        ASIN,
        ATAN,
        AVGDEV,
        CEIL,
        CLAMP01,
        CLIP,
        COS,
        COSH,
        CUBE,
        DEG2RAD,
        DEMA,
        DIV,
        EMA,
        EXP,
        EXPM1,
        FLOOR,
        HT_TRENDLINE,
        INV_SQRT,
        KAMA,
        LN,
        LOG1P,
        LOG10,
        MACD,
        MAMA,
        MAX,
        MAX2,
        MAXINDEX,
        MIDPOINT,
        MIN,
        MIN2,
        MININDEX,
        MINMAX,
        MINMAXINDEX,
        MOD,
        MULT,
        POW,
        PPO,
        RANGE,
        RECIP,
        ROUND,
        SIGN,
        SIN,
        SINH,
        SMA,
        SQ,
        SQRT,
        SUB,
        SUM,
        T3,
        TAN,
        TANH,
        TEMA,
        TRIMA,
        TRIX,
        WMA,
        ADX,
        ADXR,
        AROON,
        AROONOSC,
        BETA,
        CCI,
        CORREL,
        COVAR,
        DX,
        LINEARREG,
        LINEARREG_ANGLE,
        LINEARREG_INTERCEPT,
        LINEARREG_R2,
        LINEARREG_SLOPE,
        MINUS_DI,
        PLUS_DI,
        SAR,
        STOCH,
        TSF,
        ULTOSC,
        ATR,
        AVGPRICE,
        BollingerBands,
        HeikinAshi,
        MEDPRICE,
        MIDPRICE,
        NATR,
        STDDEV,
        TRANGE,
        TYPPRICE,
        VAR,
        WCLPRICE,
        AD,
        ADOSC,
        BOP,
        MFI,
        OBV,
    );
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "clone_indicator 只接受内建增量指标, 收到 {}; 用户自写的指标请用 copy.deepcopy",
        indicator.get_type().name()?
    )))
}

pub fn register_py_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(clone_indicator, m)?)?;
    moving_average::register_classes(m)?;
    momentum::register_classes(m)?;
    trend::register_classes(m)?;
    volatility::register_classes(m)?;
    volume::register_classes(m)?;
    Ok(())
}
