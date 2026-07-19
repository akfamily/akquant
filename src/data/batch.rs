use crate::data::columns::BarColumns;
use crate::model::Bar;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use std::collections::HashMap;

/// 从数组批量创建 Bar 列表 (Python 优化用 - Zero Copy).
///
/// 实现已下沉到 [`BarColumns`]: 先构建 f64 列式存储, 再逐行重构 `Bar`
/// (价格→`Decimal::from_f64`), 与旧实现逐位一致。
///
/// :param timestamps: 时间戳数组
/// :param opens: 开盘价数组
/// :param highs: 最高价数组
/// :param lows: 最低价数组
/// :param closes: 收盘价数组
/// :param volumes: 成交量数组
/// :param symbol: 标的代码 (可选)
/// :param symbols: 标的代码数组 (可选)
/// :param extra: 额外数据 (可选)
#[gen_stub_pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn from_arrays(
    timestamps: &Bound<'_, PyAny>,
    opens: &Bound<'_, PyAny>,
    highs: &Bound<'_, PyAny>,
    lows: &Bound<'_, PyAny>,
    closes: &Bound<'_, PyAny>,
    volumes: &Bound<'_, PyAny>,
    symbol: Option<String>,
    symbols: Option<Vec<String>>,
    extra: Option<HashMap<String, Py<PyAny>>>,
    py: Python<'_>,
) -> PyResult<Vec<Bar>> {
    let columns = BarColumns::from_py_arrays(
        timestamps, opens, highs, lows, closes, volumes, symbol, symbols, extra, py,
    )?;
    Ok(columns.to_bars())
}
