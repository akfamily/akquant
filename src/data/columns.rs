//! 列式 Bar 存储 (Structure-of-Arrays, f64 列).
//!
//! 市场数据 (价格/成交量) 以 f64 列存储 —— 输入本就是 f64 numpy, 故无精度损失;
//! 重构 `Bar` 时对"价格→钱"边界仍用 `Decimal::from_f64`, 与 `batch::from_arrays`
//! **完全一致**, 保证 golden 逐位不变 (混合路线, 详见 docs/zh/meta/columnar-rfc.md §6/§8)。

use crate::log_context::{AkqLogContext, format_event_time_nanos, render_log_message};
use crate::model::Bar;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rust_decimal::prelude::*;
use std::collections::HashMap;

fn warn_invalid_numeric(field_name: &str, value: f64, symbol: &str, timestamp_ns: i64) {
    log::warn!(
        "{}",
        render_log_message(
            format!("Invalid {field_name} {value}, defaulting to 0.0"),
            AkqLogContext::new()
                .phase("data")
                .symbol(symbol)
                .event_time(timestamp_ns)
                .event_time_iso(format_event_time_nanos(timestamp_ns)),
        )
    );
}

#[inline]
fn dec_or_zero(value: f64, field_name: &str, symbol: &str, timestamp_ns: i64) -> Decimal {
    Decimal::from_f64(value).unwrap_or_else(|| {
        warn_invalid_numeric(field_name, value, symbol, timestamp_ns);
        Decimal::ZERO
    })
}

/// 列式 Bar 数据 (SoA). 时间戳与 OHLCV 以并列向量存储, symbol 逐行保存,
/// extra 数值列以 `列名 -> 列` 形式保存。
#[derive(Debug, Clone, Default)]
pub struct BarColumns {
    pub ts: Vec<i64>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
    pub symbols: Vec<String>,
    pub extra: HashMap<String, Vec<f64>>,
}

impl BarColumns {
    /// 行数.
    pub fn len(&self) -> usize {
        self.ts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ts.is_empty()
    }

    /// 从 Python numpy 数组批量构建列式存储 (等价于 `from_arrays` 的取数与校验).
    #[allow(clippy::too_many_arguments)]
    pub fn from_py_arrays(
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
    ) -> PyResult<Self> {
        let timestamps: PyReadonlyArray1<i64> = timestamps.extract()?;
        let opens: PyReadonlyArray1<f64> = opens.extract()?;
        let highs: PyReadonlyArray1<f64> = highs.extract()?;
        let lows: PyReadonlyArray1<f64> = lows.extract()?;
        let closes: PyReadonlyArray1<f64> = closes.extract()?;
        let volumes: PyReadonlyArray1<f64> = volumes.extract()?;

        let ts_view = timestamps.as_array();
        let open_view = opens.as_array();
        let high_view = highs.as_array();
        let low_view = lows.as_array();
        let close_view = closes.as_array();
        let volume_view = volumes.as_array();

        let len = ts_view.len();
        if open_view.len() != len
            || high_view.len() != len
            || low_view.len() != len
            || close_view.len() != len
            || volume_view.len() != len
        {
            return Err(PyValueError::new_err(
                "All arrays must have the same length",
            ));
        }

        if let Some(ref syms) = symbols
            && syms.len() != len
        {
            return Err(PyValueError::new_err(
                "symbols array must have the same length as other arrays",
            ));
        }

        // extra 列: 校验长度并拷贝为 owned 列
        let mut extra_map: HashMap<String, Vec<f64>> = HashMap::new();
        if let Some(ref extra_data) = extra {
            for (key, val) in extra_data {
                let arr: PyReadonlyArray1<f64> = val.extract(py)?;
                let view = arr.as_array();
                if view.len() != len {
                    return Err(PyValueError::new_err(format!(
                        "Extra array '{}' must have the same length as other arrays",
                        key
                    )));
                }
                extra_map.insert(key.clone(), view.to_vec());
            }
        }

        // 逐行解析 symbol (与 from_arrays 相同优先级: symbols[i] > symbol > "UNKNOWN")
        let mut sym_vec: Vec<String> = Vec::with_capacity(len);
        for i in 0..len {
            let sym = if let Some(ref syms) = symbols {
                syms[i].clone()
            } else if let Some(ref s) = symbol {
                s.clone()
            } else {
                "UNKNOWN".to_string()
            };
            sym_vec.push(sym);
        }

        Ok(BarColumns {
            ts: ts_view.to_vec(),
            open: open_view.to_vec(),
            high: high_view.to_vec(),
            low: low_view.to_vec(),
            close: close_view.to_vec(),
            volume: volume_view.to_vec(),
            symbols: sym_vec,
            extra: extra_map,
        })
    }

    /// 重构第 `i` 行为 `Bar` (价格→Decimal, 与 `from_arrays` 完全一致).
    pub fn reconstruct_bar(&self, i: usize) -> Bar {
        let ts = self.ts[i];
        let sym = &self.symbols[i];
        let mut bar_extra: HashMap<String, f64> = HashMap::new();
        for (k, col) in &self.extra {
            bar_extra.insert(k.clone(), col[i]);
        }
        Bar {
            timestamp: ts,
            open: dec_or_zero(self.open[i], "open price", sym, ts),
            high: dec_or_zero(self.high[i], "high price", sym, ts),
            low: dec_or_zero(self.low[i], "low price", sym, ts),
            close: dec_or_zero(self.close[i], "close price", sym, ts),
            volume: dec_or_zero(self.volume[i], "volume", sym, ts),
            symbol: sym.clone(),
            extra: bar_extra,
        }
    }

    /// 重构为 `Vec<Bar>` (与 `from_arrays` 返回等价).
    pub fn to_bars(&self) -> Vec<Bar> {
        (0..self.len()).map(|i| self.reconstruct_bar(i)).collect()
    }

    /// 第 `i` 行时间戳.
    #[inline]
    pub fn timestamp_at(&self, i: usize) -> i64 {
        self.ts[i]
    }

    /// 追加另一段列式数据 (多次 add_arrays 时拼接).
    pub fn append(&mut self, mut other: BarColumns) {
        self.ts.append(&mut other.ts);
        self.open.append(&mut other.open);
        self.high.append(&mut other.high);
        self.low.append(&mut other.low);
        self.close.append(&mut other.close);
        self.volume.append(&mut other.volume);
        self.symbols.append(&mut other.symbols);
        for (key, mut col) in other.extra {
            self.extra.entry(key).or_default().append(&mut col);
        }
    }

    /// 按时间戳稳定排序，时间戳相等时按 symbol 字典序排序（保证确定性，issue #211）。
    pub fn sort_by_timestamp(&mut self) {
        let n = self.len();
        let mut idx: Vec<usize> = (0..n).collect();
        // 时间戳相等时按 symbol 排序，确保不同 symbol 插入顺序不影响回测结果
        idx.sort_by(|&i, &j| {
            self.ts[i].cmp(&self.ts[j])
                .then_with(|| self.symbols[i].cmp(&self.symbols[j]))
        });

        self.ts = idx.iter().map(|&i| self.ts[i]).collect();
        self.open = idx.iter().map(|&i| self.open[i]).collect();
        self.high = idx.iter().map(|&i| self.high[i]).collect();
        self.low = idx.iter().map(|&i| self.low[i]).collect();
        self.close = idx.iter().map(|&i| self.close[i]).collect();
        self.volume = idx.iter().map(|&i| self.volume[i]).collect();
        self.symbols = idx.iter().map(|&i| self.symbols[i].clone()).collect();
        for col in self.extra.values_mut() {
            *col = idx.iter().map(|&i| col[i]).collect();
        }
    }
}
