use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;

/// 平均 K 线 (Heikin-Ashi) 指标, 输出平滑后的 (open, high, low, close)。
///
/// ha_close = (O+H+L+C)/4; ha_open 第一根取 (O+C)/2, 之后取前一根
/// (ha_open+ha_close)/2; ha_high = max(H, ha_open, ha_close);
/// ha_low = min(L, ha_open, ha_close)。第一根即有值。
#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct HeikinAshi {
    prev_ha_open: Option<f64>,
    prev_ha_close: Option<f64>,
    current_value: Option<(f64, f64, f64, f64)>,
}

#[gen_stub_pymethods]
#[pymethods]
impl HeikinAshi {
    #[new]
    pub fn new() -> Self {
        HeikinAshi {
            prev_ha_open: None,
            prev_ha_close: None,
            current_value: None,
        }
    }

    pub fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<(f64, f64, f64, f64)> {
        // 任一输入非有限值(NaN/inf, 例如停牌缺价)时本根不出值, 也不推进状态:
        // 否则 NaN 会写进 prev_ha_*, 此后每一根的 ha_open 都是 NaN。
        if !(open.is_finite() && high.is_finite() && low.is_finite() && close.is_finite()) {
            return None;
        }
        let ha_close = (open + high + low + close) / 4.0;
        let ha_open = match (self.prev_ha_open, self.prev_ha_close) {
            (Some(po), Some(pc)) => (po + pc) / 2.0,
            _ => (open + close) / 2.0,
        };
        let ha_high = high.max(ha_open).max(ha_close);
        let ha_low = low.min(ha_open).min(ha_close);
        self.prev_ha_open = Some(ha_open);
        self.prev_ha_close = Some(ha_close);
        self.current_value = Some((ha_open, ha_high, ha_low, ha_close));
        self.current_value
    }

    #[allow(clippy::type_complexity)]
    pub fn update_many_ohlc<'py>(
        &mut self,
        py: Python<'py>,
        opens: Vec<f64>,
        highs: Vec<f64>,
        lows: Vec<f64>,
        closes: Vec<f64>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let n = opens.len();
        if highs.len() != n || lows.len() != n || closes.len() != n {
            return Err(PyValueError::new_err(
                "opens/highs/lows/closes length mismatch",
            ));
        }
        let mut o = Vec::with_capacity(n);
        let mut h = Vec::with_capacity(n);
        let mut l = Vec::with_capacity(n);
        let mut c = Vec::with_capacity(n);
        for i in 0..n {
            let (ao, ah, al, ac) = self
                .update(opens[i], highs[i], lows[i], closes[i])
                .unwrap_or((f64::NAN, f64::NAN, f64::NAN, f64::NAN));
            o.push(ao);
            h.push(ah);
            l.push(al);
            c.push(ac);
        }
        Ok((
            PyArray1::from_vec(py, o),
            PyArray1::from_vec(py, h),
            PyArray1::from_vec(py, l),
            PyArray1::from_vec(py, c),
        ))
    }

    #[getter]
    pub fn value(&self) -> Option<(f64, f64, f64, f64)> {
        self.current_value
    }
}

impl Default for HeikinAshi {
    fn default() -> Self {
        Self::new()
    }
}

pub fn register_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<HeikinAshi>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close_to(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn heikin_ashi_first_bar_has_value() {
        let mut ha = HeikinAshi::new();
        assert!(ha.value().is_none());
        // O=10 H=14 L=8 C=12 -> close=11, open=11, high=14, low=8
        let (o, h, l, c) = ha.update(10.0, 14.0, 8.0, 12.0).unwrap();
        assert!(close_to(o, 11.0) && close_to(h, 14.0) && close_to(l, 8.0) && close_to(c, 11.0));
        assert_eq!(ha.value(), Some((o, h, l, c)));
    }

    #[test]
    fn heikin_ashi_three_bars_hand_computed() {
        let mut ha = HeikinAshi::new();
        ha.update(10.0, 14.0, 8.0, 12.0);
        // bar2 O=12 H=16 L=11 C=15: close=13.5, open=(11+11)/2=11, high=16, low=min(11,11,13.5)=11
        let (o, h, l, c) = ha.update(12.0, 16.0, 11.0, 15.0).unwrap();
        assert!(close_to(o, 11.0) && close_to(h, 16.0) && close_to(l, 11.0) && close_to(c, 13.5));
        // bar3 O=15 H=15.5 L=9 C=10: close=12.375, open=(11+13.5)/2=12.25, high=15.5, low=9
        let (o, h, l, c) = ha.update(15.0, 15.5, 9.0, 10.0).unwrap();
        assert!(close_to(o, 12.25) && close_to(h, 15.5) && close_to(l, 9.0) && close_to(c, 12.375));
    }

    #[test]
    fn heikin_ashi_high_low_extend_to_ha_body() {
        let mut ha = HeikinAshi::new();
        // 构造 ha_open/ha_close 超出 H/L 的情形: 第二根极窄 H=L, 而 ha_open 来自前值
        ha.update(10.0, 20.0, 0.0, 20.0); // open=15, close=12.5
        // ha_open2=(15+12.5)/2=13.75, close2=(5+5+5+5)/4=5 -> high 取 13.75, low 取 5
        let (o, h, l, c) = ha.update(5.0, 5.0, 5.0, 5.0).unwrap();
        assert!(close_to(o, 13.75) && close_to(h, 13.75) && close_to(l, 5.0) && close_to(c, 5.0));
    }

    #[test]
    fn heikin_ashi_non_finite_bar_is_skipped_without_poisoning_state() {
        let mut ha = HeikinAshi::new();
        ha.update(10.0, 12.0, 9.0, 11.0); // open=10.5, close=10.5
        // 坏 bar: 不出值, 也不改状态
        assert!(ha.update(f64::NAN, 12.0, 9.0, 11.0).is_none());
        assert!(ha.update(10.0, f64::INFINITY, 9.0, 11.0).is_none());
        let last = ha.value().unwrap();
        assert!(close_to(last.0, 10.5) && close_to(last.3, 10.5));
        // 下一根好 bar 接着上一根好 bar 算: ha_open=(10.5+10.5)/2
        let (o, _, _, c) = ha.update(11.0, 13.0, 10.0, 12.0).unwrap();
        assert!(close_to(o, 10.5) && close_to(c, 11.5));
    }
}
