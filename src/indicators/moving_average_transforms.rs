use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;

macro_rules! define_unary_indicator {
    ($name:ident, $calc:expr) => {
        #[gen_stub_pyclass]
        #[pyclass(from_py_object)]
        #[allow(non_camel_case_types)]
        #[derive(Debug, Clone)]
        pub struct $name {
            current_value: Option<f64>,
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl $name {
            #[new]
            pub fn new() -> Self {
                Self {
                    current_value: None,
                }
            }

            pub fn update(&mut self, value: f64) -> Option<f64> {
                skip_non_finite!(value);
                self.current_value = Some(($calc)(value));
                self.current_value
            }

            pub fn update_many<'py>(
                &mut self,
                py: Python<'py>,
                values: Vec<f64>,
            ) -> Bound<'py, PyArray1<f64>> {
                let mut out = Vec::with_capacity(values.len());
                for value in values {
                    out.push(self.update(value).unwrap_or(f64::NAN));
                }
                PyArray1::from_vec(py, out)
            }

            #[getter]
            pub fn value(&self) -> Option<f64> {
                self.current_value
            }
        }
    };
}

macro_rules! define_binary_indicator {
    ($name:ident, $calc:expr) => {
        #[gen_stub_pyclass]
        #[pyclass(from_py_object)]
        #[derive(Debug, Clone)]
        pub struct $name {
            current_value: Option<f64>,
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl $name {
            #[new]
            pub fn new() -> Self {
                Self {
                    current_value: None,
                }
            }

            pub fn update(&mut self, left: f64, right: f64) -> Option<f64> {
                skip_non_finite!(left, right);
                self.current_value = Some(($calc)(left, right));
                self.current_value
            }

            pub fn update_many_dual<'py>(
                &mut self,
                py: Python<'py>,
                lefts: Vec<f64>,
                rights: Vec<f64>,
            ) -> PyResult<Bound<'py, PyArray1<f64>>> {
                if lefts.len() != rights.len() {
                    return Err(PyValueError::new_err("lefts/rights length mismatch"));
                }
                let mut out = Vec::with_capacity(lefts.len());
                for (left, right) in lefts.into_iter().zip(rights.into_iter()) {
                    out.push(self.update(left, right).unwrap_or(f64::NAN));
                }
                Ok(PyArray1::from_vec(py, out))
            }

            #[getter]
            pub fn value(&self) -> Option<f64> {
                self.current_value
            }
        }
    };
}

define_unary_indicator!(LOG10, |value: f64| if value > 0.0 {
    value.log10()
} else {
    f64::NAN
});
define_unary_indicator!(SQRT, |value: f64| if value >= 0.0 {
    value.sqrt()
} else {
    f64::NAN
});
define_unary_indicator!(CEIL, |value: f64| value.ceil());
define_unary_indicator!(FLOOR, |value: f64| value.floor());
define_unary_indicator!(SIN, |value: f64| value.sin());
define_unary_indicator!(COS, |value: f64| value.cos());
define_unary_indicator!(TAN, |value: f64| value.tan());
define_unary_indicator!(ASIN, |value: f64| if (-1.0..=1.0).contains(&value) {
    value.asin()
} else {
    f64::NAN
});
define_unary_indicator!(ACOS, |value: f64| if (-1.0..=1.0).contains(&value) {
    value.acos()
} else {
    f64::NAN
});
define_unary_indicator!(ATAN, |value: f64| value.atan());
define_unary_indicator!(SINH, |value: f64| value.sinh());
define_unary_indicator!(COSH, |value: f64| value.cosh());
define_unary_indicator!(TANH, |value: f64| value.tanh());
define_unary_indicator!(EXP, |value: f64| value.exp());
define_unary_indicator!(ABS, |value: f64| value.abs());
define_unary_indicator!(SIGN, |value: f64| if value > 0.0 {
    1.0
} else if value < 0.0 {
    -1.0
} else {
    0.0
});
define_binary_indicator!(ADD, |left: f64, right: f64| left + right);
define_binary_indicator!(SUB, |left: f64, right: f64| left - right);
define_binary_indicator!(MULT, |left: f64, right: f64| left * right);
define_binary_indicator!(
    DIV,
    |left: f64, right: f64| if right.abs() <= f64::EPSILON {
        f64::NAN
    } else {
        left / right
    }
);
define_binary_indicator!(MAX2, |left: f64, right: f64| left.max(right));
define_binary_indicator!(MIN2, |left: f64, right: f64| left.min(right));
define_unary_indicator!(ROUND, |value: f64| value.round());
define_binary_indicator!(POW, |left: f64, right: f64| left.powf(right));
define_binary_indicator!(
    MOD,
    |left: f64, right: f64| if right.abs() <= f64::EPSILON {
        f64::NAN
    } else {
        left % right
    }
);
define_unary_indicator!(CLAMP01, |value: f64| value.clamp(0.0, 1.0));
define_unary_indicator!(SQ, |value: f64| value * value);
define_unary_indicator!(CUBE, |value: f64| value * value * value);
define_unary_indicator!(RECIP, |value: f64| if value.abs() <= f64::EPSILON {
    f64::NAN
} else {
    1.0 / value
});
define_unary_indicator!(INV_SQRT, |value: f64| if value > 0.0 {
    1.0 / value.sqrt()
} else {
    f64::NAN
});
define_unary_indicator!(LOG1P, |value: f64| value.ln_1p());
define_unary_indicator!(EXPM1, |value: f64| value.exp_m1());
define_unary_indicator!(DEG2RAD, |value: f64| value.to_radians());

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct CLIP {
    current_value: Option<f64>,
}

#[gen_stub_pymethods]
#[pymethods]
impl CLIP {
    #[new]
    pub fn new() -> Self {
        Self {
            current_value: None,
        }
    }

    pub fn update(&mut self, value: f64, min_value: f64, max_value: f64) -> Option<f64> {
        skip_non_finite!(value, min_value, max_value);
        let lo = min_value.min(max_value);
        let hi = min_value.max(max_value);
        self.current_value = Some(value.clamp(lo, hi));
        self.current_value
    }

    pub fn update_many_clip<'py>(
        &mut self,
        py: Python<'py>,
        values: Vec<f64>,
        min_values: Vec<f64>,
        max_values: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if values.len() != min_values.len() || values.len() != max_values.len() {
            return Err(PyValueError::new_err(
                "values/min_values/max_values length mismatch",
            ));
        }
        let mut out = Vec::with_capacity(values.len());
        for (value, (min_value, max_value)) in values
            .into_iter()
            .zip(min_values.into_iter().zip(max_values.into_iter()))
        {
            out.push(self.update(value, min_value, max_value).unwrap_or(f64::NAN));
        }
        Ok(PyArray1::from_vec(py, out))
    }

    #[getter]
    pub fn value(&self) -> Option<f64> {
        self.current_value
    }
}

pub fn register_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LOG10>()?;
    m.add_class::<SQRT>()?;
    m.add_class::<CEIL>()?;
    m.add_class::<FLOOR>()?;
    m.add_class::<SIN>()?;
    m.add_class::<COS>()?;
    m.add_class::<TAN>()?;
    m.add_class::<ASIN>()?;
    m.add_class::<ACOS>()?;
    m.add_class::<ATAN>()?;
    m.add_class::<SINH>()?;
    m.add_class::<COSH>()?;
    m.add_class::<TANH>()?;
    m.add_class::<EXP>()?;
    m.add_class::<ABS>()?;
    m.add_class::<SIGN>()?;
    m.add_class::<ADD>()?;
    m.add_class::<SUB>()?;
    m.add_class::<MULT>()?;
    m.add_class::<DIV>()?;
    m.add_class::<MAX2>()?;
    m.add_class::<MIN2>()?;
    m.add_class::<CLIP>()?;
    m.add_class::<ROUND>()?;
    m.add_class::<POW>()?;
    m.add_class::<MOD>()?;
    m.add_class::<CLAMP01>()?;
    m.add_class::<SQ>()?;
    m.add_class::<CUBE>()?;
    m.add_class::<RECIP>()?;
    m.add_class::<INV_SQRT>()?;
    m.add_class::<LOG1P>()?;
    m.add_class::<EXPM1>()?;
    m.add_class::<DEG2RAD>()?;
    Ok(())
}
