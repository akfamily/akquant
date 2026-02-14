use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AkQuantError {
    #[error("Data error: {0}")]
    DataError(String),

    #[error("Order error: {0}")]
    OrderError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Market error: {0}")]
    MarketError(String),

    #[error("Execution error: {0}")]
    ExecutionError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("CSV error: {0}")]
    CsvError(#[from] csv::Error),

    #[error("Python error: {0}")]
    PythonError(#[from] PyErr),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl From<AkQuantError> for PyErr {
    fn from(err: AkQuantError) -> PyErr {
        match err {
            AkQuantError::DataError(msg) => PyValueError::new_err(msg),
            AkQuantError::OrderError(msg) => PyRuntimeError::new_err(msg),
            AkQuantError::ConfigError(msg) => PyValueError::new_err(msg),
            AkQuantError::MarketError(msg) => PyRuntimeError::new_err(msg),
            AkQuantError::ExecutionError(msg) => PyRuntimeError::new_err(msg),
            AkQuantError::IoError(e) => PyIOError::new_err(e.to_string()),
            AkQuantError::CsvError(e) => PyValueError::new_err(e.to_string()),
            AkQuantError::PythonError(e) => e,
            AkQuantError::Unknown(msg) => PyRuntimeError::new_err(msg),
        }
    }
}

#[allow(dead_code)]
pub type Result<T> = std::result::Result<T, AkQuantError>;
