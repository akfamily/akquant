use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;

mod analysis;
mod clock;
mod context;
mod data;
mod engine;
mod event;
mod execution;
mod history;
mod indicators;
mod market;
mod model;
mod portfolio;
mod risk;

use analysis::{BacktestResult, ClosedTrade, PerformanceMetrics, TradePnL};
use context::StrategyContext;
use data::{DataFeed, BarAggregator, from_arrays};
use engine::Engine;
use indicators::{ATR, BollingerBands, EMA, MACD, RSI, SMA};
use model::{
    AssetType, Bar, ExecutionMode, Instrument, Order, OrderSide, OrderStatus, OrderType, Tick,
    TimeInForce, Trade, TradingSession,
};
use portfolio::Portfolio;
use risk::{RiskConfig, RiskManager};

/// 使用 Rust 实现的 Python 模块
#[pymodule]
fn akquant(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Bar>()?;
    m.add_function(wrap_pyfunction!(from_arrays, m)?)?;
    m.add_class::<Tick>()?;
    m.add_class::<DataFeed>()?;
    m.add_class::<BarAggregator>()?;
    m.add_class::<Engine>()?;
    m.add_class::<StrategyContext>()?;
    m.add_class::<Order>()?;
    m.add_class::<Trade>()?;
    m.add_class::<OrderType>()?;
    m.add_class::<OrderSide>()?;
    m.add_class::<OrderStatus>()?;
    m.add_class::<TimeInForce>()?;
    m.add_class::<AssetType>()?;
    m.add_class::<Instrument>()?;
    m.add_class::<ExecutionMode>()?;
    m.add_class::<TradingSession>()?;
    m.add_class::<Portfolio>()?;
    m.add_class::<PerformanceMetrics>()?;
    m.add_class::<BacktestResult>()?;
    m.add_class::<TradePnL>()?;
    m.add_class::<ClosedTrade>()?;
    m.add_class::<RiskManager>()?;
    m.add_class::<RiskConfig>()?;
    m.add_class::<SMA>()?;
    m.add_class::<EMA>()?;
    m.add_class::<MACD>()?;
    m.add_class::<RSI>()?;
    m.add_class::<BollingerBands>()?;
    m.add_class::<ATR>()?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
