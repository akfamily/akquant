//! 运行中合约登记端口。
//!
//! 实盘会话跨夜盘、日盘连续运行时, 交易所新挂的合约(如期权新行权价)要在运行中登记
//! 进引擎。`Engine::run(&mut self)` 独占引擎对象, 引擎线程之外不能直接调
//! `add_instrument`; 本端口持有引擎事件通道的 sender 克隆(同 `SignalPort`), 把合约
//! 作为 `Event::RegisterInstruments` 投递, 由 `ChannelProcessor` 在下一轮循环登记。

use crossbeam_channel::Sender;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::event::Event;
use crate::model::Instrument;

/// 运行中合约登记端口。
///
/// 由 `Engine.instrument_port()` 取得, **必须在 `run()` 之前取**(取的是 channel
/// sender 的克隆, 取到后即与引擎对象解耦, 可安全交给任意线程)。
#[gen_stub_pyclass]
#[pyclass]
pub struct InstrumentPort {
    tx: Sender<Event>,
}

impl InstrumentPort {
    pub(crate) fn new(tx: Sender<Event>) -> Self {
        Self { tx }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl InstrumentPort {
    /// 投递一批合约, 引擎在处理下一个事件前登记; 同名合约按新参数覆盖。
    ///
    /// 线程安全: 可从任意线程调用。
    ///
    /// :param instruments: 合约列表
    fn register(&self, instruments: Vec<Instrument>) -> PyResult<()> {
        if instruments.is_empty() {
            return Ok(());
        }
        self.tx
            .send(Event::RegisterInstruments(instruments))
            .map_err(|err| PyRuntimeError::new_err(format!("引擎已停止, 无法登记合约: {err}")))
    }
}
