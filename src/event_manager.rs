use crossbeam_channel::{Receiver, Sender, unbounded};

use crate::event::Event;

/// 事件管理器
/// 负责事件队列的分发和处理
pub struct EventManager {
    tx: Sender<Event>,
    rx: Option<Receiver<Event>>,
}

impl EventManager {
    pub fn new() -> Self {
        let (tx, rx) = unbounded();
        EventManager { tx, rx: Some(rx) }
    }

    /// 发送事件
    pub fn send(&self, event: Event) -> Result<(), Box<crossbeam_channel::SendError<Event>>> {
        self.tx.send(event).map_err(Box::new)
    }

    /// 获取发送端 (用于克隆)
    pub fn sender(&self) -> Sender<Event> {
        self.tx.clone()
    }

    /// 借出接收端 (用于与行情通道一起 `Select` 多路等待)
    ///
    /// 只用于**探测就绪**, 不在此消费事件——消费仍由 `ChannelProcessor` 负责,
    /// 否则事件会被吞掉。
    pub fn receiver(&self) -> Option<&Receiver<Event>> {
        self.rx.as_ref()
    }

    /// 尝试接收事件 (非阻塞)
    pub fn try_recv(&self) -> Option<Event> {
        if let Some(rx) = &self.rx {
            rx.try_recv().ok()
        } else {
            None
        }
    }

    /// 接收事件 (阻塞)
    #[allow(dead_code)]
    pub fn recv(&self) -> Option<Event> {
        if let Some(rx) = &self.rx {
            rx.recv().ok()
        } else {
            None
        }
    }
}

impl Default for EventManager {
    fn default() -> Self {
        Self::new()
    }
}
