use std::sync::{
    mpsc::{self, Receiver, Sender},
    Mutex,
};

use crate::event::Event;


/// 事件管理器
/// 负责事件队列的分发和处理
pub struct EventManager {
    tx: Sender<Event>,
    rx: Option<Mutex<Receiver<Event>>>,
}

impl EventManager {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        EventManager {
            tx,
            rx: Some(Mutex::new(rx)),
        }
    }

    /// 发送事件
    pub fn send(&self, event: Event) -> Result<(), std::sync::mpsc::SendError<Event>> {
        self.tx.send(event)
    }

    /// 获取发送端 (用于克隆)
    pub fn sender(&self) -> Sender<Event> {
        self.tx.clone()
    }

    /// 尝试接收事件 (非阻塞)
    pub fn try_recv(&self) -> Option<Event> {
        if let Some(rx_mutex) = &self.rx {
            if let Ok(rx) = rx_mutex.lock() {
                if let Ok(event) = rx.try_recv() {
                    return Some(event);
                }
            }
        }
        None
    }

    /// 接收事件 (阻塞)
    /// 注意：如果 rx 被其他线程锁住，这里也会阻塞等待锁
    #[allow(dead_code)]
    pub fn recv(&self) -> Option<Event> {
        if let Some(rx_mutex) = &self.rx {
            if let Ok(rx) = rx_mutex.lock() {
                if let Ok(event) = rx.recv() {
                    return Some(event);
                }
            }
        }
        None
    }
}

impl Default for EventManager {
    fn default() -> Self {
        Self::new()
    }
}
