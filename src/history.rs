use crate::model::Bar;
use rust_decimal::prelude::ToPrimitive;
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone)]
pub struct SymbolHistory {
    pub timestamps: VecDeque<i64>,
    pub opens: VecDeque<f64>,
    pub highs: VecDeque<f64>,
    pub lows: VecDeque<f64>,
    pub closes: VecDeque<f64>,
    pub volumes: VecDeque<f64>,
    pub capacity: usize,
}

impl SymbolHistory {
    pub fn new(capacity: usize) -> Self {
        SymbolHistory {
            timestamps: VecDeque::with_capacity(capacity),
            opens: VecDeque::with_capacity(capacity),
            highs: VecDeque::with_capacity(capacity),
            lows: VecDeque::with_capacity(capacity),
            closes: VecDeque::with_capacity(capacity),
            volumes: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, bar: &Bar) {
        if self.capacity == 0 {
            // If capacity is 0, we can still store unbounded history?
            // Or should we interpret 0 as "unlimited" or "none"?
            // Usually 0 means None.
            // But if we want MAE/MFE, we need history.
            // Let's assume 0 means disabled.
            return;
        }

        if self.timestamps.len() >= self.capacity {
            // O(1) removal from front
            self.timestamps.pop_front();
            self.opens.pop_front();
            self.highs.pop_front();
            self.lows.pop_front();
            self.closes.pop_front();
            self.volumes.pop_front();
        }

        self.timestamps.push_back(bar.timestamp);
        self.opens.push_back(bar.open.to_f64().unwrap_or(0.0));
        self.highs.push_back(bar.high.to_f64().unwrap_or(0.0));
        self.lows.push_back(bar.low.to_f64().unwrap_or(0.0));
        self.closes.push_back(bar.close.to_f64().unwrap_or(0.0));
        self.volumes.push_back(bar.volume.to_f64().unwrap_or(0.0));
    }
}

#[derive(Debug)]
pub struct HistoryBuffer {
    pub data: HashMap<String, SymbolHistory>,
    pub default_capacity: usize,
}

impl HistoryBuffer {
    pub fn new(default_capacity: usize) -> Self {
        HistoryBuffer {
            data: HashMap::new(),
            default_capacity,
        }
    }

    pub fn set_capacity(&mut self, capacity: usize) {
        self.default_capacity = capacity;
        // Clear existing data when capacity changes to avoid complexity
        self.data.clear();
    }

    pub fn update(&mut self, bar: &Bar) {
        if self.default_capacity == 0 {
            return;
        }

        let history = self
            .data
            .entry(bar.symbol.clone())
            .or_insert_with(|| SymbolHistory::new(self.default_capacity));

        history.push(bar);
    }

    pub fn get_history(&self, symbol: &str) -> Option<&SymbolHistory> {
        self.data.get(symbol)
    }
}
