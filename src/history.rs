use crate::model::Bar;
use rust_decimal::prelude::ToPrimitive;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SymbolHistory {
    pub timestamps: Vec<i64>,
    pub opens: Vec<f64>,
    pub highs: Vec<f64>,
    pub lows: Vec<f64>,
    pub closes: Vec<f64>,
    pub volumes: Vec<f64>,
    pub capacity: usize,
}

impl SymbolHistory {
    pub fn new(capacity: usize) -> Self {
        SymbolHistory {
            timestamps: Vec::with_capacity(capacity),
            opens: Vec::with_capacity(capacity),
            highs: Vec::with_capacity(capacity),
            lows: Vec::with_capacity(capacity),
            closes: Vec::with_capacity(capacity),
            volumes: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, bar: &Bar) {
        if self.capacity == 0 {
            return;
        }

        if self.timestamps.len() >= self.capacity {
            // Simple O(N) removal.
            // Ideally we would use a RingBuffer, but Vec is easier for numpy integration (contiguous memory).
            // For typical history depths (e.g. < 5000), this is very fast.
            self.timestamps.remove(0);
            self.opens.remove(0);
            self.highs.remove(0);
            self.lows.remove(0);
            self.closes.remove(0);
            self.volumes.remove(0);
        }

        self.timestamps.push(bar.timestamp);
        self.opens.push(bar.open.to_f64().unwrap_or(0.0));
        self.highs.push(bar.high.to_f64().unwrap_or(0.0));
        self.lows.push(bar.low.to_f64().unwrap_or(0.0));
        self.closes.push(bar.close.to_f64().unwrap_or(0.0));
        self.volumes.push(bar.volume.to_f64().unwrap_or(0.0));
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
