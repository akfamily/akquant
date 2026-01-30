use std::cmp::Ordering;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Timer {
    pub timestamp: i64,
    pub payload: String,
}

impl Ord for Timer {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for Min-Heap behavior in BinaryHeap
        other.timestamp.cmp(&self.timestamp)
    }
}

impl PartialOrd for Timer {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
