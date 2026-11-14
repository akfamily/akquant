//! 只追加的记录表, 克隆(取快照)的代价与长度无关。

use serde::de::Deserializer;
use serde::ser::{SerializeSeq, Serializer};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

/// 每块的记录数。块写满后不再改动, 各快照共享; 只有未满的最后一块会在"有快照
/// 在外、又要追加"时被复制, 所以这个数就是追加一条的复制上限。
const CHUNK: usize = 64;

/// 只追加的记录表。
///
/// 用在"引擎不断追加、策略上下文每个回调都要拿一份当时的快照"的地方(已平仓交易):
/// 用 `Arc<Vec<T>>` 的话, 上下文握着一份引用, 引擎每追加一条都得把整张表复制一遍,
/// 上下文换快照时再把旧表整个释放, 累计开销随记录数平方增长。
///
/// 这里把表切成定长的块: `clone()` 只复制两个 `Arc`; 追加时最多复制未满的那一块。
/// 快照拿到手之后内容不再变(与 `Arc<Vec<T>>` 的快照语义相同)。
///
/// 序列化成普通的序列, 与 `Vec<T>` / `Arc<Vec<T>>` 的编码相同, 新旧存档互通。
pub struct AppendLog<T> {
    /// 已写满的块, 每块恰好 `CHUNK` 条。
    sealed: Arc<Vec<Arc<Vec<T>>>>,
    /// 未写满的最后一块(0..CHUNK 条)。
    tail: Arc<Vec<T>>,
}

impl<T> AppendLog<T> {
    pub fn new() -> Self {
        Self {
            sealed: Arc::new(Vec::new()),
            tail: Arc::new(Vec::new()),
        }
    }

    pub fn len(&self) -> usize {
        self.sealed.len() * CHUNK + self.tail.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        match self.sealed.get(index / CHUNK) {
            Some(chunk) => chunk.get(index % CHUNK),
            None if index / CHUNK == self.sealed.len() => self.tail.get(index % CHUNK),
            None => None,
        }
    }

    pub fn last(&self) -> Option<&T> {
        self.len().checked_sub(1).and_then(|i| self.get(i))
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        self.sealed
            .iter()
            .flat_map(|chunk| chunk.iter())
            .chain(self.tail.iter())
    }
}

impl<T: Clone> AppendLog<T> {
    /// 追加一条。已发出去的快照不受影响。
    pub fn push(&mut self, value: T) {
        let tail = Arc::make_mut(&mut self.tail);
        tail.push(value);
        if tail.len() == CHUNK {
            let full = std::mem::replace(&mut self.tail, Arc::new(Vec::new()));
            Arc::make_mut(&mut self.sealed).push(full);
        }
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.iter().cloned().collect()
    }
}

impl<T: Clone> From<Vec<T>> for AppendLog<T> {
    fn from(values: Vec<T>) -> Self {
        let mut log = Self::new();
        for value in values {
            log.push(value);
        }
        log
    }
}

impl<T> Clone for AppendLog<T> {
    fn clone(&self) -> Self {
        Self {
            sealed: Arc::clone(&self.sealed),
            tail: Arc::clone(&self.tail),
        }
    }
}

impl<T> Default for AppendLog<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: fmt::Debug> fmt::Debug for AppendLog<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T> std::ops::Index<usize> for AppendLog<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        self.get(index).unwrap_or_else(|| {
            panic!("下标越界: 长度 {}, 下标 {}", self.len(), index);
        })
    }
}

impl<T: Serialize> Serialize for AppendLog<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        for item in self.iter() {
            seq.serialize_element(item)?;
        }
        seq.end()
    }
}

impl<'de, T: Deserialize<'de> + Clone> Deserialize<'de> for AppendLog<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Vec::<T>::deserialize(deserializer).map(Self::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(mut seed: u64) -> impl FnMut(u64) -> u64 {
        move |n| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) % n
        }
    }

    fn assert_same(log: &AppendLog<String>, reference: &[String], what: &str) {
        assert_eq!(log.len(), reference.len(), "{what}: len");
        assert_eq!(log.is_empty(), reference.is_empty(), "{what}: is_empty");
        assert_eq!(log.to_vec(), reference, "{what}: to_vec");
        assert_eq!(
            log.iter().collect::<Vec<_>>(),
            reference.iter().collect::<Vec<_>>(),
            "{what}: iter"
        );
        assert_eq!(log.last(), reference.last(), "{what}: last");
        for i in 0..reference.len() + 2 {
            assert_eq!(log.get(i), reference.get(i), "{what}: get({i})");
        }
        if let Some(last) = reference.len().checked_sub(1) {
            assert_eq!(log[last], reference[last], "{what}: index");
        }
    }

    /// 与 Vec 逐步对照; 中途取的快照之后不随追加而变。
    #[test]
    fn matches_vec_and_snapshots_stay_frozen() {
        let mut next = rng(7);
        let mut log: AppendLog<String> = AppendLog::new();
        let mut reference: Vec<String> = Vec::new();
        let mut snapshots: Vec<(AppendLog<String>, Vec<String>)> = Vec::new();
        assert_same(&log, &reference, "空表");
        for step in 0..(CHUNK * 5 + 7) {
            let value = format!("v{step}");
            log.push(value.clone());
            reference.push(value);
            assert_same(&log, &reference, &format!("第 {step} 步"));
            // 块边界前后必取快照, 其余随机取
            let at_boundary = reference.len() % CHUNK <= 1 || reference.len() % CHUNK == CHUNK - 1;
            if at_boundary || next(5) == 0 {
                snapshots.push((log.clone(), reference.clone()));
            }
            // 随机丢掉一个旧快照: 之后的追加走"独占, 原地追加"的路径
            if !snapshots.is_empty() && next(4) == 0 {
                let victim = next(snapshots.len() as u64) as usize;
                snapshots.swap_remove(victim);
            }
        }
        assert!(snapshots.len() > 10);
        for (i, (snapshot, expected)) in snapshots.iter().enumerate() {
            assert_same(snapshot, expected, &format!("快照 {i}"));
        }
    }

    /// 快照自己也能继续追加, 与原表互不影响。
    #[test]
    fn diverging_clones_are_independent() {
        for prefix in [0, 1, CHUNK - 1, CHUNK, CHUNK + 1, CHUNK * 2 + 3] {
            let mut a: AppendLog<String> = AppendLog::new();
            let mut ref_a = Vec::new();
            for i in 0..prefix {
                a.push(format!("p{i}"));
                ref_a.push(format!("p{i}"));
            }
            let mut b = a.clone();
            let mut ref_b = ref_a.clone();
            for i in 0..CHUNK + 5 {
                a.push(format!("a{i}"));
                ref_a.push(format!("a{i}"));
                b.push(format!("b{i}"));
                ref_b.push(format!("b{i}"));
            }
            assert_same(&a, &ref_a, &format!("prefix={prefix} a"));
            assert_same(&b, &ref_b, &format!("prefix={prefix} b"));
        }
    }

    /// 克隆不复制内容: 写满的块与未满的块都与原表共享。
    #[test]
    fn clone_shares_storage() {
        let log: AppendLog<String> = (0..CHUNK * 3 + 5)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .into();
        let snapshot = log.clone();
        assert!(Arc::ptr_eq(&log.sealed, &snapshot.sealed));
        assert!(Arc::ptr_eq(&log.tail, &snapshot.tail));
        assert_eq!(log.sealed.len(), 3);
        assert_eq!(log.tail.len(), 5);
    }

    /// 有快照在外时追加: 写满的块不复制, 仍与快照共享。
    #[test]
    fn push_with_outstanding_snapshot_keeps_sealed_chunks_shared() {
        let mut log: AppendLog<String> = (0..CHUNK * 2 + 5)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .into();
        let snapshot = log.clone();
        for i in 0..CHUNK {
            log.push(format!("n{i}"));
        }
        assert_eq!(log.sealed.len(), 3);
        for chunk in 0..2 {
            assert!(Arc::ptr_eq(&log.sealed[chunk], &snapshot.sealed[chunk]));
        }
        assert_eq!(snapshot.len(), CHUNK * 2 + 5);
    }

    /// 编码与 Vec / Arc<Vec> 相同(msgpack 与 JSON), 两个方向都能解。
    #[test]
    fn serializes_like_a_vec() {
        for len in [0, 1, CHUNK - 1, CHUNK, CHUNK * 2 + 3] {
            let values: Vec<String> = (0..len).map(|i| format!("v{i}")).collect();
            let log: AppendLog<String> = values.clone().into();
            let shared = Arc::new(values.clone());

            let packed = rmp_serde::to_vec(&log).unwrap();
            assert_eq!(packed, rmp_serde::to_vec(&shared).unwrap(), "len={len}");
            let from_old: AppendLog<String> =
                rmp_serde::from_slice(&rmp_serde::to_vec(&shared).unwrap()).unwrap();
            assert_same(&from_old, &values, &format!("len={len} 旧编码解成新类型"));
            let to_old: Arc<Vec<String>> = rmp_serde::from_slice(&packed).unwrap();
            assert_eq!(*to_old, values, "len={len} 新编码解成旧类型");

            let json = serde_json::to_string(&log).unwrap();
            assert_eq!(json, serde_json::to_string(&values).unwrap());
            let back: AppendLog<String> = serde_json::from_str(&json).unwrap();
            assert_same(&back, &values, &format!("len={len} json"));
        }
    }

    #[test]
    #[should_panic(expected = "下标越界")]
    fn index_out_of_range_panics() {
        let log: AppendLog<String> = vec!["a".to_string()].into();
        let _ = &log[1];
    }
}
