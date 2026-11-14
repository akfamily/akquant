//! 进程内唯一的修订号: 给"内容没变就复用上次结果"的缓存当键。

use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(1);

/// 一份可变状态的修订号。每次 [`Revision::bump`] 换成一个进程内从未用过的值, 所以
/// 两个修订号相等就说明是同一份状态、期间没被改过: 新建或反序列化出来的状态拿到
/// 的是新值(`Default`), 克隆出来的副本与原件内容相同、修订号也相同, 各自再改动时
/// 又各取新值。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Revision(u64);

impl Revision {
    pub fn fresh() -> Self {
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }

    /// 状态被改动后调用。
    pub fn bump(&mut self) {
        *self = Self::fresh();
    }
}

impl Default for Revision {
    fn default() -> Self {
        Self::fresh()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revisions_are_unique_until_cloned() {
        let a = Revision::default();
        let mut b = Revision::default();
        assert_ne!(a, b);
        let copy = b;
        assert_eq!(copy, b);
        b.bump();
        assert_ne!(copy, b);
        assert_ne!(a, b);
    }
}
