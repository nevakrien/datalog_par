use std::cell::UnsafeCell;
use typed_arena::Arena;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::Deref;
use std::ptr;
use std::slice;
use std::str;

// ========== UniqueSlice ==========
pub struct UniqueSlice<'a, T> {
    ptr: *const T,
    len: usize,
    _ph: PhantomData<&'a [T]>,
}

// ---- Manual trait impls (no T bounds) ----
impl<'a, T> Clone for UniqueSlice<'a, T> {
    #[inline]
    fn clone(&self) -> Self { *self }
}

impl<'a, T> Copy for UniqueSlice<'a, T> {}

impl<'a, T> PartialEq for UniqueSlice<'a, T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr && self.len == other.len
    }
}

impl<'a, T> Eq for UniqueSlice<'a, T> {}

impl<'a, T> Hash for UniqueSlice<'a, T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ptr.hash(state);
        self.len.hash(state);
    }
}

// ---- PartialEq with slices ----
impl<'a, T: PartialEq> PartialEq<[T]> for UniqueSlice<'a, T> {
    #[inline]
    fn eq(&self, other: &[T]) -> bool {
        let this: &[T] = (*self).into();
        this == other
    }
}
impl<'a, T: PartialEq> PartialEq<&'a [T]> for UniqueSlice<'a, T> {
    #[inline]
    fn eq(&self, other: &&'a [T]) -> bool {
        let this: &[T] = (*self).into();
        this == *other
    }
}

// ---- Conversions ----
impl<'a, T> From<&'a [T]> for UniqueSlice<'a, T> {
    fn from(s: &'a [T]) -> Self {
        let ptr = if s.is_empty() {
            ptr::null()
        } else {
            s.as_ptr()
        };
        Self {
            ptr,
            len: s.len(),
            _ph: PhantomData,
        }
    }
}

impl<'a, T> From<UniqueSlice<'a, T>> for &'a [T] {
    fn from(s: UniqueSlice<'a, T>) -> Self {
        let p = if s.ptr.is_null() {
            ptr::NonNull::dangling().as_ptr()
        } else {
            s.ptr
        };
        unsafe { slice::from_raw_parts(p, s.len) }
    }
}

// ---- Deref ----
impl<T> Deref for UniqueSlice<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        let p = if self.ptr.is_null() {
            ptr::NonNull::dangling().as_ptr()
        } else {
            self.ptr
        };
        unsafe { slice::from_raw_parts(p, self.len) }
    }
}

// ---- Send/Sync ----
unsafe impl<'a, T: Sync> Send for UniqueSlice<'a, T> {}
unsafe impl<'a, T: Sync> Sync for UniqueSlice<'a, T> {}


// ========== UniqueStr ==========

#[repr(transparent)]
pub struct UniqueStr<'a>(UniqueSlice<'a, u8>);

// ---- Manual trait impls ----
impl<'a> Clone for UniqueStr<'a> {
    #[inline]
    fn clone(&self) -> Self { *self }
}

impl<'a> Copy for UniqueStr<'a> {}

impl<'a> PartialEq for UniqueStr<'a> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<'a> Eq for UniqueStr<'a> {}

impl<'a> Hash for UniqueStr<'a> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

// ---- PartialEq with str ----
impl<'a> PartialEq<str> for UniqueStr<'a> {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        let this: &str = (*self).into();
        this == other
    }
}
impl<'a> PartialEq<&'a str> for UniqueStr<'a> {
    #[inline]
    fn eq(&self, other: &&'a str) -> bool {
        let this: &str = (*self).into();
        this == *other
    }
}

// ---- Conversions ----
impl<'a> From<&'a str> for UniqueStr<'a> {
    fn from(s: &'a str) -> Self {
        UniqueStr(UniqueSlice::from(s.as_bytes()))
    }
}

impl<'a> From<UniqueStr<'a>> for &'a str {
    fn from(us: UniqueStr<'a>) -> Self {
        let s: &'a [u8] = us.0.into();
        unsafe { str::from_utf8_unchecked(s) }
    }
}

// ---- Deref ----
impl Deref for UniqueStr<'_> {
    type Target = str;
    fn deref(&self) -> &str {
        let s: &[u8] = self.0.deref();
        unsafe { str::from_utf8_unchecked(s) }
    }
}

// ==== registery
pub struct Registery<'a, T> {
    arena: UnsafeCell<Arena<T>>,
    exists: HashSet<&'a [T]>,
}


impl<'a,T: Clone + Eq + Hash> Registery<'a, T> {
    pub fn new() -> Self {
        Self {
            arena: UnsafeCell::new(Arena::new()),
            exists: HashSet::new(),
        }
    }

    pub fn get(&mut self, s: &[T]) -> UniqueSlice<'a, T> {
        if let Some(ans) = self.exists.get(s) {
            return (*ans).into();
        }
        // Safety: &mut self ensures exclusive access to arena
        let arena = unsafe { &mut *self.arena.get() };
        let ans: &'a [T] = arena.alloc_extend(s.iter().cloned());
        let uniq = ans.into();
        self.exists.insert(ans);
        uniq
    }
}

// ======== StrRegistery ========

pub struct StrRegistery<'a> {
    arena: UnsafeCell<Arena<u8>>,
    exists: HashSet<&'a str>,
}


impl<'a> StrRegistery<'a> {
    pub fn new() -> Self {
        Self {
            arena: UnsafeCell::new(Arena::new()),
            exists: HashSet::new(),
        }
    }

    pub fn get(&mut self, s: &str) -> UniqueStr<'a> {
        if let Some(ans) = self.exists.get(s) {
            return (*ans).into();
        }
        // Safety: &mut self ensures exclusive access to arena
        let arena = unsafe { &mut *self.arena.get() };
        let bytes = arena.alloc_extend(s.as_bytes().iter().copied());
        let arena_str = unsafe { std::str::from_utf8_unchecked(bytes) };
        let uniq = UniqueStr::from(arena_str);
        self.exists.insert(arena_str);
        uniq
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::thread;
    

    #[test]
    fn concurrent_str_registery_usage() {
        let registry = Arc::new(Mutex::new(StrRegistery::new()));

        let s1 = "hello";
        let s2 = "world";

        let mut handles = Vec::new();

        for _ in 0..2 {
            let registry = Arc::clone(&registry);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let _u1 = registry.lock().unwrap().get(s1);
                    let _u2 = registry.lock().unwrap().get(s2);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }
}
