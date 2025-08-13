use std::slice;
use std::hash::Hash;
use std::sync::RwLock;
use std::collections::HashSet;
// use std::hash::{Hash, Hasher};
// use std::marker::PhantomData;
// use std::ops::Deref;
// use std::ptr;
// use std::slice;
// use std::str;

// // ========== UniqueSlice ==========
// struct UniqueSlice<'a, T> {
//     ptr: *const T,
//     len: usize,
//     _ph: PhantomData<&'a [T]>,
// }

// // ---- Manual trait impls (no T bounds) ----
// impl<'a, T> Clone for UniqueSlice<'a, T> {
//     #[inline]
//     fn clone(&self) -> Self { *self }
// }

// impl<'a, T> Copy for UniqueSlice<'a, T> {}

// impl<'a, T> PartialEq for UniqueSlice<'a, T> {
//     #[inline]
//     fn eq(&self, other: &Self) -> bool {
//         self.ptr == other.ptr && self.len == other.len
//     }
// }

// impl<'a, T> Eq for UniqueSlice<'a, T> {}

// impl<'a, T> Hash for UniqueSlice<'a, T> {
//     #[inline]
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         self.ptr.hash(state);
//         self.len.hash(state);
//     }
// }

// // ---- PartialEq with slices ----
// impl<'a, T: PartialEq> PartialEq<[T]> for UniqueSlice<'a, T> {
//     #[inline]
//     fn eq(&self, other: &[T]) -> bool {
//         let this: &[T] = (*self).into();
//         this == other
//     }
// }
// impl<'a, T: PartialEq> PartialEq<&'a [T]> for UniqueSlice<'a, T> {
//     #[inline]
//     fn eq(&self, other: &&'a [T]) -> bool {
//         let this: &[T] = (*self).into();
//         this == *other
//     }
// }

// // ---- Conversions ----
// impl<'a, T> From<&'a [T]> for UniqueSlice<'a, T> {
//     fn from(s: &'a [T]) -> Self {
//         let ptr = if s.is_empty() {
//             ptr::null()
//         } else {
//             s.as_ptr()
//         };
//         Self {
//             ptr,
//             len: s.len(),
//             _ph: PhantomData,
//         }
//     }
// }

// impl<'a, T> From<UniqueSlice<'a, T>> for &'a [T] {
//     fn from(s: UniqueSlice<'a, T>) -> Self {
//         let p = if s.ptr.is_null() {
//             ptr::NonNull::dangling().as_ptr()
//         } else {
//             s.ptr
//         };
//         unsafe { slice::from_raw_parts(p, s.len) }
//     }
// }

// // ---- Deref ----
// impl<T> Deref for UniqueSlice<'_, T> {
//     type Target = [T];
//     fn deref(&self) -> &[T] {
//         let p = if self.ptr.is_null() {
//             ptr::NonNull::dangling().as_ptr()
//         } else {
//             self.ptr
//         };
//         unsafe { slice::from_raw_parts(p, self.len) }
//     }
// }

// // ---- Send/Sync ----
// unsafe impl<'a, T> Send for UniqueSlice<'a, T> where &'a T : Send {}
// unsafe impl<'a, T> Sync for UniqueSlice<'a, T> where &'a T : Sync {}


// // ========== UniqueStr ==========
// pub(crate) struct UniqueStr<'a>(UniqueSlice<'a, u8>);

// // ---- Manual trait impls ----
// impl<'a> Clone for UniqueStr<'a> {
//     #[inline]
//     fn clone(&self) -> Self { *self }
// }

// impl<'a> Copy for UniqueStr<'a> {}

// impl<'a> PartialEq for UniqueStr<'a> {
//     #[inline]
//     fn eq(&self, other: &Self) -> bool {
//         self.0 == other.0
//     }
// }

// impl<'a> Eq for UniqueStr<'a> {}

// impl<'a> Hash for UniqueStr<'a> {
//     #[inline]
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         self.0.hash(state)
//     }
// }

// // ---- PartialEq with str ----
// impl<'a> PartialEq<str> for UniqueStr<'a> {
//     #[inline]
//     fn eq(&self, other: &str) -> bool {
//         let this: &str = (*self).into();
//         this == other
//     }
// }
// impl<'a> PartialEq<&'a str> for UniqueStr<'a> {
//     #[inline]
//     fn eq(&self, other: &&'a str) -> bool {
//         let this: &str = (*self).into();
//         this == *other
//     }
// }

// // ---- Conversions ----
// impl<'a> From<&'a str> for UniqueStr<'a> {
//     fn from(s: &'a str) -> Self {
//         UniqueStr(UniqueSlice::from(s.as_bytes()))
//     }
// }

// impl<'a> From<UniqueStr<'a>> for &'a str {
//     fn from(us: UniqueStr<'a>) -> Self {
//         let s: &'a [u8] = us.0.into();
//         unsafe { str::from_utf8_unchecked(s) }
//     }
// }

// // ---- Deref ----
// impl Deref for UniqueStr<'_> {
//     type Target = str;
//     fn deref(&self) -> &str {
//         let s: &[u8] = self.0.deref();
//         unsafe { str::from_utf8_unchecked(s) }
//     }
// }

// ==== registery
struct RegisteryInner<'a, T> {
	//we inline the arena implementation here
	//this is mostly for not needing to mess with sync/send
    arena_cur:Vec<T>,
    arena_store:  Vec<Vec<T>>,
    exists: HashSet<&'a [T]>,
}


impl<'a, T> RegisteryInner<'a, T> {
    pub fn new() -> Self {
        Self {
            arena_cur:Vec::with_capacity(Self::get_base_size()),
    		arena_store:Vec::new(),
            exists: HashSet::new(),
        }
    }

    const fn get_base_size() -> usize{
    	let a = 4096/size_of::<T>();
    	if a > 8 {
    		a
    	}else{
    		8
    	}
    }

    const fn get_small_size() -> usize{
    	let a = 4096/size_of::<T>();
    	if a > 4 {
    		a
    	}else{
    		4
    	}
    }

    unsafe fn alloc(&mut self,s:&[T])->&'a [T] where T:Clone{
    	let start = self.arena_cur.len();
    	if self.arena_cur.capacity()-start <= s.len(){
    		self.arena_cur.extend(s.iter().cloned());
    		unsafe{
    			return slice::from_raw_parts(self.arena_cur.as_ptr().add(start),s.len());
    		}
    	}

    	//store the too full current in
    	{
    		let mut new_cur = Vec::with_capacity(Self::get_base_size());
    		std::mem::swap(&mut self.arena_cur,&mut new_cur);
    		self.arena_store.push(new_cur);
    	}

    	//big enough to exist with another allocation
    	if s.len() <= Self::get_small_size() {
    		self.arena_cur.extend(s.iter().cloned());
    		unsafe{
    			return slice::from_raw_parts(self.arena_cur.as_ptr().add(start),s.len());
    		}
    	}

    	//too big to do anything meaningful with just make ita seprate allocation
    	let single_use : Vec<T> = s.into();
    	let ans = unsafe{
    		slice::from_raw_parts(single_use.as_ptr(),s.len())
    	};
    	self.arena_store.push(single_use);
    	ans

    }
}


pub struct Registery<'a, T>(RwLock<RegisteryInner<'a,T>>);
impl<T> Registery<'_,T>{
	pub fn new()->Self{Self(RwLock::new(RegisteryInner::new()))}

}

impl<'a,T:Hash+Eq> Registery<'a,T>{
	pub fn try_get(&'a self,s:&[T])->Option<&'a [T]>{
		self.0.read().unwrap().exists.get(s).map(|v| &**v)
	}
}

impl<'a,T:Hash+Eq+Clone> Registery<'a,T>{
	pub fn get(&'a self,s:&[T])->&'a [T]{
		//first try the fast path no relocking
		if let Some(ans) = self.try_get(s) {
			return ans;
		}

		self.get_write(s)
	}

	pub fn get_write(&'a self,s:&[T])->&'a [T]{
		let mut p = self.0.write().unwrap();
		if let Some(ans) = p.exists.get(s).map(|v| &**v){
			return ans;
		}

		let ans = unsafe{p.alloc(s)};
		p.exists.insert(ans);
		ans
	}
}