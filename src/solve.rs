use std::ops::Deref;
use std::marker::PhantomData;

///basically just a slice but compared and hashed by pointer
#[derive(Debug,Clone,PartialEq,Copy,Eq,Hash)]
pub struct ListId<'a,T>{
	addr: usize,
	len: usize,
	_ph:PhantomData<&'a [T]>
}

impl<T> Deref for ListId<'_,T>{

type Target = [T];
fn deref(&self) -> &[T] { unsafe{
	core::slice::from_raw_parts(self.addr as *const T,self.len)
} }
}

impl<'a, T> From<ListId<'a,T>> for &'a [T]{
fn from(l: ListId<'a, T>) -> Self { unsafe{
	core::slice::from_raw_parts(l.addr as *const T,l.len)
} }
}

impl<'a,T> ListId<'a,T>{
	pub fn new(s:&'a [T])->Self{
		Self{
			addr:s.as_ptr().expose_provenance(),
			len:s.len(),
			_ph:PhantomData
		}
	}
}