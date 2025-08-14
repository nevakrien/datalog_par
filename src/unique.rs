use std::hash::Hash;
use std::sync::RwLock;
use hashbrown::HashSet;
use std::ops::Deref;
use std::marker::PhantomData;

///basically just a slice but compared and hashed by pointer
#[derive(Debug,PartialEq,Eq,Hash)]
pub struct ListId<'a,T>{
	addr: usize,
	len: usize,
	_ph:PhantomData<&'a [T]>
}

impl<T> Clone for ListId<'_,T>{

fn clone(&self) -> Self { Self{
	addr:self.addr,
	len:self.len,
	_ph:self._ph,
} }
}
impl<T> Copy for ListId<'_,T> {}

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

impl<'a, T> From<&'a [T]> for  ListId<'a,T>{
fn from(s: &'a [T]) -> Self { Self::new(s)}
}

impl<'a,T> ListId<'a,T>{
	pub fn new(s:&'a [T])->Self{
		Self{
			addr:s.as_ptr().expose_provenance(),
			len:s.len(),
			_ph:PhantomData
		}
	}

	pub fn dangle(self) -> ListId<'static,()>{
		ListId{
			addr:self.addr,
			len:self.len,
			_ph:PhantomData
		}
	}
}

#[derive(Debug)]
pub struct Registery<T>(RwLock<HashSet<Box<[T]>>>);

impl<T> Default for Registery<T>{

fn default() -> Self { Self(RwLock::new(HashSet::new()))}
}

impl<T:Eq+Hash+Clone> Registery<T>{
	pub fn new()->Self{
		Self::default()
	}
	pub fn get_unique<'b>(&self,x:&'b [T]) -> ListId<T>{
		self.alloc(x).into()
	}



	pub fn alloc<'b>(&self,x:&'b [T]) -> &[T]{
		if let Some(temp) = self.0.read().unwrap().get(x){
			return unsafe{
				core::slice::from_raw_parts(temp.as_ptr(),temp.len())
			}
		}

		let mut binding = self.0.write().unwrap();

  		let temp  = binding.get_or_insert_with(x, |x| {
			x.into()
		});


		unsafe{
			core::slice::from_raw_parts(temp.as_ptr(),temp.len())
		}
	}
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_get_single_slice() {
        let registry = Registery::default();
        let data = vec![1, 2, 3, 4, 5];
        
        let result = registry.get_unique(&data);
        assert_eq!(&*result, &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_get_same_slice_returns_same_reference() {
        let registry = Registery::default();
        let data = vec![1, 2, 3];
        
        let result1 = registry.get_unique(&data);
        let result2 = registry.get_unique(&data);
        
        // Should return the same memory location for identical data
        assert_eq!(result1, result2);
        assert_eq!(&*result1, &*result2);
    }

    #[test]
    fn test_get_different_slices() {
        let registry = Registery::default();
        let data1 = vec![1, 2, 3];
        let data2 = vec![4, 5, 6];
        
        let result1 = registry.get_unique(&data1);
        let result2 = registry.get_unique(&data2);
        
        assert_ne!(result1.as_ptr(), result2.as_ptr());
        assert_eq!(&*result1, &[1, 2, 3]);
        assert_eq!(&*result2, &[4, 5, 6]);
    }

    #[test]
    fn test_get_empty_slice() {
        let registry = Registery::default();
        let empty: Vec<i32> = vec![];
        
        let result = registry.alloc(&empty);
        assert_eq!(result, &[] as &[i32]);
    }

    #[test]
    fn test_get_with_strings() {
        let registry = Registery::default();
        let data = vec!["hello".to_string(), "world".to_string()];
        
        let result = registry.alloc(&data);
        assert_eq!(result, &["hello".to_string(), "world".to_string()]);
    }

    #[test]
    fn test_concurrent_access() {
        let registry = Arc::new(Registery::default());
        let mut handles = vec![];

        // Create different data sets outside the threads
        let test_data: Vec<Vec<_>> = (0..10).map(|i| vec![i, i + 1, i + 2]).collect();

        for i in 0..10 {
            let registry_clone = Arc::clone(&registry);
            let data = test_data[i].clone();
            let handle = thread::spawn(move || {
                let result = registry_clone.alloc(&data);
                assert_eq!(result, &[i, i + 1, i + 2]);
                result.as_ptr() as usize // Convert to address for comparison
            });
            handles.push(handle);
        }

        let addrs: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        
        // All different data should have different addresses
        for i in 0..addrs.len() {
            for j in i + 1..addrs.len() {
                assert_ne!(addrs[i], addrs[j]);
            }
        }
    }

    #[test]
    fn test_concurrent_same_data() {
        let registry = Arc::new(Registery::default());
        let mut handles = vec![];

        // Create the shared data outside the threads
        let shared_data = vec![1, 2, 3];

        // Multiple threads accessing the same data
        for _ in 0..5 {
            let registry_clone = Arc::clone(&registry);
            let data = shared_data.clone();
            let handle = thread::spawn(move || {
                let result = registry_clone.get_unique(&data);
                assert_eq!(&*result, &[1, 2, 3]);
                result.dangle()
            });
            handles.push(handle);
        }

        let addrs: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        
        // All should point to the same memory location
        for addr in &addrs[1..] {
            assert_eq!(addrs[0], *addr);
        }
    }

    #[test]
    fn test_drop_cleanup() {
        // This test ensures the Drop implementation doesn't panic
        let registry = Registery::default();
        let data1 = vec![1, 2, 3];
        let data2 = vec![4, 5, 6];
        
        let _result1 = registry.get_unique(&data1);
        let _result2 = registry.get_unique(&data2);
        
        // Drop should clean up without panicking
        drop(registry);
    }

    #[test]
    fn test_custom_struct() {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[derive(Default)]
		struct TestStruct {
            id: u32,
            name: String,
        }

        let registry = Registery::default();
        let data = vec![
            TestStruct { id: 1, name: "Alice".to_string() },
            TestStruct { id: 2, name: "Bob".to_string() },
        ];
        
        let result = registry.get_unique(&data);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, 1);
        assert_eq!(result[0].name, "Alice");
    }

    #[test]
    fn test_repeated_identical_calls() {
        let registry = Registery::default();
        let data = vec![42, 24, 12];
        
        // Call get multiple times with the same data
        let results: Vec<_> = (0..100)
            .map(|_| registry.get_unique(&data))
            .collect();
        
        // All should be equal and point to same memory
        for result in &results[1..] {
            assert_eq!(results[0], *result);
        }
    }

    // Test to verify memory safety with lifetimes
    #[test]
    fn test_lifetime_safety() {
        let registry = Registery::default();
        let addr = {
            let temp_data = vec![1, 2, 3, 4];
            let result = registry.get_unique(&temp_data);
            result.as_ptr() as usize
        }; // temp_data goes out of scope here
        
        // The returned slice should still be valid because it's stored in the registry
        let data2 = vec![1, 2, 3, 4];
        let result2 = registry.get_unique(&data2);
        assert_eq!(addr, result2.as_ptr() as usize);
    }
}