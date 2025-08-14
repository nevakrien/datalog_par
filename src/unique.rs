use std::marker::PhantomData;
use std::cell::UnsafeCell;
use std::sync::RwLockReadGuard;
use std::slice;
use std::hash::Hash;
use std::sync::RwLock;
use std::collections::HashSet;

// ==== registery
struct RegisteryInner<'a, T> {
	//we inline the arena implementation here
	//this is mostly for not needing to mess with sync/send
    arena_cur:Vec<T>,
    arena_store:  Vec<Vec<T>>,
    exists: HashSet<&'a [T]>,
}


impl<'a, T> RegisteryInner<'a, T> {
	unsafe fn clear(&mut self){
		self.exists.clear();
		self.arena_cur.clear();
		self.arena_store.clear();
	}

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
    	if self.arena_cur.capacity()-start < s.len(){
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
    			return slice::from_raw_parts(self.arena_cur.as_ptr(),s.len());
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


pub struct Registery<'a, T>(RwLock<RwLock<RegisteryInner<'a,T>>>);
impl<'a,T> Registery<'a,T>{
	pub fn new()->Self{Self(RwLock::new(RegisteryInner::new().into()))}
	pub fn borrow_thread<'me>(&'me self)-> RegisteryRef<'me,'a,T>{
		RegisteryRef(self.0.read().unwrap(),PhantomData)
	}

}

impl<T> Drop for Registery<'_,T>{
fn drop(&mut self) { 	
	//make sure all borrows are properly dead
	let _d = self.0.write().unwrap();
 }
}

pub struct RegisteryRef<'me, 'a, T>(RwLockReadGuard<'me ,RwLock<RegisteryInner<'a,T>>>,PhantomData<&'a UnsafeCell<T>>);


impl<'a,T:Hash+Eq> RegisteryRef<'_,'a,T>{
	pub fn try_get(&self,s:&[T])->Option<&'a [T]>{
		self.0.read().unwrap().exists.get(s).map(|v| &**v)
	}
}

impl<'a,T:Hash+Eq+Clone> RegisteryRef<'_,'a,T>{
	pub fn get(&self,s:&[T])->&'a [T]{
		//first try the fast path no relocking
		if let Some(ans) = self.try_get(s) {
			return ans;
		}

		self.get_write(s)
	}

	pub fn get_write(&self,s:&[T])->&'a [T]{
		let mut p = self.0.write().unwrap();
		if let Some(ans) = p.exists.get(s).map(|v| &**v){
			return ans;
		}

		let ans = unsafe{p.alloc(s)};
		p.exists.insert(ans);
		ans
	}
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, atomic::{AtomicUsize, Ordering}};
    use std::thread;
    use std::collections::HashSet;
    use std::time::Duration;

    // Test with a type that has a destructor to ensure proper memory management
    #[derive(Debug, Clone)]
    struct DropTracker {
        id: usize,
        counter: std::sync::Arc<AtomicUsize>,
    }

    impl DropTracker {
        fn new(id: usize, counter: std::sync::Arc<AtomicUsize>) -> Self {
            counter.fetch_add(1, Ordering::SeqCst);
            Self { id, counter }
        }
    }

    impl PartialEq for DropTracker {
        fn eq(&self, other: &Self) -> bool {
            self.id == other.id
        }
    }

    impl Eq for DropTracker {}

    impl std::hash::Hash for DropTracker {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            self.id.hash(state);
        }
    }

    impl Drop for DropTracker {
        fn drop(&mut self) {
            self.counter.fetch_sub(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn test_registery_basic_functionality() {
        let reg =Registery::new();
        let reg = reg.borrow_thread();
        
        // Test with simple integers
        let slice1 = vec![1, 2, 3, 4, 5];
        let slice2 = vec![1, 2, 3, 4, 5];
        let slice3 = vec![6, 7, 8, 9, 10];
        
        let result1 = reg.get(&slice1);
        let result2 = reg.get(&slice2);
        let result3 = reg.get(&slice3);
        
        // Same content should return same pointer
        assert_eq!(result1.as_ptr(), result2.as_ptr());
        assert_eq!(result1, result2);
        
        // Different content should return different pointer
        assert_ne!(result1.as_ptr(), result3.as_ptr());
        assert_ne!(result1, result3);
    }

    #[test]
    fn test_registery_different_sizes() {
        let reg =Registery::new();
        let reg = reg.borrow_thread();
        
        // Test various sizes to trigger different allocation strategies
        let sizes = vec![
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
        ];
        
        let mut results = Vec::new();
        
        for size in sizes {
            let data: Vec<i32> = (0..size).collect();
            let result = reg.get(&data);
            assert_eq!(result.len(), size as usize);
            assert_eq!(result, &data[..]);
            results.push((size, result));
        }
        
        // Verify same data returns same pointers
        for (size, expected_result) in &results {
            let data: Vec<i32> = (0..*size).collect();
            let result = reg.get(&data);
            assert_eq!(result.as_ptr(), expected_result.as_ptr());
        }
    }

    #[test]
    fn test_registery_empty_slices() {
        let reg =Registery::new();
        let reg = reg.borrow_thread();
        
        let empty1: Vec<i32> = vec![];
        let empty2: Vec<i32> = vec![];
        
        let result1 = reg.get(&empty1);
        let result2 = reg.get(&empty2);
        
        assert_eq!(result1.len(), 0);
        assert_eq!(result2.len(), 0);
        assert_eq!(result1.as_ptr(), result2.as_ptr());
    }

    #[test]
    fn test_registery_with_destructors() {
        thread::scope(|s| {
            let drop_counter = std::sync::Arc::new(AtomicUsize::new(0));
            
            let reg =Registery::new();
       		let reg = reg.borrow_thread();
            
            {
                let data = vec![
                    DropTracker::new(1, drop_counter.clone()),
                    DropTracker::new(2, drop_counter.clone()),
                    DropTracker::new(3, drop_counter.clone()),
                ];
                
                // Should have 3 live objects
                assert_eq!(drop_counter.load(Ordering::SeqCst), 3);
                
                let result1 = reg.get(&data);
                let result2 = reg.get(&data);
                
                // Should still have original 3 + 3 cloned into registry = 6
                assert_eq!(drop_counter.load(Ordering::SeqCst), 6);
                assert_eq!(result1.as_ptr(), result2.as_ptr());
                
                // Original data goes out of scope, should drop 3
            }
            
            // Should have 3 remaining (the ones stored in registry)
            assert_eq!(drop_counter.load(Ordering::SeqCst), 3);
        });
    }

    #[test]
    fn test_registery_multithreaded_stress() {
        let reg =Registery::new();
       	let reg = &reg;

        thread::scope(|s| {
            
            let reg = &reg;

            let num_threads = 8;
            let iterations_per_thread = 100;
            
            let handles: Vec<_> = (0..num_threads)
                .map(|thread_id| {
                    s.spawn( move || {
                        let reg = reg.borrow_thread();
                        let mut local_results = Vec::new();
                        
                        for i in 0..iterations_per_thread {
                            // Create different patterns of data
                            let data = match i % 5 {
                                0 => vec![thread_id, i],                    // Small, unique
                                1 => vec![thread_id; 10],                  // Medium, repeated value
                                2 => (0..100).map(|x| x + thread_id).collect(), // Large, unique pattern
                                3 => vec![42; 1000],                       // Very large, same across threads
                                _ => vec![thread_id, i % 10],              // Small, some overlap
                            };
                            
                            let result = reg.get(&data);
                            assert_eq!(result, &data[..]);
                            local_results.push((data, result));
                        }
                        
                        // Verify consistency within thread
                        for (original_data, expected_result) in &local_results {
                            let new_result = reg.get(original_data);
                            assert_eq!(new_result.as_ptr(), expected_result.as_ptr());
                        }
                        
                        local_results.len()
                    })
                })
                .collect();
            
            let total_operations: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
            assert_eq!(total_operations, num_threads * iterations_per_thread);
        });
    }

    #[test]
    fn test_registery_try_get() {
        let reg =Registery::new();
       	let reg = reg.borrow_thread();
        
        let data = vec![1, 2, 3];
        
        // Should return None before insertion
        assert!(reg.try_get(&data).is_none());
        
        // Insert the data
        let result = reg.get(&data);
        
        // Now try_get should return Some
        let try_result = reg.try_get(&data).unwrap();
        assert_eq!(try_result.as_ptr(), result.as_ptr());
        assert_eq!(try_result, &data[..]);
    }

    #[test]
    fn test_registery_mixed_types_stress() {
        let reg =Registery::new();
       	let reg = &reg;

        thread::scope(|s| {
        	let reg = &reg;
            let num_threads = 4;
            
            let handles: Vec<_> = (0..num_threads)
                .map(|thread_id| {
                    s.spawn(move || {
                        let reg = reg.borrow_thread();
                        let mut results = HashSet::new();
                        
                        // Create various string patterns
                        for i in 0..50 {
                            let patterns = [
                                format!("thread_{}_item_{}", thread_id, i),
                                format!("common_pattern_{}", i % 10),
                                "shared_string".to_string(),
                                format!("{}", i),
                            ];
                            
                            for pattern in &patterns {
                                let chars: Vec<char> = pattern.chars().collect();
                                let result = reg.get(&chars);
                                assert_eq!(result, &chars[..]);
                                results.insert(result.as_ptr() as usize);
                            }
                        }
                        
                        results.len()
                    })
                })
                .collect();
            
            let unique_pointers: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
            println!("Total unique pointers across all threads: {}", unique_pointers);
        });
    }

    #[test]
    fn test_registery_large_allocations() {
        thread::scope(|s| {
            let reg =Registery::new();
       		let reg = reg.borrow_thread();
            
            // Test very large allocations that exceed normal arena sizes
            let large_data: Vec<i32> = (0..10000).collect();
            let huge_data: Vec<i32> = (0..100000).collect();
            
            let result1 = reg.get(&large_data);
            let result2 = reg.get(&large_data);
            let result3 = reg.get(&huge_data);
            let result4 = reg.get(&huge_data);
            
            assert_eq!(result1.as_ptr(), result2.as_ptr());
            assert_eq!(result3.as_ptr(), result4.as_ptr());
            assert_ne!(result1.as_ptr(), result3.as_ptr());
            
            assert_eq!(result1, &large_data[..]);
            assert_eq!(result3, &huge_data[..]);
        });
    }

    // Helper for generating random data
    mod fastrand {
        use std::cell::Cell;
        
        thread_local! {
            static RNG: Cell<u64> = Cell::new(1);
        }
        
        pub fn u64(range_start: u64, range_end: u64) -> u64 {
            RNG.with(|rng| {
                let mut x = rng.get();
                if x == 0 { x = 1; }
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                rng.set(x);
                range_start + (x % (range_end - range_start))
            })
        }
    }
}