// /*!
//  * we are doing a naive solve for now,
//  * note however that we can improve things a lot by optimizing what we run over
//  * we could also add magic sets althout these require more work
//  * it is probably also a good idea to put some extra work to grammar rewrting
//  */

// use crate::compile::QueryInfo;
// use crate::compile::SolveAction;
// use crate::parser::{ConstId, KB, PredId};
// use hashbrown::{HashMap, HashSet};
// use rayon::prelude::*;
// use std::cell::UnsafeCell;
// use std::hash::Hash;
// use std::ops::Deref;
// use std::ops::DerefMut;

// ///this is considered a T for safe code but its actualy UnsafeCell
// #[repr(transparent)]
// pub struct Cheat<T>(pub UnsafeCell<T>);
// impl<T> Cheat<T> {
//     pub fn into_inner(self) -> T {
//         self.0.into_inner()
//     }
//     pub fn get(&self) -> *mut T {
//         self.0.get()
//     }
//     pub unsafe fn unsafe_mut(&self) -> &mut T {
//         unsafe { &mut *self.0.get() }
//     }
// }
// impl<T> Deref for Cheat<T> {
//     type Target = T;
//     fn deref(&self) -> &T {
//         unsafe { &*self.0.get() }
//     }
// }
// impl<T> DerefMut for Cheat<T> {
//     fn deref_mut(&mut self) -> &mut T {
//         self.0.get_mut()
//     }
// }

// unsafe impl<T: Sync> Sync for Cheat<T> {}
// unsafe impl<T: Send> Send for Cheat<T> {}

// impl<T> From<T> for Cheat<T> {
//     fn from(t: T) -> Self {
//         Self(t.into())
//     }
// }

// pub type Group = Box<[ConstId]>;
// pub type AnsSet = HashMap<PredId, Cheat<HashSet<Group>>>;

// pub struct QueryData {
//     full: AnsSet,
//     delta: AnsSet,
// }

// impl QueryData {
//     pub fn new(preds: &[PredId]) -> Self {
//         Self {
//             full: preds.iter().map(|p| (*p, HashSet::new().into())).collect(),
//             delta: AnsSet::new(),
//         }
//     }
//     pub fn rotate(&mut self, mut new_delta: AnsSet) {
//         std::mem::swap(&mut self.delta, &mut new_delta);
//         if new_delta.is_empty() {
//             return;
//         }

//         new_delta.into_par_iter().for_each(|(k, mut v)| {
//             //this is safe since every value in new_delta is unique
//             unsafe { self.full.get(&k).unwrap().unsafe_mut().extend(v.drain()) }
//         });
//         // for (k,mut v) in new_delta.into_iter(){
//         //  	//this is safe since every value in new_delta is unique
//         //  	unsafe{
//         //  		self.full.get(&k).unwrap().unsafe_mut().extend(v.drain())
//         //  	}
//         // };
//     }
// }

// pub fn run_to_end(info: &QueryInfo, data: &mut QueryData) {
//     loop {
//         data.rotate(run_iteration(info, data));
//         if data.delta.is_empty() {
//             return;
//         }
//     }
// }

// fn merge_sets<T: Eq + Hash>(iter: impl ParallelIterator<Item = HashSet<T>>) -> HashSet<T> {
//     iter.reduce_with(|mut a, mut b| {
//         if a.len() >= b.len() {
//             a.reserve(b.len());
//             a.extend(b.into_iter());
//             a
//         } else {
//             b.reserve(a.len());
//             b.extend(a.into_iter());
//             b
//         }
//     })
//     .unwrap_or_default()
// }

// fn run_iteration(info: &QueryInfo, data: &QueryData) -> AnsSet {
//     todo!()
//     // info.preds
//     //     .par_iter()
//     //     .filter_map(|p| {
//     //         let map = info.kb.producers[p]
//     //             .rules
//     //             .par_iter()
//     //             .map(|r| run_iteration_rule(r, info.kb, data));

//     //         let table = merge_sets(map);
//     //         if table.is_empty() {
//     //             None
//     //         } else {
//     //             Some((*p, table.into()))
//     //         }
//     //     })
//     //     .collect()

//     // info.plans
//     //     .par_iter()
//     //     .filter_map(|(p, rules)| {
//     //         let map = rules
//     //             .par_iter()
//     //             .map(|r| run_iteration_rule(r, info.kb, data));

//     //         let table = merge_sets(map);
//     //         if table.is_empty() {
//     //             None
//     //         } else {
//     //             Some((*p, table.into()))
//     //         }
//     //     })
//     //     .collect()
// }

// fn run_iteration_rule(rule: &SolveAction, info: &KB, data: &QueryData) -> HashSet<Group> {
//     todo!()
// }

// #[cfg(test)]
// mod tests {
//     // use std::num::NonZero;
//     use super::*;
//     use std::num::NonZeroU32;

//     fn pid(n: u32) -> PredId {
//         PredId(NonZeroU32::new(n).unwrap())
//     }
//     fn cid(n: u32) -> ConstId {
//         ConstId(NonZeroU32::new(n).unwrap())
//     }
//     fn g(xs: &[u32]) -> Group {
//         xs.iter()
//             .map(|&x| cid(x))
//             .collect::<Vec<_>>()
//             .into_boxed_slice()
//     }

//     #[test]
//     fn rayon_wroks() {
//         let a: i32 = [1; 300].par_iter().sum();
//         assert_eq!(a, (0..300).map(|_| { 1 }).sum())
//     }
//     #[test]
//     fn rotate_equals_sequential_union() {
//         let preds: Vec<_> = (1..=16).map(pid).collect();
//         let mut qd = QueryData::new(&preds);

//         // seed delta (will be unioned on first rotate)
//         qd.delta = preds
//             .iter()
//             .map(|&p| {
//                 let mut s = HashSet::new();
//                 for j in 2..4 {
//                     s.insert(g(&[p.0.get(), j]));
//                 }
//                 (p, s.into())
//             })
//             .collect();

//         // build expected by sequential union
//         let mut expected: HashMap<PredId, HashSet<Group>> = HashMap::new();
//         for (&p, set) in &qd.delta {
//             expected.entry(p).or_default().extend(set.iter().cloned());
//         }

//         // rotate with empty "new"
//         qd.rotate(HashMap::new());

//         // snapshot full
//         let mut got: HashMap<PredId, HashSet<Group>> = HashMap::new();
//         for (p, cell) in &qd.full {
//             let set: &HashSet<Group> = &*cell; // safe read
//             got.insert(*p, set.clone());
//         }

//         assert_eq!(got, expected);
//         assert!(qd.delta.is_empty());
//     }

//     // #[test]
//     // #[cfg_attr(miri, ignore)] // don’t run this under Miri
//     // fn stress_test_rotate() {

//     //     // pick some fake preds
//     //     let preds: Vec<PredId> = (1..16).map(|i| PredId(NonZeroU32::new(i).unwrap())).collect();
//     //     let mut qd = QueryData::new(&preds);

//     //     for j in 1..50 {
//     //         // build a new_delta with large random sets
//     //         let mut new_delta: AnsSet = HashMap::new();

//     //         for &p in &preds {
//     //             let mut hs = HashSet::new();

//     //             for i in 1..10000 {
//     //                 let len = i%5;
//     //                 let mut g = Vec::with_capacity(len);
//     //                 for k in 0..len {
//     //                     g.push(ConstId(NonZeroU32::new((1+i*j*k) as u32).unwrap()));
//     //                 }
//     //                 hs.insert(g.into_boxed_slice());
//     //             }

//     //             new_delta.insert(p, hs.into());
//     //         }

//     //         // rotate should merge these safely
//     //         qd.rotate(new_delta);
//     //     }

//     //     // sanity check: all preds have some data
//     //     for (p, set) in &qd.full {
//     //         assert!(!set.is_empty(), "predicate {:?} unexpectedly empty", p);
//     //     }
//     // }
// }
