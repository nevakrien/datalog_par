/*!
 * we are doing a naive solve for now,
 * note however that we can improve things a lot by optimizing what we run over
 * we could also add magic sets althout these require more work
 * it is probably also a good idea to put some extra work to grammar rewrting
 */

use std::collections::hash_map;
use std::hash::Hash;
use crate::compile::RuleId;
use crate::compile::{AtomId, ConstId, KB, PredId};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

pub struct QueryInfo<'kb> {
    pub(crate) query: AtomId,
    pub(crate) preds: Box<[PredId]>,
    pub(crate) kb: &'kb KB,
}

impl<'a> QueryInfo<'a> {
    pub fn new(query: AtomId, kb: &'a KB) -> Self {
        let preds = kb.trace_pred(query.pred).into_iter().collect();
        Self { query, preds, kb }
    }
}

pub type Group = Box<[ConstId]>;
pub type AnsSet = HashMap<PredId, HashSet<Group>>;

pub struct QueryData {
    full: AnsSet,
    delta: AnsSet,
}

impl QueryData {
    pub fn rotate(&mut self, mut new_delta: AnsSet) {
        std::mem::swap(&mut self.delta, &mut new_delta);
        for (k,mut v) in new_delta.into_iter(){
        	match self.full.entry(k) {
        	   hash_map::Entry::Occupied(mut o)=>o.get_mut().extend(v.drain()),
        	   hash_map::Entry::Vacant(x) => {x.insert(v);},
        	}
        }
        
    }
}

pub fn run_to_end(info: &QueryInfo, data: &mut QueryData) {
    loop {
        data.rotate(run_iteration(info, data));
        if data.delta.is_empty() {
            return;
        }
    }
}

fn merge_sets<T: Eq + Hash>(iter: impl ParallelIterator<Item = HashSet<T>>) -> HashSet<T> {
    iter.reduce_with(|mut a, mut b| {
        if a.len() >= b.len() {
            a.reserve(b.len());
            a.extend(b.into_iter());
            a
        } else {
            b.reserve(a.len());
            b.extend(a.into_iter());
            b
        }
    }).unwrap_or_default()
}


fn run_iteration(info: &QueryInfo, data: &QueryData) -> AnsSet {
    info.preds
        .par_iter()
        .filter_map(|p| {
            let map = info.kb.producers[p]
                .rules
                .par_iter()
                .map(|r| run_iteration_rule(r, info.kb, data));

            let table = merge_sets(map);
            if table.is_empty(){
            	None
            }else{
            	Some((*p, table))
            }       
        })
        .collect()
}

fn run_iteration_rule(rule: &RuleId, info: &KB, data: &QueryData) -> HashSet<Group> {
     todo!()
}
