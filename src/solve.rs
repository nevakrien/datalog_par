/*!
 * we are doing a naive solve for now,
 * note however that we can improve things a lot by optimizing what we run over
 * we could also add magic sets althout these require more work
 * it is probably also a good idea to put some extra work to grammar rewrting
 */

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
        self.full.extend(new_delta.into_iter());
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

fn run_iteration(info: &QueryInfo, data: &QueryData) -> AnsSet {
    info.preds
        .par_iter()
        .map(|p| {
            let map = info.kb.producers[p]
                .rules
                .par_iter()
                .flat_map(|r| run_iteration_rule(r, info.kb, data))
                .collect();

            (*p, map)
        })
        .collect()
}

fn run_iteration_rule(rule: &RuleId, info: &KB, data: &QueryData) -> HashSet<Group> {
    todo!()
}
