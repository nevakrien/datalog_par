//----- Core solve actions-----------
use crate::parser::{PredId,ConstId,AtomId,KB};

use hashbrown::HashSet;

use hashbrown::HashMap;

pub type ValPlan = Box<[Option<ConstId>]>;

/// full info for magic set lookup
#[derive(Debug,Clone,PartialEq,Eq,Hash)]
pub struct MagicKey {
    pub pred:PredId,
    pub vars: Box<[Option<ConstId>]>
}

#[derive(Debug,Clone)]
pub struct SolveAction {
    pub (crate) magic_template:MagicKey,
    pub (crate) magic_insert: Box<[(usize,usize)]>,//found -> template
    pub (crate) reduce_indecies: Box<[usize]>,
}

impl SolveAction {
    pub fn apply(&self,start: &HashSet<ValPlan>,magic:&HashMap<MagicKey,HashSet<Box<[ConstId]>>>) -> HashSet<ValPlan>{
        let mut ans = HashSet::new();

        for x in start.iter() {
            let mut key = self.magic_template.clone();
            for (i,j) in &self.magic_insert{
                key.vars[*j] = x[*i];
            }

            for item in &magic[&key]{
                let mut new = x.clone();
                for (i,v) in self.reduce_indecies.iter().zip(item.iter()){
                    new[*i] = Some(*v);
                }

                ans.insert(new);
            }

        }

        ans
    }
}

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