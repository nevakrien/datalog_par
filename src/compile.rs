//----- Core solve actions-----------
use rayon::iter::Either;
use rayon::prelude::*;
use crate::parser::{PredId,ConstId,AtomId,KB};

use hashbrown::HashSet;

use hashbrown::HashMap;

pub type ValPlan = Box<[Option<ConstId>]>;
pub type ValPlanRef<'a> = &'a [Option<ConstId>];

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

    pub fn apply_par_owned(&self,start: impl ParallelIterator<Item=ValPlan>,magic:&HashMap<MagicKey,HashSet<Box<[ConstId]>>>) -> impl ParallelIterator<Item=ValPlan>{
        start.flat_map(move |x| {
            self.run_par_item_owned(x,magic)
        })

    }
    pub fn run_par_item_owned(&self,x: ValPlan,magic:&HashMap<MagicKey,HashSet<Box<[ConstId]>>>) -> impl ParallelIterator<Item=ValPlan>{
        let mut key = self.magic_template.clone();
        for (i,j) in &self.magic_insert{
            key.vars[*j] = x[*i];
        }

        magic[&key].par_iter().map(move |item|{
            let mut new : ValPlan = x.clone();
            for (i,v) in self.reduce_indecies.iter().zip(item.iter()){
                new[*i] = Some(*v);
            }

            new
        })
    }

    pub fn apply_par_ref<'a>(&self,start: impl ParallelIterator<Item=ValPlanRef<'a>>,magic:&HashMap<MagicKey,HashSet<Box<[ConstId]>>>) -> impl ParallelIterator<Item=ValPlan>{
        start.flat_map(|x| {
            self.run_par_item(x,magic)
        })

    }

    pub fn run_par_item<'a>(&self,x: ValPlanRef<'a>,magic:&HashMap<MagicKey,HashSet<Box<[ConstId]>>>) -> impl ParallelIterator<Item=ValPlan>{
        let mut key = self.magic_template.clone();
        for (i,j) in &self.magic_insert{
            key.vars[*j] = x[*i];
        }

        magic[&key].par_iter().map(|item|{
            let mut new : ValPlan = x.into();
            for (i,v) in self.reduce_indecies.iter().zip(item.iter()){
                new[*i] = Some(*v);
            }

            new
        })
    }
}

pub struct SolveRule {
    actions:Box<[SolveAction]>,
    gather:Box<[usize]>//made -> plan
}

impl SolveRule{
    pub fn apply(&self,start: &HashSet<ValPlan>,magic:&HashMap<MagicKey,HashSet<Box<[ConstId]>>>) -> HashSet<Box<[ConstId]>>{
        let mut work = self.actions[0].apply(start,magic);
        for a in &self.actions[1..]{
            work = a.apply(&work,magic);
        }
        work.into_iter().map(|x|{
            self.gather.iter().map(|i|{x[*i].unwrap()})
            .collect()
        }).collect()
    }

    pub fn par_apply<'a>(&self,start: impl ParallelIterator<Item=ValPlanRef<'a>>,magic:&HashMap<MagicKey,HashSet<Box<[ConstId]>>>) -> impl ParallelIterator<Item=Box<[ConstId]>>{
        //we would of liked to just make 1 giant par iter
        //rust cant really compile that...
        //there is also an argument that doing these vec stores are fine
        
        //there are no dups
        let mut work: Vec<ValPlan> = self.actions[0].apply_par_ref(start,magic).collect();
        let mut store = Vec::with_capacity(work.len());//get some memory for us
        
        for a in &self.actions[1..]{
            store.par_extend(a.apply_par_ref(work.par_iter().map(|b|{&**b}),magic));
            work.clear();
            std::mem::swap(&mut store,&mut work);

        }

        work.into_par_iter().map(|x|{
            self.gather.iter().map(|i|{x[*i].unwrap()})
            .collect()
        })
    }
}

pub struct QueryInfo<'kb> {
    pub(crate) query: AtomId,
    pub(crate) plans:HashMap<PredId,Box<[SolveAction]>>,
    pub(crate) kb: &'kb KB,
}

impl<'a> QueryInfo<'a> {
    pub fn new(query: AtomId, kb: &'a KB) -> Self {
        let preds: Vec<_> = kb.trace_pred(query.pred).into_iter().collect();
        Self { query, kb,plans:todo!() }
    }
}