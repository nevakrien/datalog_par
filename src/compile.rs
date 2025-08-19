// //----- Core solve actions-----------
// use crate::parser::RuleId;
// use crate::parser::Term32;
// use crate::parser::TermId;
// use crate::parser::{AtomId, ConstId, KB, PredId};
// use rayon::prelude::*;

// use hashbrown::HashSet;

// use hashbrown::HashMap;

// pub type ValPlan = Box<[Option<ConstId>]>;
// pub type ValPlanRef<'a> = &'a [Option<ConstId>];

// /// full info for magic set lookup
// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
// pub struct MagicKey {
//     pub pred: PredId,
//     pub vars: Box<[Option<ConstId>]>,
// }

// impl MagicKey {
//     pub fn from_atom(atom: &AtomId) -> Self {
//         Self {
//             pred: atom.pred,
//             vars: atom.args.iter().map(|a| a.try_const()).collect(),
//         }
//     }
// }

// #[derive(Debug, Clone)]
// pub struct SolveAction {
//     pub(crate) magic_template: MagicKey,
//     pub(crate) magic_insert: Box<[(usize, usize)]>, //found -> template
//     pub(crate) reduce_indecies: Box<[usize]>,
// }

// impl SolveAction {
//     // pub fn apply<'a>(&self,start: impl Iterator<Item = ValPlanRef<'a>>,magic:&HashMap<MagicKey,HashSet<Box<[ConstId]>>>) -> Vec<ValPlan>{
//     //     let mut ans = Vec::with_capacity(start.size_hint().0);

//     //     for x in start {
//     //         let mut key = self.magic_template.clone();
//     //         for (i,j) in &self.magic_insert{
//     //             key.vars[*j] = x[*i];
//     //         }

//     //         for item in &magic[&key]{
//     //             let mut new : ValPlan= x.into();
//     //             for (i,v) in self.reduce_indecies.iter().zip(item.iter()){
//     //                 new[*i] = Some(*v);
//     //             }

//     //             ans.push(new);
//     //         }

//     //     }

//     //     ans
//     // }

//     // pub fn apply_par_owned(&self,start: impl ParallelIterator<Item=ValPlan>,magic:&HashMap<MagicKey,HashSet<Box<[ConstId]>>>) -> impl ParallelIterator<Item=ValPlan>{
//     //     start.flat_map(move |x| {
//     //         self.run_par_item_owned(x,magic)
//     //     })

//     // }
//     // pub fn run_par_item_owned(&self,x: ValPlan,magic:&HashMap<MagicKey,HashSet<Box<[ConstId]>>>) -> impl ParallelIterator<Item=ValPlan>{
//     //     let mut key = self.magic_template.clone();
//     //     for (i,j) in &self.magic_insert{
//     //         key.vars[*j] = x[*i];
//     //     }

//     //     magic[&key].par_iter().map(move |item|{
//     //         let mut new : ValPlan = x.clone();
//     //         for (i,v) in self.reduce_indecies.iter().zip(item.iter()){
//     //             new[*i] = Some(*v);
//     //         }

//     //         new
//     //     })
//     // }

//     // pub fn apply_par_ref<'a>(&self,start: impl ParallelIterator<Item=ValPlanRef<'a>>,magic:&HashMap<MagicKey,HashSet<Box<[ConstId]>>>) -> impl ParallelIterator<Item=ValPlan>{
//     //     start.flat_map(|x| {
//     //         self.run_par_item(x,magic)
//     //     })

//     // }

//     // pub fn run_par_item<'a>(&self,x: ValPlanRef<'a>,magic:&HashMap<MagicKey,HashSet<Box<[ConstId]>>>) -> impl ParallelIterator<Item=ValPlan>{
//     //     let mut key = self.magic_template.clone();
//     //     for (i,j) in &self.magic_insert{
//     //         key.vars[*j] = x[*i];
//     //     }

//     //     magic[&key].par_iter().map(|item|{
//     //         let mut new : ValPlan = x.into();
//     //         for (i,v) in self.reduce_indecies.iter().zip(item.iter()){
//     //             new[*i] = Some(*v);
//     //         }

//     //         new
//     //     })
//     // }
// }

// pub struct SolveRule {
//     actions: Box<[SolveAction]>,
//     gather: Box<[Term32]>, //made -> plan
//     var_count: usize,
// }

// impl SolveRule {
//     pub fn simple_compile(rule: &RuleId) -> Self {
//         let gather = rule.head.args.clone();
//         let var_count = rule.var_count();

//         let mut bound: Box<[bool]> = (0..var_count).map(|_| false).collect();
//         let actions = rule
//             .body
//             .iter()
//             .map(|a| {
//                 let magic_template = MagicKey::from_atom(a);

//                 let mut magic_insert = Vec::with_capacity(rule.body.len());
//                 let mut reduce_indecies = Vec::with_capacity(rule.body.len());

//                 for (i, t) in a.args.iter().enumerate() {
//                     let TermId::Var(v) = t.term() else {
//                         continue;
//                     };
//                     let v = v as usize;

//                     if bound[v] {
//                         if reduce_indecies.contains(&v) {
//                             //was bound now X X style
//                             reduce_indecies.push(v);
//                         } else {
//                             //was bound previously
//                             magic_insert.push((v, i)) //found(v)->template(i)
//                         }
//                     } else {
//                         bound[v] = true;
//                         reduce_indecies.push(v);
//                     }
//                 }

//                 let magic_insert = magic_insert.into();
//                 let reduce_indecies = reduce_indecies.into();
//                 SolveAction {
//                     magic_template,
//                     reduce_indecies,
//                     magic_insert,
//                 }
//             })
//             .collect();
//         Self {
//             gather,
//             var_count,
//             actions,
//         }
//     }
//     // pub fn apply(&self,start: &HashSet<ValPlan>,magic:&HashMap<MagicKey,HashSet<Box<[ConstId]>>>) -> Vec<Box<[ConstId]>>{
//     //     let mut work = self.actions[0].apply(start.iter().map(|x| {&**x}),magic);

//     //     for a in &self.actions[1..]{
//     //         work = a.apply(work.iter().map(|x| {&**x}),magic);
//     //     }
//     //     work.into_iter().map(|x|{
//     //         self.gather.iter().map(|t|{
//     //             match t.term() {
//     //                 TermId::Var(i)=> x[i as usize].unwrap(),
//     //                 TermId::Const(c) => c,
//     //             }
//     //         })
//     //         .collect()
//     //     }).collect()
//     // }

//     // pub fn par_apply<'a>(&self,start: impl ParallelIterator<Item=ValPlanRef<'a>>,magic:&HashMap<MagicKey,HashSet<Box<[ConstId]>>>) -> impl ParallelIterator<Item=Box<[ConstId]>>{
//     //     //we would of liked to just make 1 giant par iter
//     //     //rust cant really compile that...
//     //     //there is also an argument that doing these vec stores are fine

//     //     //there are no dups
//     //     let mut work: Vec<ValPlan> = self.actions[0].apply_par_ref(start,magic).collect();
//     //     let mut store = Vec::with_capacity(work.len());//get some memory for us

//     //     for a in &self.actions[1..]{
//     //         store.par_extend(a.apply_par_ref(work.par_iter().map(|b|{&**b}),magic));
//     //         work.clear();
//     //         std::mem::swap(&mut store,&mut work);

//     //     }

//     //     work.into_par_iter().map(|x|{
//     //         self.gather.iter().map(|t|{
//     //             match t.term() {
//     //                 TermId::Var(i)=> x[i as usize].unwrap(),
//     //                 TermId::Const(c) => c,
//     //             }
//     //         })
//     //         .collect()
//     //     })
//     // }
// }

// pub struct QueryInfo<'kb> {
//     pub(crate) query: AtomId,
//     pub(crate) plans: HashMap<PredId, Box<[SolveRule]>>,
//     pub(crate) kb: &'kb KB,
// }

// impl<'a> QueryInfo<'a> {
//     pub fn new(query: AtomId, kb: &'a KB) -> Self {
//         let preds = kb.trace_pred(query.pred);
//         let mut plans = HashMap::new();

//         for pred in preds.into_iter() {
//             let p = kb.producers[&pred]
//                 .rules
//                 .iter()
//                 .map(|r| SolveRule::simple_compile(r))
//                 .collect();
//             plans.insert(pred, p);
//         }

//         Self { query, kb, plans }
//     }
// }
