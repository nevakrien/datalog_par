/*!
 * this module basically exposes a tiny PL
 * we allow mapings from sets to sets taking into acount 1 magic set entry
 * we are not using a dyn trait here since that would cause multiple functions to be made 
 * and all of those functions would be very similar with very littele benifit to switch
 * most of this runs in the same sort of loop so it should get branch predicted fairly quickly
 **/
use rayon::iter::Empty;
use std::hash::Hash;
use hashbrown::HashSet;
use rayon::iter::Either;
use crate::magic::MagicSet;
use crate::magic::KeyId;
use crate::parser::ConstId;
use rayon::prelude::*;

pub type QueryElem = [ConstId];

pub enum Gather{
    Exists(u32),
    Found(u32),
    Const(ConstId),
}

pub enum KeyGather{
    Var(u32),
    Const(ConstId),
}

pub struct RuleSolver {
    keyid:KeyId,
    key_gathers: Box<[KeyGather]>,
    val_gathers: Box<[Gather]>,
    exists_only: bool, //whether to stop on the first instance or no
}

impl RuleSolver {
    pub fn apply(&self,elems:&HashSet<Box<QueryElem>>,magic:&MagicSet,delta:bool)->HashSet<Box<QueryElem>>{
        elems.par_iter().map_init(
            || Vec::with_capacity(self.key_gathers.len()),
            |key,elem|{
            let elem = &**elem;
            key.clear();

            for k in self.key_gathers.iter(){
                key.push(match k{
                    KeyGather::Var(i) => elem[*i as usize],
                    KeyGather::Const(c) => *c,
                })
            }

            let Some(spot) = &magic[self.keyid].map.get(&**key) else{
                return Either::Left(Either::Left(rayon::iter::empty()));
            };

            //TODO: we do 1 set at time
            //some times we actually do wana run both
            //so maybe this can be improved
            let s = if delta {&spot.1} else{&spot.0};
            
            self.run_on(s,elem)
        }).flatten().collect()
    }

    fn run_on(&self,s:&HashSet<Box<QueryElem>>,elem:&QueryElem)
    -> Either<Either<Empty<Box<QueryElem>>,impl ParallelIterator<Item=Box<QueryElem>>>,impl ParallelIterator<Item=Box<QueryElem>>>
    {
        //if no gathers we just check existance
        if self.exists_only{
            return Either::Left(
                if s.is_empty(){
                    Either::Left(rayon::iter::empty())
            }else{
                Either::Right(rayon::iter::once(
                    self.val_gathers.iter().map(|g|
                        match g{
                            Gather::Exists(u)=>elem[*u as usize],
                            Gather::Found(_u)=>unreachable!(),
                            Gather::Const(c) => *c,
                        }
                    ).collect()
                ))
            })
        }

        //we we have to actually loop
        Either::Right(s.par_iter().map(|v|{
            self.val_gathers.iter().map(|g|
                match g{
                    Gather::Exists(u)=>elem[*u as usize],
                    Gather::Found(u)=>v[*u as usize],
                    Gather::Const(c) => *c,
                }
            ).collect()
        }))

    }

}

pub fn combine_sets<T:Hash+Eq>(mut a:HashSet<T>,mut b:HashSet<T>)->HashSet<T>{
    if b.len() > a.len(){
        b.extend(a);
        b
    }else{
        a.extend(b);
        a
    }
}

pub struct FullSolver {
    start:KeyId,
    first_key:Box<QueryElem>,
    parts:Box<[RuleSolver]>,
}

impl FullSolver {
    pub fn apply(&self,magic:&MagicSet)->HashSet<Box<QueryElem>>{
        //get our first set
        let start =  &magic[self.start].map[&self.first_key];
        
        //we wana run on all cases except base base base ... base
        let (base,delta) = rayon::join(
            ||self._apply(&start.0,magic,false,0),
            ||self._apply(&start.1,magic,true,0),
        );

        //merge the results
        combine_sets(base,delta)

    }

    fn _apply(&self,elems:&HashSet<Box<QueryElem>>,magic:&MagicSet,used_delta:bool,i:usize)->HashSet<Box<QueryElem>>{
        if i==self.parts.len()-1{
            let (base,delta) = rayon::join(
                || if used_delta {
                        self.parts[i].apply(elems, magic, false)
                   } else {
                        HashSet::new()
                   },
                || self.parts[i].apply(elems, magic, true),
            );

            combine_sets(base,delta)
        }else{
            let (base,delta) = rayon::join(
                || {
                    let x = self.parts[i].apply(elems,magic,false);
                    self._apply(&x,magic,used_delta,i+1)
                },
                || {
                    let x = self.parts[i].apply(elems,magic,true);
                    self._apply(&x,magic,true,i+1)
                },
            );

            combine_sets(base,delta)
        }
    }
}

#[cfg(test)]
mod tests {



    use crate::magic::MagicKey;
use super::*;
    use hashbrown::HashSet;
    use std::num::NonZeroU32;
    use crate::parser::{TermId, AtomId, PredId};

    // --- tiny helpers --------------------------------------------------------
        // --- Minimal helpers for non-zero ids ---
    // If PredId/ConstId already expose a public `new(NonZeroU32)` or similar,
    // use that instead of these cfg(test) shims.
    fn pred_id(n: u32) -> PredId {
        // replace with your real way to make a PredId
        PredId(NonZeroU32::new(n).expect("PredId must be non-zero"))
    }
    fn const_id(n: u32) -> ConstId {
        // replace with your real way to make a ConstId
        ConstId(NonZeroU32::new(n).expect("PredId must be non-zero"))
    }

    // tiny helper to build an AtomId without the parser
    fn atom(pred: u32, terms: &[TermId]) -> AtomId {
        AtomId {
            pred: pred_id(pred),
            args: terms.iter().cloned().map(Into::into).collect(),
        }
    }

    fn box1(a: ConstId) -> Box<[ConstId]> { Box::from([a]) }
    fn box2(a: ConstId, b: ConstId) -> Box<[ConstId]> { Box::from([a, b]) }



    // Goal: prove `FullSolver::apply` does NOT produce results via a pure
    // FULL→FULL→FULL path in a single iteration.
    //
    // Setup (predicate p/3):
    //   • Generic key (bounds=0):        key=[]         FULL: {(x,y,z)}
    //   • Bounded key K1 (bounds=0b101): key=[x,z]      FULL: { [y] }   // projects middle
    //   • Bounded key K2 (bounds=0b010): key=[y]        FULL: { [w] }   // projects y→w
    //
    // Pipeline:
    //   part0: elem=[x,y,z]  --K1(x,z)-->  [y]
    //   part1: elem=[y]      --K2(y)-----> [w]
    //
    // Round 1: only FULL entries exist → all-base path is excluded → output = ∅.
    // Round 2: mark K2’s result as DELTA for the same key → now one leg is delta → output = {[w]}.
    //
    #[test]
    fn excludes_all_base_path_then_allows_when_delta_present() {
        // predicate p/3
        let p = pred_id(1);
        let arity = 3;

        let mut ms = MagicSet::new();

        // Generic (bounds=0) bucket, key is [].
        let kid_generic = ms.ensure_generic(p, arity);
        assert_eq!(kid_generic.0, 0);

        // K1: bounds=0b101 (key = {0,2} = [x,z]), value projects the middle [y].
        let k1 = ms.register(MagicKey {
            atom: atom(1, &[TermId::Var(0), TermId::Var(1), TermId::Var(2)]),
            bounds: 0b101,
        });

        // K2: bounds=0b010 (key = {1} = [y]), value projects something from y (→ [w]).
        let k2 = ms.register(MagicKey {
            atom: atom(1, &[TermId::Var(0), TermId::Var(1), TermId::Var(2)]),
            bounds: 0b010,
        });

        // Concrete constants
        let x = const_id(10);
        let y = const_id(20);
        let z = const_id(30);
        let w = const_id(40);

        let k_generic = Box::<[ConstId]>::from([]);  // key for generic bucket
        let k1_key    = box2(x, z);                  // [x,z]
        let k2_key    = box1(y);                     // [y]

        // Seed FULL sets only:
        //
        // generic: [] -> {(x,y,z)}
        ms[kid_generic].map
            .entry(k_generic.clone())
            .or_insert_with(|| (HashSet::new(), HashSet::new()))
            .0
            .insert(Box::from([x, y, z]));

        // K1: [x,z] -> {[y]}
        ms[k1].map
            .entry(k1_key.clone())
            .or_insert_with(|| (HashSet::new(), HashSet::new()))
            .0
            .insert(box1(y));

        // K2: [y] -> {[w]}
        ms[k2].map
            .entry(k2_key.clone())
            .or_insert_with(|| (HashSet::new(), HashSet::new()))
            .0
            .insert(box1(w));

        // Build the 2-step pipeline:
        // part0: from [x,y,z] via K1(x,z) → [y]
        let part0 = RuleSolver {
            keyid: k1,
            key_gathers: Box::from([KeyGather::Var(0), KeyGather::Var(2)]), // [x,z]
            val_gathers: Box::from([Gather::Found(0)]),                     // emit [y]
            exists_only: false,
        };
        // part1: from [y] via K2(y) → [w]
        let part1 = RuleSolver {
            keyid: k2,
            key_gathers: Box::from([KeyGather::Var(0)]), // [y]
            val_gathers: Box::from([Gather::Found(0)]),  // emit [w]
            exists_only: false,
        };

        let solver = FullSolver {
            start: kid_generic,
            first_key: k_generic.clone(),
            parts: Box::from([part0, part1]),
        };

        // Round 1: only FULL→FULL→FULL is available → should yield ∅.
        let out1 = solver.apply(&ms);
        assert!(out1.is_empty(), "all-base path must be excluded for a single iteration");

        // Round 2: introduce a DELTA on K2 for the same key [y] so one leg is delta.
        // (If you have `put_new_delta(k2, k2_key.clone(), [box1(w)])`, use that instead.)
        ms[k2].map.get_mut(&k2_key).unwrap().1.insert(box1(w));
        let out2 = solver.apply(&ms);
        assert!(
            out2.contains(&box1(w)),
            "expected [w] once a delta leg exists; got: {out2:?}"
        );

        // After rotate (delta→full), next single iteration is back to all-base → ∅.
        ms.rotate();
        let out3 = solver.apply(&ms);
        assert!(out3.is_empty(), "after rotate, only base remains; single-iteration result should be empty");
    }
}
