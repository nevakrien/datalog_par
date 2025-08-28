/*!
 * this module basically exposes a tiny PL
 * we allow mapings from sets to sets taking into acount 1 magic set entry
 * we are not using a dyn trait here since that would cause multiple functions to be made
 * and all of those functions would be very similar with very littele benifit to switch
 * most of this runs in the same sort of loop so it should get branch predicted fairly quickly
 **/
use std::collections::VecDeque;
use crate::magic::KeyId;
use crate::magic::MagicSet;
use crate::parser::ConstId;
use crate::parser::PredId;
use hashbrown::HashMap;
use hashbrown::HashSet;
use rayon::iter::Either;
use rayon::iter::Empty;
use rayon::prelude::*;
use std::hash::Hash;

pub type QueryElem = [ConstId];

pub enum Gather {
    Exists(u32),
    Found(u32),
    Const(ConstId),
}

pub enum KeyGather {
    Var(u32),
    Const(ConstId),
}

pub struct RuleSolver {
    keyid: KeyId,
    key_gathers: Box<[KeyGather]>,
    val_gathers: Box<[Gather]>,
    exists_only: bool, //whether to stop on the first instance or no
}

impl RuleSolver {
    pub fn apply(
        &self,
        elems: &HashSet<Box<QueryElem>>,
        magic: &MagicSet,
        do_base: bool,
    ) -> (HashSet<Box<QueryElem>>, HashSet<Box<QueryElem>>) {
        elems
            .par_iter()
            .map_init(
                || Vec::with_capacity(self.key_gathers.len()),
                |key, elem| {
                    let elem = &**elem;
                    key.clear();

                    for k in self.key_gathers.iter() {
                        key.push(match k {
                            KeyGather::Var(i) => elem[*i as usize],
                            KeyGather::Const(c) => *c,
                        })
                    }

                    //we have a lot of empty cases so...
                    let empty = || Either::Left(Either::Left(rayon::iter::empty()));

                    let (base, delta) = match &magic[self.keyid].map.get(&**key) {
                        None => (empty(), empty()),
                        Some(spot) => {
                            let delta = self.run_on(&spot.1, elem);
                            let base = if do_base {
                                self.run_on(&spot.0, elem)
                            } else {
                                empty()
                            };

                            (base, delta)
                        }
                    };

                    base.map(|x| Either::Left(x))
                        .chain(delta.map(|x| Either::Right(x)))
                },
            )
            .flatten()
            .partition_map(|x| x)
    }

    fn run_on(
        &self,
        s: &HashSet<Box<QueryElem>>,
        elem: &QueryElem,
    ) -> Either<
        Either<Empty<Box<QueryElem>>, impl ParallelIterator<Item = Box<QueryElem>>>,
        impl ParallelIterator<Item = Box<QueryElem>>,
    > {
        //if no gathers we just check existance
        if self.exists_only {
            return Either::Left(if s.is_empty() {
                Either::Left(rayon::iter::empty())
            } else {
                Either::Right(rayon::iter::once(
                    self.val_gathers
                        .iter()
                        .map(|g| match g {
                            Gather::Exists(u) => elem[*u as usize],
                            Gather::Found(_u) => unreachable!(),
                            Gather::Const(c) => *c,
                        })
                        .collect(),
                ))
            });
        }

        //we we have to actually loop
        Either::Right(s.par_iter().map(|v| {
            self.val_gathers
                .iter()
                .map(|g| match g {
                    Gather::Exists(u) => elem[*u as usize],
                    Gather::Found(u) => v[*u as usize],
                    Gather::Const(c) => *c,
                })
                .collect()
        }))
    }
}

pub fn combine_sets<T: Hash + Eq>(mut a: HashSet<T>, mut b: HashSet<T>) -> HashSet<T> {
    if b.len() > a.len() {
        b.extend(a);
        b
    } else {
        a.extend(b);
        a
    }
}

pub fn combine_maps<T: Hash + Eq, V>(mut a: HashMap<T, V>, mut b: HashMap<T, V>) -> HashMap<T, V> {
    if b.len() > a.len() {
        b.extend(a);
        b
    } else {
        a.extend(b);
        a
    }
}

pub struct FullSolver {
    pub(crate)start: KeyId,
    pub(crate)parts: Box<[RuleSolver]>,
}

impl FullSolver {
    pub fn apply(&self, magic: &MagicSet) -> HashSet<Box<QueryElem>> {
        //get our first set
        // let Some(start) = &magic[self.start].map.get(&self.first_key) else {
        //     return HashSet::new();
        // };
        let Some(start) = &magic[self.start].map.get(&[][..]) else {
            return HashSet::new();
        };

        //we wana run on all cases except base base base ... base
        let (base, delta) = rayon::join(
            || self._apply(&start.0, magic, false, 0),
            || self._apply(&start.1, magic, true, 0),
        );

        //merge the results
        combine_sets(base, delta)
    }

    fn _apply(
        &self,
        elems: &HashSet<Box<QueryElem>>,
        magic: &MagicSet,
        used_delta: bool,
        i: usize,
    ) -> HashSet<Box<QueryElem>> {
        if i == self.parts.len() - 1 {
            let (base, delta) = self.parts[i].apply(elems, magic, used_delta);

            combine_sets(base, delta)
        } else {
            let (base, delta) = self.parts[i].apply(elems, magic, true);
            let (base, delta) = rayon::join(
                || self._apply(&base, magic, used_delta, i + 1),
                || self._apply(&delta, magic, true, i + 1),
            );

            combine_sets(base, delta)
        }
    }
}

pub struct SolveEngine {
    pub solvers: Vec<(FullSolver, Vec<PredId>)>, //predid should be unique
}

impl SolveEngine {
    //returns true if there was no meaningful thing to add
    pub fn solve_round(&self, magic: &mut MagicSet) -> bool {
        let magic_mut = magic;
        let magic = &*magic_mut;

        //make additon sets
        let new_set = self
            .solvers
            .par_iter()
            //1. actually run each solver
            .flat_map(|(s, pids)| {
                s.apply(magic).into_par_iter().flat_map(|x| {
                    pids.par_iter()
                        .filter_map(move |pred| {
                            //actual op
                            magic.additions(*pred, &x)
                        })
                        .flatten()
                })
            })
            //2. insert into a hashmap
            .fold(
                || magic.empty_new_set(),
                |mut m, (id, (k, v))| {
                    m[id.0].entry(k).or_default().insert(v);
                    m
                },
            )
            //3. combine to 1 hashmap
            .reduce_with(|mut x, y| {
                x.par_iter_mut()
                    .zip(y.into_par_iter())
                    .for_each(|(x, mut y)| {
                        if y.len() > x.len() {
                            std::mem::swap(x, &mut y);
                        }

                        x.extend(y.drain())
                    });
                x
            });
        // 4 put it in
        magic_mut.rotate();
        if let Some(mut new) = new_set {
            magic_mut.put_new_delta(&mut new)
        } else {
            true
        }
    }
}

pub struct QuerySolver{
    pub engine:SolveEngine,
    pub magic:MagicSet,
    pub target:KeyId
}

impl QuerySolver {
    pub fn current(&self)->&HashSet<Box<QueryElem>>{
        &self.magic[self.target].map.get(&[][..]).unwrap().1
    }

    #[inline]
    pub fn current_full(&self)->impl Iterator<Item=Box<[ConstId]>>{
        self.current().iter().map(|x| self.move_to_full(x))
    }

    pub fn solve_round(&mut self) -> Option<&HashSet<Box<QueryElem>>>{
        if self.engine.solve_round(&mut self.magic){
            return None
        }

        Some(self.current())
    }

    #[inline]
    pub fn move_to_full(&self,found:&QueryElem)->Box<[ConstId]>{
        self.magic[self.target].magic.move_to_full(&[],found).into()
    }

    #[inline]
    pub fn map_stop<F:FnMut(Box<[ConstId]>)->bool>(mut self,mut f:F) -> bool{
        if self.current_full().any(&mut f) {
            return true;
        }

        while !self.engine.solve_round(&mut self.magic){
            if self.current_full().any(&mut f) {
                return true;
            }
        }

        false
    }

    #[inline(always)]
    fn _print_n(self,mut n:usize){
        self.map_stop(|x| {
            if n == 0 {
                return true
            }
            println!("{:?}",x);
            n-=1;
            false
        });
    }

    pub fn print_n(self,n:usize){
        self._print_n(n)
    }

    pub fn print_all(self){
        self._print_n(usize::MAX)
    }

    pub fn get_all(&mut self) -> Vec<Box<QueryElem>>{
        let mut ans: Vec<_> = self.current_full().collect();
        while !self.engine.solve_round(&mut self.magic) {
            ans.extend(self.current().iter().map(|x| self.move_to_full(x)))
        }
        ans
    }

    /// Iterates over *deltas* round by round, cloning items once per yield.
    #[inline]
    pub fn iter<'a>(&'a mut self) -> impl Iterator<Item = Box<QueryElem>> + 'a {
        // Local buffer of the current round
        let mut buf: VecDeque<Box<QueryElem>> = self.current_full().collect();

        std::iter::from_fn(move || {
            if let Some(ans) = buf.pop_front(){
                return Some(ans);
            }
            if self.engine.solve_round(&mut self.magic){
                return None
            }
            buf.extend(self.current_full());
            buf.pop_front()

        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::magic::MagicKey;
    use crate::parser::{AtomId, PredId, TermId};
    use hashbrown::HashSet;
    use std::num::NonZeroU32;

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

    #[test]
    fn solve_round_no_new_additions_returns_true_and_keeps_delta_empty() {
        // --- arrange: empty MagicSet + a FullSolver that yields no results ---
        let p = pred_id(1);
        let mut ms = MagicSet::new();

        // Set up a generic bucket so rotate()/delta inspection are well-defined.
        // Arity here can be anything consistent with your KB; 1 is convenient.
        let kid_generic = ms.ensure_generic(p, 1);

        // Construct a FullSolver that will yield an empty collection from `apply(&ms)`.
        // This mirrors the "missing root key / empty parts" pattern from your other test.
        let solver_yields_nothing = FullSolver {
            start: kid_generic,
            parts: Box::from([RuleSolver{
                exists_only:true,
                keyid:kid_generic,
                key_gathers:Box::from([]),
                val_gathers:Box::from([]),
            }]),
        };

        // Wire it into a SolveEngine; pids can be anything — they won’t be touched
        // because `apply()` yields nothing.
        let engine = SolveEngine {
            solvers: vec![(solver_yields_nothing, vec![p])],
        };

        // --- act ---
        let changed = engine.solve_round(&mut ms);

        // --- assert: with no produced items, we expect the empty path:
        // reduce_with -> None, then rotate(), then return true
        assert!(
            changed,
            "expected solve_round() to return true on empty reduce"
        );

        // Also double-check that the *delta* for the bucket is still empty after the rotation.
        let mut any_delta_nonempty = false;
        if let Some((full, delta)) = ms[kid_generic].map.get(&Box::from([])) {
            if !delta.is_empty() {
                any_delta_nonempty = true;
            }
            // `full` can be anything; we only care that no *new* delta was created.
            let _ = full;
        } else {
            // If there's no entry at all, that's also fine — nothing was added.
        }

        assert!(
            !any_delta_nonempty,
            "expected no new delta entries when solvers produce no work"
        );
    }

    #[test]
    fn solve_round_inserts_new_tuple_into_q_from_p_delta_returns_false() {
        // predicates p/1 (source) and q/1 (target)
        let p = pred_id(1);
        let q = pred_id(2);

        let mut ms = MagicSet::new();

        // generic buckets (key = [])
        let kid_p = ms.ensure_generic(p, 1);
        let kid_q = ms.ensure_generic(q, 1);

        let a = const_id(777);
        let empty_key: Box<[ConstId]> = Box::from([]);

        // Seed p/1 Δ with [a] so we have a delta-leg
        ms[kid_p]
            .map
            .entry(empty_key.clone())
            .or_insert_with(|| (HashSet::new(), HashSet::new()))
            .1
            .insert(Box::from([a]));

        // Forward the found element (NOT Exists) so arity is correct: [a] -> [a]
        let forward_found = RuleSolver {
            keyid: kid_p,
            key_gathers: Box::from([]),                 // key=[]
            val_gathers: Box::from([Gather::Found(0)]), // take v[0] from found set
            exists_only: false,
        };

        let fs = FullSolver {
            start: kid_p,
            parts: Box::from([forward_found]),
        };

        // additions() targets q
        let engine = SolveEngine {
            solvers: vec![(fs, vec![q])],
        };

        // pre: q’s Δ empty
        assert!(
            ms[kid_q]
                .map
                .get(&empty_key)
                .map(|(_, d)| d.is_empty())
                .unwrap_or(true),
            "pre: q/1 Δ must be empty"
        );

        // act
        let done = engine.solve_round(&mut ms);

        // assert: since we DID insert, the function must return FALSE (no-op==true, change==false)
        assert!(
            !done,
            "solve_round() should return false when it performs insertions"
        );

        // p/1 rotated: Δ empty, FULL has [a]
        {
            let (pfull, pdelta) = ms[kid_p].map.get(&empty_key).expect("p/1 exists");
            assert!(pdelta.is_empty(), "p/1 Δ should be empty after rotate()");
            assert!(
                pfull.contains(&Box::from([a])),
                "p/1 FULL should contain [a]"
            );
        }

        // q/1 now has new Δ = {[a]}
        {
            let (_qfull, qdelta) = ms[kid_q].map.get(&empty_key).expect("q/1 exists");
            assert!(
                qdelta.contains(&Box::from([a])),
                "q/1 Δ should contain the newly inserted [a]"
            );
        }
    }

    #[test]
    fn rulesolver_missing_inner_key_returns_empty() {
        // predicate p/1
        let p = pred_id(1);
        let mut ms = MagicSet::new();
        let kid_generic = ms.ensure_generic(p, 1);

        // no entries added to ms[kid_generic].map

        // RuleSolver expects to look up elem[0], but it won't exist
        let rs = RuleSolver {
            keyid: kid_generic,
            key_gathers: Box::from([KeyGather::Var(0)]),
            val_gathers: Box::from([Gather::Exists(0)]),
            exists_only: false,
        };

        let elems: HashSet<Box<QueryElem>> = [Box::from([const_id(42)])].into_iter().collect();

        // previously, rs.apply() would panic indexing a missing key
        // now, with `.get()` fix, it should just yield empty
        let (base, delta) = rs.apply(&elems, &ms, false);
        assert!(
            base.is_empty() && delta.is_empty(),
            "expected empty result for missing inner key"
        );
    }

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

        let k_generic = Box::<[ConstId]>::from([]); // key for generic bucket
        let k1_key: Box<QueryElem> = Box::from([x, z]); // [x,z]
        let k2_key: Box<QueryElem> = Box::from([y]); // [y]

        // Seed FULL sets only:
        //
        // generic: [] -> {(x,y,z)}
        ms[kid_generic]
            .map
            .entry(k_generic.clone())
            .or_insert_with(|| (HashSet::new(), HashSet::new()))
            .0
            .insert(Box::from([x, y, z]));

        // K1: [x,z] -> {[y]}
        ms[k1]
            .map
            .entry(k1_key.clone())
            .or_insert_with(|| (HashSet::new(), HashSet::new()))
            .0
            .insert(Box::from([y]));

        // K2: [y] -> {[w]}
        ms[k2]
            .map
            .entry(k2_key.clone())
            .or_insert_with(|| (HashSet::new(), HashSet::new()))
            .0
            .insert(Box::from([w]));

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
            // first_key: k_generic.clone(),
            parts: Box::from([part0, part1]),
        };

        // Round 1: only FULL→FULL→FULL is available → should yield ∅.
        let out1 = solver.apply(&ms);
        assert!(
            out1.is_empty(),
            "all-base path must be excluded for a single iteration"
        );

        // Round 2: introduce a DELTA on K2 for the same key [y] so one leg is delta.
        // (If you have `put_new_delta(k2, k2_key.clone(), [box1(w)])`, use that instead.)
        ms[k2]
            .map
            .get_mut(&k2_key)
            .unwrap()
            .1
            .insert(Box::from([w]));
        let out2 = solver.apply(&ms);
        assert!(
            out2.contains(&Box::from([w])),
            "expected [w] once a delta leg exists; got: {out2:?}"
        );

        // After rotate (delta→full), next single iteration is back to all-base → ∅.
        ms.rotate();
        let out3 = solver.apply(&ms);
        assert!(
            out3.is_empty(),
            "after rotate, only base remains; single-iteration result should be empty"
        );
    }
}
