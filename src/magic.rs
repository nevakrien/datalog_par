use std::ops::IndexMut;
use std::ops::Index;
use crate::parser::AtomId;
use crate::parser::PredId;
use crate::parser::ConstId;
use crate::parser::TermId;
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;


// ---- Types ----
pub type Bound = u64;                    // arity <= 64 (debug assert)
pub type InnerKey = Box<[ConstId]>;      // projection of pattern-constants (in pos order)
pub type Projected = Box<[ConstId]>;     // projection of returned columns (in pos order)
pub type ConstSet = HashSet<Projected>;
pub type FullDelta = (ConstSet, ConstSet); // (full, delta)

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MagicKey {
    pub atom: AtomId,   // pattern: Const/Var per position
    pub bounds: Bound,  // bit i=1 => include tuple[i] in projected result
}

impl MagicKey {
    #[inline]
    pub fn generic(pred: PredId, arity: u32) -> Self {
        let atom = AtomId { pred, args: (0..arity).map(|i| TermId::Var(i).into()).collect() };
        Self { atom, bounds: 0 }
    }

    pub fn is_generic(&self)->bool{
        if self.bounds != 0 {
            return false;
        }
    	self.atom.args.iter().enumerate().all(|(i,t)| {
    		if let TermId::Var(v) = t.term() {
    			v==i as u32
    		}else{
    			false
    		}
    	}) 
    }

    
}

#[derive(Debug)]
pub struct CompiledMagic{
    key:MagicKey,
    var_dup_lookup:HashMap<u32, usize> //var -> first occurance
}

impl CompiledMagic {
    pub fn make_me(key:MagicKey)->Self{
        debug_assert!(key.atom.is_canon());

        // let mut id = 0;
        // let mut h: HashMap<_, u32> = HashMap::new();
        let mut var_dup_lookup = HashMap::new();

        for (pos, arg) in key.atom.args.iter().enumerate() {
            match arg.term() {
                TermId::Const(_kc) => {

                }
                TermId::Var(v) => {
                    // let id =match h.entry(v) {
                    //     Entry::Occupied(o) => {*o.get()},
                    //     Entry::Vacant(v) => {
                    //         v.insert(id);
                    //         id+=1;
                    //         id-1
                    //     }
                    // };

                   var_dup_lookup.entry(v).or_insert(pos);
                }
            }
        }
        Self{key,var_dup_lookup}
    }
    #[inline]
    pub fn matches(&self,c:&[ConstId])->bool{

        let arity = self.key.atom.args.len();
        if c.len() != arity { return false; }

        // constants must match; build inner_key in pos order
        for (pos, arg) in self.key.atom.args.iter().enumerate() {            
            match arg.term() {
                TermId::Const(kc) => {
                    if c[pos] != kc { return false; }
                }
                TermId::Var(v) => {
                    let idx = self.var_dup_lookup[&v];
                    if c[idx as usize]!=c[pos]{
                        return false
                    }
                }
            }
        }

        true

    }
    /// If `c` matches this pattern, compute (inner_key, projected) for insertion.
    //TODO this needs to be optimized... like ALOT optimized
    #[inline]
    pub fn match_and_project(
        &self,
        c: &[ConstId],
    ) -> Option<(InnerKey, Vec<ConstId>)> {
        if !self.matches(c){
            return None;
        }
        let arity = self.key.atom.args.len();
        debug_assert_eq!(self.key.bounds >> arity, 0, "bounds has bits beyond arity");
        
        let key_len = self.key.bounds.count_ones() as usize;
        let val_len = (!self.key.bounds & (1u64<<arity -1)).count_ones() as usize;
        
        let mut key=Vec::with_capacity(key_len);
        let mut val=Vec::with_capacity(val_len);

        // projection by bounds (pos order)
        for pos in 0..arity {     
            if (self.key.bounds >> pos) & 1 == 1 {
                key.push(c[pos])
            }
            else {
                val.push(c[pos]);
            }
        }

        Some((key.into_boxed_slice(), val))
    }
}

// Flat bucket with its pattern
#[derive(Debug)]
pub struct Bucket {
    pub magic: CompiledMagic,
    pub map: HashMap<InnerKey, FullDelta>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct KeyId(pub usize);

#[derive(Debug)]
pub struct MagicSet {
    buckets: Vec<Bucket>,                        // index by KeyId.0
    by_pred: HashMap<PredId, Vec<KeyId>>,        // pred -> registered KeyIds
    generic: HashMap<PredId, KeyId>,      // (pred, arity) -> generic KeyId (bounds=0)
    existing: HashMap<MagicKey,KeyId>,
}

impl Index<KeyId> for MagicSet {
type Output = Bucket;
fn index(&self, id: KeyId) -> &Bucket{ &self.buckets[id.0] }
}

impl IndexMut<KeyId> for MagicSet{
fn index_mut(&mut self, id: KeyId) -> &mut Bucket { &mut self.buckets[id.0]  }
}

impl MagicSet {
    pub fn new() -> Self {
        Self { buckets: Vec::new(), by_pred: HashMap::new(), generic: HashMap::new(),existing:HashMap::new() }
    }


    /// Register a magic pattern. Returns its KeyId.
    pub fn register(&mut self, key: MagicKey) -> KeyId {
        assert!(key.atom.args.len() <= 64, "arity > 64 not supported with Bound=u64");
        if let Some(i) = self.existing.get(&key){
        	return *i;
        }

        //register the key
        let id = KeyId(self.buckets.len());
        self.by_pred.entry(key.atom.pred).or_default().push(id);

        self.buckets.push(Bucket { magic: CompiledMagic::make_me(key.clone()), map: HashMap::new() });
        self.existing.insert(key.clone(),id);

        if key.is_generic(){
            self.generic.insert(key.atom.pred, id);
            self.buckets[id.0]
            	.map.entry(InnerKey::from(&[][..]))
	            .or_insert_with(|| (ConstSet::new(), ConstSet::new()));
        }else{
        	self.ensure_generic(key.atom.pred,key.atom.args.len() as u32);
        }
        id
    }

    #[inline]
	fn generic_id(&self, pred: PredId) -> KeyId {
	    *self.generic.get(&pred).expect("generic KeyId must be pre-registered")
	}

    pub fn generic_bucket(&self, pred: PredId)->&FullDelta{
        let gid = self.generic_id(pred);
        debug_assert!(self.buckets[gid.0].magic.key.is_generic());

        let gmap = &self.buckets[gid.0].map;
        gmap
            .get(&InnerKey::from(&[][..])).unwrap()
    }


    /// Register (or get) the generic bucket for (pred, arity).
    pub fn ensure_generic(&mut self, pred: PredId, arity:u32) -> KeyId {
        if let Some(&id) = self.generic.get(&pred) { return id; }
        let id = self.register(MagicKey::generic(pred, arity));
        id
    }


    pub fn additions(&self, pred: PredId, c: &[ConstId])->Option<Vec<(KeyId,(InnerKey, Box<[ConstId]>))>>{
        let entry = self.generic_bucket(pred);

        if entry.0.contains(c) || entry.1.contains(c) {
            return None;
        }

        let ids = &self.by_pred[&pred];
        Some(ids.iter().filter_map(|id| {
            let magic = &self.buckets[id.0].magic;

            // println!("in id {} with {c:?}", id.0);
            let (k,v) =
                magic.match_and_project(c)?;
                // println!("found addition with val {v:?}");
                Some((*id,(k,v.into())))
            
        }).collect())
    }

    pub fn empty_new_set(&self) -> Vec<HashMap<InnerKey, ConstSet>> {
        (0..self.buckets.len()).map(|_| HashMap::new()).collect()
    }


    ///puts in a new set of delta and checks if they are all empty
    //we want to try avoid destructors so the old delta is put back into new
    pub fn put_new_delta(
        &mut self,
        new: &mut Vec<HashMap<InnerKey, ConstSet>>
    ) -> bool {
        assert_eq!(new.len(), self.buckets.len());

        self.buckets
            .par_iter_mut()
            .zip(new.into_par_iter())
            .map(|(Bucket { map, .. }, map2)| {
                if map2.is_empty(){
                    return true;
                }
                map2.drain().for_each(|(k, mut v)| {
                    debug_assert!(!v.is_empty(),"we have a nonsensical empty insert");

                    std::mem::swap(&mut map.entry(k).or_default().1, &mut v);
                    debug_assert!(v.is_empty());

                });
                false
            })
            .reduce(|| true, |a, b| a && b)
    }

    /// Move all delta -> full.
    pub fn rotate(&mut self) {
    	//this blocks everything so we prallalize as much as we can
        self.buckets.par_iter_mut()
        .for_each(|Bucket { map, .. }|{
        	map.par_iter_mut().for_each(move |(_k,(full, delta))|{

                full.extend(delta.drain()); //serial since we have enough (extend would side allocate)
                // full.par_extend(delta.par_drain());

                //memory is useless now and would be droped later
                //better do it now in parallel
                delta.shrink_to_fit();
                debug_assert!(delta.is_empty());

        	});
        });
    }

}



#[cfg(test)]
mod tests {
    use super::*;
    use core::num::NonZeroU32;

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
    fn end_to_end() {
        // predicate p/3
        let p = pred_id(1);
        let arity = 3;

        let mut ms = MagicSet::new();

        // Ensure generic bucket (bounds = 0)
        let gid = ms.ensure_generic(p, arity);
        assert_eq!(gid.0, 0);


        // Also register one bounded key to exercise projection:
        // p(X,Y,Z), bounds=0b101 => key {0,2}, projec?t middle arg
        let bounded = MagicKey {
            atom: atom(1, &[TermId::Var(0), TermId::Var(1), TermId::Var(2)]),
            bounds: 0b101,
        };
        let kid_bounded = ms.register(bounded);

        println!("got magic set entries {:?}",ms.by_pred);
        println!("with {} buckets observing {kid_bounded:?}",ms.buckets.len());

        // Base candidates (with a duplicate)
        let c10 = const_id(10);
        let c20 = const_id(20);
        let c21 = const_id(21);
        let c30 = const_id(30);
        let c31 = const_id(31);
        let base: Vec<[ConstId; 3]> = vec![
            [c10, c20, c30],
            [c10, c20, c31],
            [c10, c21, c30],
            [c10, c21, c31],
            [c10, c20, c30], // dup of #0
        ];

        // Track a concrete projection to assert bounded FULL later
        let mut seen_bounded: Option<(InnerKey, Box<[ConstId]>)> = None;

        // 5 rounds: PREP -> ROTATE -> PUT_NEW_DELTA
        for round in 0..5 {
            println!("round:{round}", );
            // --- PREPARE: compute additions and stage into `new` BEFORE rotate ---
            // Add one fresh tuple per round so each round definitely stages something.
            let fresh = [c10, const_id(100 + round as u32), c30];
            let mut cand = base.clone();
            cand.push(fresh);

            let mut new = ms.empty_new_set();
            let mut staged_any = false;
            for c in &cand {
                if let Some(adds) = ms.additions(p, c) {
                    for (id, (ik, v)) in adds {
                        new[id.0]
                            .entry(ik.clone())
                            .or_insert_with(ConstSet::new)
                            .insert(v.clone());

                        if id == kid_bounded && seen_bounded.is_none() {
                            seen_bounded = Some((ik, v));
                        }
                        staged_any = true;
                    }
                }
            }
            assert!(staged_any, "must stage at least one tuple per round");


            // --- ROTATE: promote prior delta -> full ---
            ms.rotate();

            // --- IMMEDIATELY PUT NEW DELTA: delta becomes non-empty for the rest of the round ---
            let all_empty = ms.put_new_delta(&mut new);
            assert!(!all_empty, "put_new_delta should report non-empty in round {round}");

            // Check that generic DELTA is now non-empty (middle-of-round invariant)
            let (_full_mid, delta_mid) = ms.generic_bucket(p);
            assert!(
                !delta_mid.is_empty(),
                "generic DELTA must be non-empty in the middle of round {round}"
            );

            if let Some((ref ik,ref v)) = seen_bounded {
                let (bfull, bdelta) = &ms.buckets[kid_bounded.0].map[ik];
                assert!(bfull.contains(&*v) || bdelta.contains(&*v), 
                "bounded ({kid_bounded:?}) missing expected projection {v:?}");
            }


            // The fresh tuple is now “known” (already in DELTA), so additions must return None.
            assert!(
                ms.additions(p, &fresh).is_none(),
                "fresh tuple should be recognized in DELTA in round {round}"
            );
        }

        // Final ROTATE to promote last round's DELTA -> FULL
        ms.rotate();

        // Generic FULL should have 4 distinct from base + 5 fresh = 9
        let (gfull, gdelta) = ms.generic_bucket(p);
        assert_eq!(gfull.len(), 4 + 5, "generic FULL cardinality mismatch at end");
        assert!(gdelta.is_empty(), "generic DELTA should be empty after final rotate");

        // Every tuple we ever tried should now be known
        for c in base.iter().cloned().chain((0..5).map(|r| [c10, const_id(100 + r), c30])) {
            assert!(ms.additions(p, &c).is_none(), "tuple {:?} should be known at end", c);
        }

        // Bounded bucket sanity: at least one FULL projection, empty DELTA
        let (ik, v) = seen_bounded.expect("bounded key should have projected at least once");
        let (bfull, bdelta) = &ms.buckets[kid_bounded.0].map[&ik];
        let mut got: Vec<_> = bfull.iter().collect();
        got.sort();
        println!("got {got:?}");
        assert!(bfull.contains(&*v), "bounded FULL ({kid_bounded:?}) missing expected projection {v:?}");
        assert!(bdelta.is_empty(), "bounded DELTA should be empty after final rotate");
    }


    #[test]
    fn matches_repeated_vars() {
        // p(X, X, Y)
        let key = MagicKey {
            atom: atom(1, &[TermId::Var(0), TermId::Var(0), TermId::Var(1)]),
            bounds: 0,
        };
        debug_assert!(key.atom.is_canon());

        let cm = CompiledMagic::make_me(key);

        let a = const_id(1);
        let b = const_id(2);

        assert!(cm.matches(&[a, a, b]));   // ok
        assert!(!cm.matches(&[a, b, b]));  // X must equal X
        assert!(!cm.matches(&[a, a]));     // arity mismatch
    }

    #[test]
    fn matches_mix_constants_and_vars() {
        // p(c, X, c)
        let c = const_id(7);
        let key = MagicKey {
            atom: atom(2, &[TermId::Const(c), TermId::Var(0), TermId::Const(c)]),
            bounds: 0,
        };
        debug_assert!(key.atom.is_canon());

        let cm = CompiledMagic::make_me(key);

        let x = const_id(9);
        let y = const_id(9);
        assert!(cm.matches(&[c, x, c]));              // constants align
        assert!(!cm.matches(&[c, x, y])); // last const mismatch
        assert!(!cm.matches(&[y, x, c])); // first const mismatch
    }

    #[test]
    fn matches_fully_generic() {
        // p(X, Y, Z)
        let key = MagicKey {
            atom: atom(3, &[TermId::Var(0), TermId::Var(1), TermId::Var(2)]),
            bounds: 0,
        };
        debug_assert!(key.atom.is_canon());

        let cm = CompiledMagic::make_me(key);

        let a = const_id(1);
        let b = const_id(2);
        let c = const_id(3);

        assert!(cm.matches(&[a, b, c]));
        assert!(cm.matches(&[a, a, a]));
    }

    //============================================================================================
    //============================================================================================
    //============================================================================================
    //============================================================================================
    //============================================================================================

    #[test]
    fn fully_generic_bounds_101() {
        // p(X, Y, Z), bounds = 0b101 -> left: pos 0,2 ; right: pos 1
        let key = MagicKey {
            atom: atom(1, &[TermId::Var(0), TermId::Var(1), TermId::Var(2)]),
            bounds: 0b101,
        };
        assert!(key.atom.is_canon());
        let cm = CompiledMagic::make_me(key);

        let a = const_id(10);
        let b = const_id(20);
        let c = const_id(30);

        let some = cm.match_and_project(&[a, b, c]).expect("should match");
        let (left, right) = some;

        assert_eq!(left, vec![a, c].into());     // positions 0 and 2
        assert_eq!(right, vec![b]);       // remaining position 1
        assert_eq!(left.len() + right.len(), 3);
    }

    #[test]
    fn constants_are_included_where_selected_by_bounds() {
        // p(k, X, k, Y), bounds = 0b0110 -> left: pos 1,2 ; right: pos 0,3
        let k = const_id(7);
        let key = MagicKey {
            atom: atom(2, &[TermId::Const(k), TermId::Var(0), TermId::Const(k), TermId::Var(1)]),
            bounds: 0b0110,
        };
        assert!(key.atom.is_canon());
        let cm = CompiledMagic::make_me(key);

        let x = const_id(8);
        let y = const_id(9);

        let (left, right) = cm.match_and_project(&[k, x, k, y]).expect("should match");
        assert_eq!(left,  vec![x, k].into());    // pos 1,2 (includes constant k)
        assert_eq!(right, vec![k, y]);    // pos 0,3
        assert_eq!(left.len() + right.len(), 4);
    }

    #[test]
    fn repeated_vars_enforced_then_split_by_bounds() {
        // p(X, X, Y), bounds = 0b011 -> left: pos 0,1 ; right: pos 2
        let key = MagicKey {
            atom: atom(3, &[TermId::Var(0), TermId::Var(0), TermId::Var(1)]),
            bounds: 0b011,
        };
        assert!(key.atom.is_canon());
        let cm = CompiledMagic::make_me(key);

        let x = const_id(11);
        let y = const_id(12);

        // matches: X == X
        let (left, right) = cm.match_and_project(&[x, x, y]).expect("should match");
        assert_eq!(left,  vec![x, x].into());
        assert_eq!(right, vec![y]);

        // does not match when repeated var differs
        assert!(cm.match_and_project(&[x, y, y]).is_none());
    }

    #[test]
    fn arity_mismatch_is_none() {
        let key = MagicKey {
            atom: atom(4, &[TermId::Var(0), TermId::Var(1)]),
            bounds: 0b01,
        };
        let cm = CompiledMagic::make_me(key);
        let a = const_id(1);
        assert!(cm.match_and_project(&[a]).is_none());
        assert!(cm.match_and_project(&[a, a, a]).is_none());
    }


    //============constant stuff
    #[test]
    fn constants_only_with_projection_bounds_101() {
        // p(c1, c2, c3), bounds = 0b101 -> left: {0,2} = [c1,c3], right: {1} = [c2]
        let c1 = const_id(1);
        let c2 = const_id(2);
        let c3 = const_id(3);

        let key = MagicKey {
            atom: atom(1, &[TermId::Const(c1), TermId::Const(c2), TermId::Const(c3)]),
            bounds: 0b101,
        };
        debug_assert!(key.atom.is_canon());
        let cm = CompiledMagic::make_me(key);

        let (left, right) = cm.match_and_project(&[c1, c2, c3]).expect("should match");
        assert_eq!(left, vec![c1, c3].into());
        assert_eq!(right, vec![c2]);

        // mismatch on any const => no match
        assert!(cm.match_and_project(&[c1, c3, c2]).is_none());
    }

    #[test]
    fn constants_and_vars_projection_bound_const_first() {
        // p(c1, X, c2), bounds = 0b001 -> left: {0} = [c1], right: {1,2} = [X,c2]
        let c1 = const_id(11);
        let c2 = const_id(12);

        let key = MagicKey {
            atom: atom(2, &[TermId::Const(c1), TermId::Var(0), TermId::Const(c2)]),
            bounds: 0b001,
        };
        debug_assert!(key.atom.is_canon());
        let cm = CompiledMagic::make_me(key);

        let x = const_id(99);
        let (left, right) = cm.match_and_project(&[c1, x, c2]).expect("should match");
        assert_eq!(left,  vec![c1].into());
        assert_eq!(right, vec![x, c2]);

        // const must align
        assert!(cm.match_and_project(&[c1, x, x]).is_none()); // last const mismatch
        assert!(cm.match_and_project(&[x,  x, c2]).is_none()); // first const mismatch
    }

    #[test]
    fn repeated_var_with_const_projection() {
        // p(X, X, c), bounds = 0b110 -> left: {0,1} = [X,X], right: {2} = [c]
        let c = const_id(7);

        let key = MagicKey {
            atom: atom(3, &[TermId::Var(0), TermId::Var(0), TermId::Const(c)]),
            bounds: 0b011,
        };
        debug_assert!(key.atom.is_canon());
        let cm = CompiledMagic::make_me(key);

        let x = const_id(42);

        // matches when X==X
        let (left, right) = cm.match_and_project(&[x, x, c]).expect("should match");
        assert_eq!(right, vec![c]);
        assert_eq!(left,  vec![x, x].into());

        // fails when repeated var differs
        assert!(cm.match_and_project(&[x, const_id(43), c]).is_none());
    }

    #[test]
    fn all_bound_projection_is_empty_tuple() {
        // p(c1, Y, c2), bounds = 0b111 -> left: {0,1,2} = [c1,Y,c2], right: {} = []
        let c1 = const_id(21);
        let c2 = const_id(22);

        let key = MagicKey {
            atom: atom(4, &[TermId::Const(c1), TermId::Var(0), TermId::Const(c2)]),
            bounds: 0b111,
        };
        debug_assert!(key.atom.is_canon());
        let cm = CompiledMagic::make_me(key);

        let y = const_id(5);
        let (left, right) = cm.match_and_project(&[c1, y, c2]).expect("should match");
        assert_eq!(left,  vec![c1, y, c2].into());
        assert!(right.is_empty(), "projected (right) must be empty when all positions are bound");

        // const mismatch -> no match
        assert!(cm.match_and_project(&[c1, y, const_id(99)]).is_none());
    }

    #[test]
    fn generic_constants_projection_bounds_000() {
        // p(c1, c2, c3), bounds = 0 -> left: {} = [], right: {all} = [c1,c2,c3]
        let c1 = const_id(31);
        let c2 = const_id(32);
        let c3 = const_id(33);

        let key = MagicKey {
            atom: atom(5, &[TermId::Const(c1), TermId::Const(c2), TermId::Const(c3)]),
            bounds: 0,
        };
        debug_assert!(key.atom.is_canon());
        let cm = CompiledMagic::make_me(key);

        let (left, right) = cm.match_and_project(&[c1, c2, c3]).expect("should match");
        assert_eq!(left,  Vec::<ConstId>::new().into()); // empty key
        assert_eq!(right, vec![c1, c2, c3]);

        // mismatch -> no match
        assert!(cm.match_and_project(&[c1, c3, c2]).is_none());
    }

}
