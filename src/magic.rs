use hashbrown::hash_map::Entry;
use crate::solve::Cheat;
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
    	self.atom.args.iter().enumerate().all(|(i,t)| {
    		if let TermId::Var(v) = t.term() {
    			v==i as u32
    		}else{
    			false
    		}
    	}) 
    }

    
}

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
        
        let mut key=Vec::with_capacity(arity);
        let mut val=Vec::with_capacity(arity);

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
struct Bucket {
    magic: CompiledMagic,
    map: HashMap<InnerKey, FullDelta>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct KeyId(usize);

pub struct MagicSet {
    buckets: Vec<Bucket>,                        // index by KeyId.0
    by_pred: HashMap<PredId, Vec<KeyId>>,        // pred -> registered KeyIds
    generic: HashMap<PredId, KeyId>,      // (pred, arity) -> generic KeyId (bounds=0)
    existing: HashMap<MagicKey,KeyId>,
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

	fn generic_bucket_mut(&mut self, pred: PredId)->&mut FullDelta{
		let gid = self.generic_id(pred);
        let gmap = &mut self.buckets[gid.0].map;
        gmap
            .get_mut(&InnerKey::from(&[][..])).unwrap()
	}

    fn generic_bucket(&self, pred: PredId)->&FullDelta{
        let gid = self.generic_id(pred);
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

        let Some(ids) = self.by_pred.get(&pred) else {
            panic!()
        };

        Some(ids.iter().filter_map(move |id| {
            let magic = &self.buckets[id.0].magic;

            let (k,v) =
                magic.match_and_project(c)?;
                Some((*id,(k,v.into())))
            
        }).collect())
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
                // Per-bucket: all incoming sets empty?
                // this is serial but fairly small per bucket
                map2.drain().all(|(k, mut v)| {
                    if v.is_empty() {
                        true
                    } else {
                        // swap into this bucket’s delta; old delta should be empty
                        std::mem::swap(&mut map.entry(k).or_default().1, &mut v);
                        false
                    }
                })
            })
            .reduce(|| true, |a, b| a && b)
    }

    /// Move all delta -> full.
    pub fn rotate(&mut self) {
    	//this blocks everything so we prallalize as much as we can
        self.buckets.par_iter_mut()
        .for_each(|Bucket { map, .. }|{
        	map.par_iter_mut().for_each(move |(_k,(full, delta))|{
        		full.extend(delta.drain());

                //memory is useless now and would be droped later
                //better do it now in parallel
                *delta=HashSet::new();
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
    #[cfg(test)]
    impl PredId {
        pub fn from_u32(n: u32) -> Self {
            Self(NonZeroU32::new(n).expect("PredId must be non-zero"))
        }
    }
    #[cfg(test)]
    impl ConstId {
        pub fn from_u32(n: u32) -> Self {
            Self(NonZeroU32::new(n).expect("ConstId must be non-zero"))
        }
    }

    // tiny helper to build an AtomId without the parser
    fn atom(pred: u32, terms: &[TermId]) -> AtomId {
        AtomId {
            pred: PredId::from_u32(pred),
            args: terms.iter().cloned().map(Into::into).collect(),
        }
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

        let a = ConstId::from_u32(1);
        let b = ConstId::from_u32(2);

        assert!(cm.matches(&[a, a, b]));   // ok
        assert!(!cm.matches(&[a, b, b]));  // X must equal X
        assert!(!cm.matches(&[a, a]));     // arity mismatch
    }

    #[test]
    fn matches_mix_constants_and_vars() {
        // p(c, X, c)
        let c = ConstId::from_u32(7);
        let key = MagicKey {
            atom: atom(2, &[TermId::Const(c), TermId::Var(0), TermId::Const(c)]),
            bounds: 0,
        };
        debug_assert!(key.atom.is_canon());

        let cm = CompiledMagic::make_me(key);

        let x = ConstId::from_u32(9);
        let y = ConstId::from_u32(9);
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

        let a = ConstId::from_u32(1);
        let b = ConstId::from_u32(2);
        let c = ConstId::from_u32(3);

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

        let a = ConstId::from_u32(10);
        let b = ConstId::from_u32(20);
        let c = ConstId::from_u32(30);

        let some = cm.match_and_project(&[a, b, c]).expect("should match");
        let (left, right) = some;

        assert_eq!(left, vec![a, c].into());     // positions 0 and 2
        assert_eq!(right, vec![b]);       // remaining position 1
        assert_eq!(left.len() + right.len(), 3);
    }

    #[test]
    fn constants_are_included_where_selected_by_bounds() {
        // p(k, X, k, Y), bounds = 0b0110 -> left: pos 1,2 ; right: pos 0,3
        let k = ConstId::from_u32(7);
        let key = MagicKey {
            atom: atom(2, &[TermId::Const(k), TermId::Var(0), TermId::Const(k), TermId::Var(1)]),
            bounds: 0b0110,
        };
        assert!(key.atom.is_canon());
        let cm = CompiledMagic::make_me(key);

        let x = ConstId::from_u32(8);
        let y = ConstId::from_u32(9);

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

        let x = ConstId::from_u32(11);
        let y = ConstId::from_u32(12);

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
        let a = ConstId::from_u32(1);
        assert!(cm.match_and_project(&[a]).is_none());
        assert!(cm.match_and_project(&[a, a, a]).is_none());
    }
}
