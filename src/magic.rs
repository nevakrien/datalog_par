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

    //TODO remove the inner vecs
    /// If `c` matches this pattern, compute (inner_key, projected) for insertion.
    #[inline]
    pub fn match_and_project<'inner, 'proj>(
        &self,
        c: &[ConstId],
        scratch_inner: &'inner mut Vec<ConstId>,
        scratch_proj:  &'proj  mut Vec<ConstId>,
    ) -> Option<(InnerKey, &'proj [ConstId])> {
        let arity = self.atom.args.len();
        if c.len() != arity { return None; }
        debug_assert_eq!(self.bounds >> arity, 0, "bounds has bits beyond arity");

        scratch_inner.clear();
        scratch_proj.clear();

        // constants must match; build inner_key in pos order
        for (pos, arg) in self.atom.args.iter().enumerate() {
            match arg.term() {
                TermId::Const(kc) => {
                    if c[pos] != kc { return None; }
                    scratch_inner.push(c[pos]);
                }
                TermId::Var(_v) => {todo!()/*check clashes*/}
            }
        }

        // projection by bounds (pos order)
        for pos in 0..arity {
            if (self.bounds >> pos) & 1 == 1 {
                scratch_proj.push(c[pos]);
            }
        }

        Some((scratch_inner.clone().into_boxed_slice(), &*scratch_proj))
    }
}

// Flat bucket with its pattern
struct Bucket {
    key: MagicKey,
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

        self.buckets.push(Bucket { key: key.clone(), map: HashMap::new() });
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

    /// Insert a fully-ground tuple `c` for predicate `pred`.
	/// Returns true if any bucket staged a new delta.
	pub fn insert(&mut self, pred: PredId, c: &[ConstId]) -> bool {
	    // 1) generic bucket (pattern = all vars, bounds = 0, inner_key = [])
        let entry = self.generic_bucket_mut(pred);
        if !entry.0.contains(c) && !entry.1.insert(c.into()) {
            return false;
        }

	    // 2) specialized buckets for this pred
	    if let Some(ids) = self.by_pred.get(&pred) {
	        let (mut inner, mut proj) = (Vec::new(), Vec::new());
	        for &id in ids {
	            let key = &self.buckets[id.0].key;
	            if let Some((inner_key_box, proj_borrowed)) =
	                key.match_and_project(c, &mut inner, &mut proj)
	            {
	                let entry = self.buckets[id.0]
	                    .map
	                    .entry(inner_key_box)
	                    .or_insert_with(|| (ConstSet::new(), ConstSet::new()));

	                if !entry.0.contains(proj_borrowed) {
	                    entry.1.insert(Projected::from(proj_borrowed));
	                }
	            }
	        }
	    }

	    true
	}

    pub fn additions(&self, pred: PredId, c: &[ConstId])->Option<Vec<(KeyId,(InnerKey, Box<[ConstId]>))>>{
        let entry = self.generic_bucket(pred);

        if entry.0.contains(c) || entry.1.contains(c) {
            return None;
        }

        let Some(ids) = self.by_pred.get(&pred) else {
            panic!()
        };

        let (mut inner, mut proj) = (Vec::new(), Vec::new());
        Some(ids.iter().filter_map(move |id| {
            let key = &self.buckets[id.0].key;

            let (k,v) =
                key.match_and_project(c, &mut inner, &mut proj)?;
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
    	//first we do all the extends in parallel
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
