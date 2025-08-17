use crate::parser::TermId;
use crate::parser::AtomId;
use crate::parser::ConstId;
use crate::parser::PredId;
use crate::solve::Cheat;
use hashbrown::HashMap;
use hashbrown::HashSet;
use rayon::prelude::*;

pub type Bound = u64;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MagicKey {
    pub atom: AtomId, //canonized
    pub bounds: Bound, //mask asking what mebers to return
                      //note this has to be precalculated and better kept to a small set
}

impl MagicKey{
	pub fn generic(pred:PredId,arity:u32)->Self{
		let atom = AtomId{
			pred,
			args:(0..arity).map(|i|{TermId::Var(i).into()}).collect()
		};
		Self{
			atom,
			bounds:0,
		}
	}

	//checks whether constant matches the pattern
	//if it does returns the key and value for magic sets inner
	fn get_magic_value(&self,_c:&[ConstId])->Option<(Box<[ConstId]>,Box<[ConstId]>)>{
		todo!()
	}
}

pub type ConstSet = HashSet<Box<[ConstId]>>;
pub type FullDelta = (Cheat<ConstSet>, Cheat<ConstSet>);

pub struct MagicSet {
    members: HashMap<MagicKey, Cheat<HashMap<Box<[ConstId]>, FullDelta>>>,
    by_pred: HashMap<PredId, Vec<MagicKey>>,
}

fn empty<'a,T>()->&'a[T]{
	&[]
}

impl MagicSet {
	pub fn insert_many(&mut self,_pred:PredId,_c:Vec<Box<[ConstId]>>)->bool{
		todo!()
	}
	pub fn insert(&mut self,pred:PredId,c:&[ConstId])->bool{
		//first try and insert into the most generic category
		let key = MagicKey::generic(pred,c.len() as u32);
		let spot = self.members.get_mut(&key).unwrap().get_mut(empty()).unwrap();
		if spot.0.contains(c){
			return false;
		}
		if !spot.1.insert(c.into()){
			return false;
		}
		self.by_pred[&pred].par_iter()
		.filter_map(|k| {Some((k,k.get_magic_value(c)?))})
		.for_each(|(k,v)|{
			//each key is unique and so each hashset is unique
			unsafe{
				self.members[k].unsafe_mut().entry(v.0)
				.or_insert_with(||{(HashSet::new().into(),HashSet::new().into())}).1.insert(v.1);

			}
		});
		true

	}
    pub fn rotate(&mut self) -> bool {
        !self
            .members
            .par_iter_mut()
            .flat_map(|(_, m)| {
                m.par_iter_mut().map(|(_, (full, delta))| {
                    let changed = delta.is_empty();
                    for e in delta.drain() {
                        full.insert(e);
                    }
                    changed
                })
            })
            .any(|b| b)
    }
}
