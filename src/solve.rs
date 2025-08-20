
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
}

pub struct RuleSolver {
    keyid:KeyId,
    key_gathers: Box<[usize]>,
    val_gathers: Box<[Gather]>,//found -> existing (empty implies return the input)
}

impl RuleSolver {
    pub fn apply(&self,elems:&HashSet<Box<QueryElem>>,magic:&MagicSet,delta:bool)->HashSet<Box<QueryElem>>{
        elems.par_iter().flat_map(|elem|{
            let elem = &**elem;
            let key: Box<[_]> = self.key_gathers.iter().map(|i| elem[*i]).collect();
            let spot = &magic[self.keyid].map[&key];

            let s = if delta {&spot.1} else{&spot.0};
            
            //if no gathers we just check existance
            if self.val_gathers.is_empty(){
                return Either::Left(
                    if s.is_empty(){
                        Either::Left(rayon::iter::empty())
                }else{
                    Either::Right(rayon::iter::once(elem.into()))
                })
            }

            //we we have to actually loop
            Either::Right(s.par_iter().map(|v|{
                self.val_gathers.iter().map(|g|
                    match g{
                        Gather::Exists(u)=>elem[*u as usize],
                        Gather::Found(u)=>v[*u as usize],
                    }
                ).collect()
            }))
        }).collect()
    }
}

pub struct FullSolver {
    parts:Box<[RuleSolver]>
}

impl FullSolver {
    fn _apply(&self,elems:&HashSet<Box<QueryElem>>,magic:&MagicSet,used_delta:bool,i:usize)->HashSet<Box<QueryElem>>{
        if i==self.parts.len()-1{
            let mut ans = self.parts[i].apply(elems,magic,true);
            if used_delta {
                ans.extend(self.parts[i].apply(elems,magic,false));
            }
            return ans
        }
        todo!()
    }
    pub fn apply(&self,elems:&HashSet<Box<QueryElem>>,magic:&MagicSet)->HashSet<Box<QueryElem>>{
        //TODO make this not go over all elements
        let mut work = self.parts[0].apply(elems,magic,false);
        let delta = self.parts[0].apply(&work,magic,true);
        work.extend(delta);

        for p in self.parts[1..].iter(){
            let delta = p.apply(&work,magic,true);
            work = p.apply(&work,magic,false);
            work.extend(delta);
        }

        work
    }
}