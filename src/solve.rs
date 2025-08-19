
use std::mem::transmute;
use crate::magic::MagicSet;
use crate::magic::KeyId;
use crate::parser::ConstId;
use rayon::prelude::*;

pub type QueryElem = [Option<ConstId>];

pub struct RuleSolver {
    keyid:KeyId,
    key_gathers: Box<[usize]>,
    val_gathers: Box<[(usize,usize)]>,//found -> existing
}

impl RuleSolver {
    pub fn single_apply(&self,elem:&QueryElem,magic:&MagicSet,delta:bool)->Vec<Box<QueryElem>>{
        let key: Box<[_]> = self.key_gathers.iter().map(|i| elem[*i].unwrap()).collect();
        let spot = &magic[self.keyid].map[&key];

        let s = if delta {&spot.1} else{&spot.0};
        
        //if no gathers we just check existance
        if self.val_gathers.is_empty(){
            return if s.is_empty(){
                Vec::new()
            }else{
                vec![elem.into()]
            }
        }

        //wekk we have to actually loop
        s.par_iter().map(|v|{
            let mut ans:Box<[_]> = elem.into();
            for (i,k) in self.val_gathers.iter() {
                ans[*k] = Some(v[*i]);
            }
            ans
        }).collect()
    }
}

//we kinda wish this could be removed entirly
//with a clever compiler it can be 
//if we just put the elems in the right place
pub struct FinalGather {
    gathers: Box<[(usize,usize)]>, //swaps *i=j
    trunc:usize
}

impl FinalGather {
    pub fn finalize(&self,mut elem:Box<QueryElem>)->Box<[ConstId]>{
        for (i,j) in &self.gathers{
            elem[*i]=elem[*j];
        }

        let mut v:Vec<_> = elem.into();
        v.truncate(self.trunc);

        assert!(v.iter().all(|x|x.is_some()));

        //nonzero and option non zero are the same
        //see https://doc.rust-lang.org/beta/std/num/struct.NonZero.html
        unsafe{transmute(v.into_boxed_slice())}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU32;

    // Helper: make a ConstId from a nonzero u32.
    // Adjust to your real constructor if different.
    #[inline]
    fn cid(n: u32) -> ConstId {
        assert!(n != 0, "ConstId must be nonzero");
        // If ConstId is a #[repr(transparent)] newtype around NonZeroU32,
        // this transmute in test code is fine. Replace with your real ctor
        // e.g. ConstId(NonZeroU32::new(n).unwrap()) if that exists.
        unsafe { std::mem::transmute::<NonZeroU32, ConstId>(NonZeroU32::new(n).unwrap()) }
    }

    #[test]
    fn finalize_basic_trunc_and_swaps_work() {
        // Start with [Some(1), Some(2), None, Some(4)]
        // We will swap index 2 <- index 3, then truncate to 3,
        // so the result should be [1, 2, 4].
        let gathers: Box<[(usize, usize)]> = Box::from([(2, 3)]); // elem[2] = elem[3]
        let fg = FinalGather { gathers, trunc: 3 };

        let elem: Box<[Option<ConstId>]> = Box::from([
            Some(cid(1)),
            Some(cid(2)),
            None,
            Some(cid(4)),
        ]);

        let out: Box<[ConstId]> = fg.finalize(elem);
        assert_eq!(out.len(), 3);
        assert_eq!(out, vec![cid(1), cid(2), cid(4)].into());
    }

    #[test]
    #[should_panic(expected = "all(|x| x.is_some())")]
    fn finalize_panics_if_any_none_remains() {
        // No gather to fill the None at index 2 ⇒ assert! should trip.
        let fg = FinalGather { gathers: Box::new([]), trunc: 4 };
        let elem: Box<[Option<ConstId>]> = Box::from([
            Some(cid(1)),
            Some(cid(2)),
            None,
            Some(cid(4)),
        ]);
        let _ = fg.finalize(elem);
    }
}
