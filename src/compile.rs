/*!
 * note that this module itself does not need to be performent
 * it wont be a major cost from overall runtime
*/

use crate::parser::KB;
use crate::solve::KeyGather;
use crate::parser::TermId;
use crate::solve::RuleSolver;
use hashbrown::HashSet;
use rayon::prelude::*;
use crate::parser::ConstId;
use crate::solve::SolveEngine;
use crate::magic::MagicKey;
use crate::solve::QuerySolver;
use crate::magic::KeyId;
use crate::magic::MagicSet;
use crate::solve::FullSolver;
use crate::parser::AtomId;
use crate::parser::PredId;
use crate::parser::RuleId;
use crate::parser::Term32;
use hashbrown::HashMap;

#[derive(Debug,Clone,PartialEq,Eq,Hash)]
pub struct SolvePattern{
    head:Box<[Term32]>,//always canon
    conds:Box<[AtomId]>,
}

impl SolvePattern{
    pub fn new(rule:&RuleId)->Self{
        Self{
            head:rule.head.args.clone(),
            conds:rule.body.clone(),
        }
    }

    pub fn make_solver(self,magic:&mut MagicSet)->FullSolver{
        //the start is somewhat obvious so we do it here
        let mut atom =self.conds[0].clone();
        let map = atom.canonize();
        let start = magic.register(MagicKey{
            atom,
            bounds:0
        });

        let arity = self.head.len();
        
        //special case for specifically a fetch quest
        if self.conds.len() == 1 {
            let gather = self.head.iter().filter_map(|x|{
                match x.term() {
                    TermId::Const(_c)=> None,
                    TermId::Var(v)  => Some(KeyGather::Var(map[&v])),
                }
            }).collect();
            return FullSolver{
                start,
                parts: Box::from([]),
                end_gather:Some(gather),
            }
        }

        //main loop is very tricky
        let mut bounds: u64 = 0;
        let mut parts = Vec::new();
        for (i,a) in self.conds[1..].iter().enumerate() {
            let needed = |x:&&Term32|{
                if x.is_const(){
                    return false;
                }
                if self.head.contains(&x) {
                    return true
                }

                self.conds[i+1..].iter().any(|v| v.args.contains(&x))
            };

            let needed: HashSet<_> = a.args.iter().filter(needed).collect();
            let mut atom = a.clone();
            let map = atom.canonize();

            let key_len = bounds.count_ones() as usize;
            let val_len = (!bounds & (1u64 << arity - 1)).count_ones() as usize;

            let key_gathers = Vec::with_capacity(key_len);
            let val_gathers = Vec::with_capacity(val_len);

            // todo!();
            let exists_only = val_gathers.is_empty();
            parts.push(RuleSolver{
                key_gathers: key_gathers.into(),
                val_gathers: val_gathers.into(),
                exists_only,
                keyid:todo!(),
            });
            todo!()
        }

        FullSolver{
            start,
            parts: parts.into(),
            end_gather:todo!()

        }
    }
}

pub fn full_compile(kb:&KB,query:AtomId)->QuerySolver{
    let preds = kb.trace_pred(query.pred);
    let mut db = QueryRules::new(query);
    for p in preds {
        let prod = &kb.producers[&p];
        for r in &prod.rules{
            db.add_rule(&r);
        }
        for f in &prod.facts{
            db.add_fact(f.pred,f.args.iter().map(|x| x.try_const().unwrap()).collect())
        }
    }
    db.simple_compile()
}

pub struct QueryRules {
    pub rules: HashMap<SolvePattern, Vec<PredId>>,
    pub facts: Vec<(PredId,Box<[ConstId]>)>,
    pub target: AtomId,
}

impl QueryRules {
    pub fn new(target: AtomId) -> Self {
        let rules = HashMap::new();
        Self { rules,facts:Vec::new(), target }
    }
    pub fn add_rule(&mut self, rule: &RuleId) {
        self.rules
            .entry(SolvePattern::new(rule))
            .or_default()
            .push(rule.head.pred);
    }

    pub fn add_fact(&mut self, pred:PredId,args:Box<[ConstId]>){
        self.facts.push((pred,args));
    }

    pub fn simple_compile(self) -> QuerySolver{
        let mut magic = MagicSet::new();

        let target = magic.register(
            MagicKey{
                bounds:0,
                atom:self.target
            }
        );
        magic[target].map.entry(Box::from([])).or_default();

        let solvers: Vec<_> = self.rules.into_iter().map(|(k,v)|{
            (k.make_solver(&mut magic),v)
        }).collect();

        let mut full = magic.empty_new_set();
        for (pred,cons) in &self.facts {
            let Some(new) = magic.additions(*pred,&*cons) else { continue;};
            for (k1,(k2,v)) in new {
                full[k1.0].entry(k2).or_default().insert(v);
            }
        }
        // println!("start facts {full:?}");
        magic.put_new_delta(&mut full);


        QuerySolver{
            magic,
            engine:SolveEngine{solvers},
            target
        }
    }
}

#[cfg(test)]
mod tests_compile_simple {
    use super::*; // pulls in full_compile, KB, etc.
    use crate::parser::{DatalogParser, Statement};

    /// Helper: build a KB from all non-query statements in `src`,
    /// and also return the (single) parsed query Atom (as `Statement::Query`).
    fn build_kb_and_query(src: &str) -> (KB, crate::parser::Atom) {
        let mut parser = DatalogParser::new(src);
        let statements = parser.parse_all().expect("parse_all failed");

        let mut kb = KB::new();
        let mut query_atom: Option<crate::parser::Atom> = None;

        for st in &statements {
            match st {
                Statement::Query(a) => {
                    assert!(query_atom.is_none(), "expected exactly one query");
                    query_atom = Some(a.clone());
                }
                _ => {
                    // facts/rules go straight into the KB
                    kb.add_statement(st).expect("KB add_statement failed");
                }
            }
        }

        (kb, query_atom.expect("no query found"))
    }

    #[test]
    fn single_condition_rule() {
        // Two direct parent edges out of 'tom' to check we get multiple hits.
        // Rule is single-condition: ancestor(X,Z) :- parent(X,Z).
        // Query asks for all Z such that ancestor(tom,Z).
        let src = r#"
            parent(tom, bob).
            parent(tom, charlie).
            parent(bob, alice).   % irrelevant for the single-step rule

            ancestor(X, Z) :- parent(X, Z).

            ?- ancestor(tom, Who).
        "#;

        let (kb, q_ast) = build_kb_and_query(src);

        // Encode the query against the KB (checks arity & constant interning, etc.)
        let q_enc = kb.get_query_encoding(&q_ast).expect("encode query failed");

        // Compile a QuerySolver for the query.
        let mut qs = full_compile(&kb, q_enc);
        println!("compiled solver {:?}",qs.engine);

        // Run it to completion and collect all hits.
        // Assumption: `get_all()` returns a collection of boxed ConstId slices,
        // one row per solution, in the same arity/order as the query head gather.
        let results = qs.get_all();

        // Map expected constant names to their interned ConstId for robust comparison.
        let cid = |name: &str| kb.interner().get_const_if_known(&name.into()).unwrap();
        let tom = cid("tom");
        let bob = cid("bob");
        let charlie = cid("charlie");

        // We expect exactly two rows: (tom,bob) and (tom,charlie).
        // Build a set for order-independent equality.
        let want: HashSet<Box<[ConstId]>> = [
            Box::from([tom,bob]),
            Box::from([tom,charlie]),
        ]
        .into_iter()
        .collect();

        let got: HashSet<Box<[ConstId]>> = results.into_iter().collect();

        assert_eq!(
            got, want,
        );
    }
}
