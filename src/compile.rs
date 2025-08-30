/*!
 * note that this module itself does not need to be performent
 * it wont be a major cost from overall runtime
*/

use crate::magic::MagicKey;
use crate::magic::MagicSet;
use crate::parser::AtomId;
use crate::parser::ConstId;
use crate::parser::KB;
use crate::parser::PredId;
use crate::parser::RuleId;
use crate::parser::Term32;
use crate::parser::TermId;
use crate::solve::FullSolver;
use crate::solve::Gather;
use crate::solve::KeyGather;
use crate::solve::QuerySolver;
use crate::solve::RuleSolver;
use crate::solve::SolveEngine;
use hashbrown::HashMap;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SolvePattern {
    head: Box<[Term32]>, //always canon
    conds: Box<[AtomId]>,
}

impl SolvePattern {
    pub fn new(rule: &RuleId) -> Self {
        Self {
            head: rule.head.args.clone(),
            conds: rule.body.clone(),
        }
    }

    pub fn make_solver(self, magic: &mut MagicSet) -> FullSolver {
        //the start is somewhat obvious so we do it here
        let mut atom = self.conds[0].clone();
        let map = atom.canonize();
        let start = magic.register(MagicKey { atom, bounds: 0 });
        println!("starting magic set {start:?}");

        //special case for specifically a fetch quest
        if self.conds.len() == 1 {
            let gather = self
                .head
                .iter()
                .filter_map(|x| match x.term() {
                    TermId::Const(_c) => None,
                    TermId::Var(v) => Some(KeyGather::Var(map[&v])),
                })
                .collect();
            return FullSolver {
                start,
                parts: Box::from([]),
                end_gather: Some(gather),
            };
        }

        //main loop is very tricky
        let mut parts = Vec::new();
        let mut var_map = BTreeMap::<u32, usize>::new();

        //we know all the vars are in the first query are unbound
        //this means we have them in order for our map
        for t in self.conds[0].args.iter() {
            let Some(v) = t.try_var() else { continue };
            let pos = var_map.len();
            var_map.entry(v).or_insert(pos);
        }

        let need_end_gather = !self.head.iter().all(|x| x.is_var());

        for (i, a) in self.conds[1..].iter().enumerate() {
            let i = i + 1; //adjust proper

            let mut atom = a.clone();
            let map = atom.canonize();
            let mut bounded_vars = 0u64;

            let key_gathers = a
                .args
                .iter()
                .filter_map(|x| match x.term() {
                    TermId::Const(c) => Some(KeyGather::Const(c)),
                    TermId::Var(v) => {
                        let id = map[&v];
                        if (bounded_vars >> id) & 1 == 1 {
                            return None;
                        }
                        let loc = var_map.get(&v)?;
                        bounded_vars |= 1u64 << id;
                        Some(KeyGather::Var(*loc as u32))
                    }
                })
                .collect();

            //move the var based bounded to location based
            let mut bounds = 0;
            for (i, t) in atom.args.iter().enumerate() {
                if let Some(v) = t.try_var() {
                    if (bounded_vars >> v) & 1 == 1 {
                        bounds |= 1u64 << i
                    }
                }
            }

            //magic key would be made in the end
            let mut val_gathers: Vec<_>;

            if (i + 1 < self.conds.len()) || need_end_gather {
                //make sure we take stuff we need
                let needed = |x: u32| {
                    let x: Term32 = TermId::Var(x).into();
                    if self.head.contains(&x) {
                        return true;
                    }
                    self.conds[i + 1..].iter().any(|v| v.args.contains(&x))
                };

                //first get all the vars we already had
                let mut new_varmap = BTreeMap::new();
                val_gathers = var_map
                    .into_iter()
                    .filter(|(id, _)| needed(*id))
                    .enumerate()
                    .map(|(new_loc, (id, old_loc))| {
                        new_varmap.insert(id, new_loc);
                        Gather::Exists(old_loc as u32)
                    })
                    .collect();
                var_map = new_varmap;

                //now add what we found
                let mut loc_in_val = -1isize;
                for t in a.args.iter() {
                    let Some(id) = t.try_var() else {
                        continue;
                    };

                    if (bounded_vars >> map[&id]) & 1 == 1 {
                        continue;
                    }
                    loc_in_val += 1;

                    if needed(id) {
                        if var_map.insert(id, val_gathers.len()).is_none() {
                            val_gathers.push(Gather::Found(loc_in_val as u32))
                        }
                    }
                }
            } else {
                val_gathers = self
                    .head
                    .iter()
                    .map(|x| {
                        let id = x.try_var().unwrap();
                        if let Some(loc) = var_map.get(&id) {
                            Gather::Exists(*loc as u32)
                        } else {
                            Gather::Found(u32::MAX)
                        }
                    })
                    .collect();
                var_map.clear();

                //now add what we found
                let mut loc_in_val = -1isize;
                for t in a.args.iter() {
                    let Some(id) = t.try_var() else {
                        continue;
                    };

                    if (bounded_vars >> map[&id]) & 1 == 1 {
                        continue;
                    }
                    loc_in_val += 1;

                    for (i, spot) in self.head.iter().enumerate() {
                        if id == spot.try_var().unwrap() {
                            val_gathers[i] = Gather::Found(loc_in_val as u32);
                        }
                    }
                }
            }

            //now we can get the key
            let keyid = magic.register(MagicKey { atom, bounds });

            let exists_only = val_gathers.iter().all(|x| !matches!(x, Gather::Found(_)));
            parts.push(RuleSolver {
                key_gathers: key_gathers,
                val_gathers: val_gathers.into(),
                exists_only,
                keyid,
            });

            println!("solver {:?}", parts.last().unwrap());

            // todo!()
        }
        let end_gather = if !need_end_gather {
            None
        } else {
            todo!()
            // Some(self.head.iter().map(todo!()).collect())
        };

        FullSolver {
            start,
            parts: parts.into(),
            end_gather,
        }
    }
}

pub fn full_compile(kb: &KB, query: AtomId) -> QuerySolver {
    let preds = kb.trace_pred(query.pred);
    let mut db = QueryRules::new(query);
    for p in preds {
        let prod = &kb.producers[&p];
        for r in &prod.rules {
            db.add_rule(&r);
        }
        for f in &prod.facts {
            db.add_fact(
                f.pred,
                f.args.iter().map(|x| x.try_const().unwrap()).collect(),
            )
        }
    }
    db.simple_compile()
}

pub struct QueryRules {
    pub rules: HashMap<SolvePattern, Vec<PredId>>,
    pub facts: Vec<(PredId, Box<[ConstId]>)>,
    pub target: AtomId,
}

impl QueryRules {
    pub fn new(target: AtomId) -> Self {
        let rules = HashMap::new();
        Self {
            rules,
            facts: Vec::new(),
            target,
        }
    }
    pub fn add_rule(&mut self, rule: &RuleId) {
        self.rules
            .entry(SolvePattern::new(rule))
            .or_default()
            .push(rule.head.pred);
    }

    pub fn add_fact(&mut self, pred: PredId, args: Box<[ConstId]>) {
        self.facts.push((pred, args));
    }

    pub fn simple_compile(self) -> QuerySolver {
        let mut magic = MagicSet::new();

        let target = magic.register(MagicKey {
            bounds: 0,
            atom: self.target,
        });
        magic[target].map.entry(Box::from([])).or_default();

        let solvers: Vec<_> = self
            .rules
            .into_iter()
            .map(|(k, v)| (k.make_solver(&mut magic), v))
            .collect();

        let mut full = magic.empty_new_set();
        for (pred, cons) in &self.facts {
            let Some(new) = magic.additions(*pred, &*cons) else {
                continue;
            };
            for (k1, (k2, v)) in new {
                full[k1.0].entry(k2).or_default().insert(v);
            }
        }
        // println!("start facts {full:?}");
        magic.put_new_delta(&mut full);

        QuerySolver {
            magic,
            engine: SolveEngine { solvers },
            target,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*; // pulls in full_compile, KB, etc.
    use crate::parser::DatalogParser;
    use hashbrown::HashSet;

    /// Helper: build a KB from all non-query statements in `src`,
    /// and also return the (single) parsed query Atom (as `Statement::Query`).
    fn build_kb_and_query(src: &str) -> (KB, crate::parser::AtomId) {
        let mut parser = DatalogParser::new(src);
        let statements = parser.parse_all().expect("parse_all failed");

        let mut kb = KB::new();
        let mut query_atom = None;

        for st in &statements {
            match kb.add_statement(st).expect("KB add_statement failed") {
                Some(a) => {
                    assert!(query_atom.is_none(), "expected exactly one query");
                    query_atom = Some(a);
                }
                _ => {
                    // facts/rules go straight into the KB
                }
            }
        }

        (kb, query_atom.expect("no query found"))
    }

    #[test]
    fn two_step_ancestor_rule() {
        // Ancestor(X,Z) :- parent(X,Y), parent(Y,Z).
        // Query binds X=tom and asks for all Z; head has only variables.
        let src = r#"
            parent(tom, bob).
            parent(bob, alice).

            parent(tom, charlie).
            parent(charlie, dana).

            ancestor(X, Z) :- parent(X, Y), parent(Y, Z).

            ?- ancestor(tom, Who).
        "#;

        let (kb, q_enc) = build_kb_and_query(src);

        let mut qs = full_compile(&kb, q_enc);
        let results = qs.get_all();

        let cid = |name: &str| kb.interner().get_const_if_known(&name.into()).unwrap();
        let tom = cid("tom");
        let alice = cid("alice");
        let dana = cid("dana");

        // Expect the two grand-children of 'tom'.
        let want: HashSet<Box<[ConstId]>> = [Box::from([tom, alice]), Box::from([tom, dana])]
            .into_iter()
            .collect();

        let got: HashSet<Box<[ConstId]>> = results.into_iter().collect();
        assert_eq!(got, want);
    }

    #[test]
    fn six_ary_chain_rule_head_has_5plus_args() {
        // Path of length 5 (6 nodes) with a 6-arity head, all variables.
        //
        // path6(A,B,C,D,E,F) :- edge(A,B), edge(B,C), edge(C,D), edge(D,E), edge(E,F).
        // Query binds the first to n1 and asks for the rest.
        let src = r#"
            edge(n1, n2).
            edge(n2, n3).
            edge(n3, n4).
            edge(n4, n5).
            edge(n5, n6).

            % add a bit of unrelated noise
            edge(x1, x2).
            edge(x2, x3).

            path6(A,B,C,D,E,F) :-
                edge(A,B), edge(B,C), edge(C,D), edge(D,E), edge(E,F).

            ?- path6(n1, B, C, D, E, Who).
        "#;

        let (kb, q_enc) = build_kb_and_query(src);

        let mut qs = full_compile(&kb, q_enc);
        let results = qs.get_all();

        let cid = |name: &str| kb.interner().get_const_if_known(&name.into()).unwrap();
        let n1 = cid("n1");
        let n2 = cid("n2");
        let n3 = cid("n3");
        let n4 = cid("n4");
        let n5 = cid("n5");
        let n6 = cid("n6");

        // Exactly one 6-tuple should match the bound first arg and the unique chain.
        let want: HashSet<Box<[ConstId]>> =
            [Box::from([n1, n2, n3, n4, n5, n6])].into_iter().collect();

        let got: HashSet<Box<[ConstId]>> = results.into_iter().collect();
        assert_eq!(
            got, want,
            "expected the single length-5 chain from n1 to n6"
        );
    }

    //===========REAPET VARS
    #[test]
    fn repeated_vars_in_head_simple_positive() {
        // dup(X,X) :- parent(X,Y).
        // Query asks for dup(tom,tom). Should succeed with exactly one row.
        let src = r#"
            parent(tom, bob).
            parent(bob, alice).

            dup(X, X) :- parent(X, Y).

            ?- dup(tom, tom).
        "#;

        let (kb, q_enc) = build_kb_and_query(src);

        let mut qs = full_compile(&kb, q_enc);
        let results = qs.get_all();

        let cid = |name: &str| kb.interner().get_const_if_known(&name.into()).unwrap();
        let tom = cid("tom");

        let want: HashSet<Box<[ConstId]>> = [Box::from([tom, tom])]
            .into_iter()
            .collect();
        let got: HashSet<Box<[ConstId]>> = results.into_iter().collect();

        assert_eq!(got, want, "dup(tom,tom) should be derivable");
    }

    #[test]
    fn repeated_vars_in_head_simple_negative_mismatch() {
        // dup(X,X) :- parent(X,Y).
        // Query dup(tom,bob) must fail (head enforces equality X = X).
        let src = r#"
            parent(tom, bob).

            dup(X, X) :- parent(X, Y).

            ?- dup(tom, bob).
        "#;

        let (kb, q_enc) = build_kb_and_query(src);

        let mut qs = full_compile(&kb, q_enc);
        let results = qs.get_all();

        assert!(
            results.is_empty(),
            "dup(tom,bob) must not succeed; repeated head vars enforce equality"
        );
    }

    //============EASY===============================
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

        let (kb, q_enc) = build_kb_and_query(src);

        // Compile a QuerySolver for the query.
        let mut qs = full_compile(&kb, q_enc);
        println!("compiled solver {:?}", qs.engine);

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
        let want: HashSet<Box<[ConstId]>> = [Box::from([tom, bob]), Box::from([tom, charlie])]
            .into_iter()
            .collect();

        let got: HashSet<Box<[ConstId]>> = results.into_iter().collect();

        assert_eq!(got, want,);
    }

    #[test]
    fn nullary_head_two_body_atoms_true() {
        // We have at least one chain tom -> bob -> alice,
        // so grandparent_exists should derive (boolean true).
        // Multiple chains should NOT create multiple rows: set semantics collapse to one empty tuple.
        let src = r#"
            parent(tom, bob).
            parent(bob, alice).

            % extra noise shouldn't matter
            parent(tom, charlie).
            parent(charlie, dana).

            grandparent_exists :- parent(X, Y), parent(Y, Z).

            ?- grandparent_exists.
        "#;

        let (kb, q_enc) = build_kb_and_query(src);

        let mut qs = full_compile(&kb, q_enc);
        let results = qs.get_all();

        // For a nullary query, a true result is represented as a single empty-tuple row.
        let want: HashSet<Box<[ConstId]>> = [Box::<[ConstId]>::from([])].into_iter().collect();

        let got: HashSet<Box<[ConstId]>> = results.into_iter().collect();

        assert_eq!(got, want, "expected boolean success (one empty-tuple row)");
    }

    #[test]
    fn nullary_head_two_body_atoms_false() {
        // No Y such that parent(X,Y) and parent(Y,Z) both hold,
        // so grandparent_exists should NOT derive (boolean false).
        let src = r#"
            parent(tom, bob).
            % missing the 'parent(bob, _)' link, so no chain completes

            grandparent_exists :- parent(X, Y), parent(Y, Z).

            ?- grandparent_exists.
        "#;

        let (kb, q_enc) = build_kb_and_query(src);

        let mut qs = full_compile(&kb, q_enc);
        let results = qs.get_all();

        let got: HashSet<Box<[ConstId]>> = results.into_iter().collect();

        assert!(
            got.is_empty(),
            "expected boolean failure (no rows) when no X,Y,Z satisfy the two restrictions"
        );
    }
}
