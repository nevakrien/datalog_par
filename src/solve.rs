// ================== kb builder / deduper ==================
use hashbrown::hash_map::Entry;
use hashbrown::{HashMap, HashSet};
use std::fmt;
use std::num::NonZeroU32;

use crate::parser::{Atom as RawAtom, Rule as RawRule, Statement, Term as RawTerm};

// ---- Non-zero IDs (separate namespaces) ----
#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)]
pub struct PredId(NonZeroU32);
#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)]
pub struct ConstId(NonZeroU32);

// ---- Compact term representation ----
// Vars:  0,1,2,...  (clause-local index, 0-based)
// Const: -id         (id is NonZeroU32, so negative never overlaps vars)
#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)]
pub struct Term32(pub i32);

impl Term32 {
    #[inline] pub const fn var(ix0: u32) -> Self { Self(ix0 as i32) }
    #[inline] pub const fn from_const(cid: ConstId) -> Self { Self(-(cid.0.get() as i32)) }

    #[inline] pub const fn is_var(self) -> bool { self.0 >= 0 }
    #[inline] pub const fn is_const(self) -> bool { self.0 < 0 }

    // Infallible getters with debug checks
    #[inline] pub fn var_index(self) -> u32 {
        debug_assert!(self.is_var(), "Term32::var_index called on non-var: {:?}", self);
        self.0 as u32
    }
    #[inline] pub fn const_id(self) -> ConstId {
        debug_assert!(self.is_const(), "Term32::const_id called on non-const: {:?}", self);
        let raw = (-self.0) as u32;
        debug_assert!(raw >= 1, "encoded const id must be >= 1");
        // safe: we just asserted raw >= 1
        unsafe { ConstId(NonZeroU32::new_unchecked(raw)) }
    }
}

// Public ergonomic view (optional, handy for APIs)
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Term {
    Var(u32),       // 0-based
    Const(ConstId), // interned
}

impl From<Term> for Term32 {
    #[inline]
    fn from(t: Term) -> Self {
        match t {
            Term::Var(ix) => Term32::var(ix),
            Term::Const(c) => Term32::from_const(c),
        }
    }
}

impl From<Term32> for Term {
    #[inline]
    fn from(t: Term32) -> Self {
        if t.is_var() {
            Term::Var(t.var_index())
        } else {
            Term::Const(t.const_id())
        }
    }
}

// ---- Canonicalized structures ----
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct AtomId {
    pub pred: PredId,
    pub args: Box<[Term32]>, // immutable, arity is fixed
}

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct RuleId {
    pub head: AtomId,
    pub body: Box<[AtomId]>, // immutable
}

// ---- Interner for predicates (with arity) and constants ----
#[derive(Default)]
pub struct Interner {
    pred_map: HashMap<Box<str>, PredId>,
    pred_names: Vec<Box<str>>,
    pred_arity: Vec<u32>, // fixed at first sight

    const_map: HashMap<Box<str>, ConstId>,
    const_names: Vec<Box<str>>,
}

impl Interner {
    #[inline]
    fn nz(n: u32) -> NonZeroU32 { NonZeroU32::new(n).unwrap() }

    pub fn intern_pred_with_arity(&mut self, name: &Box<str>, arity: u32) -> Result<PredId, KBError> {
        if let Some(&id) = self.pred_map.get(name) {
            let ix = (id.0.get() - 1) as usize;
            let a = self.pred_arity[ix];
            if a == arity { Ok(id) }
            else { Err(KBError::WrongArity { predicate: self.pred_names[ix].clone(), expected: a, found: arity }) }
        } else {
            let id = PredId(Self::nz(self.pred_names.len() as u32 + 1));
            self.pred_names.push(name.clone());
            self.pred_map.insert(self.pred_names.last().unwrap().clone(), id);
            self.pred_arity.push(arity);
            Ok(id)
        }
    }

    pub fn get_pred_if_known(&self, name: &Box<str>) -> Option<(PredId, u32)> {
        let &id = self.pred_map.get(name)?;
        let ix = (id.0.get() - 1) as usize;
        Some((id, self.pred_arity[ix]))
    }

    pub fn intern_const(&mut self, name: &Box<str>) -> ConstId {
        if let Some(&id) = self.const_map.get(name) { return id; }
        let id = ConstId(Self::nz(self.const_names.len() as u32 + 1));
        self.const_names.push(name.clone());
        self.const_map.insert(self.const_names.last().unwrap().clone(), id);
        id
    }

    pub fn get_const_if_known(&self, name: &Box<str>) -> Option<ConstId> {
        self.const_map.get(name).copied()
    }

    pub fn pred_name(&self, id: PredId) -> &str { self.pred_names[(id.0.get() - 1) as usize].as_ref() }
    pub fn const_name(&self, id: ConstId) -> &str { self.const_names[(id.0.get() - 1) as usize].as_ref() }
    pub fn pred_arity_of(&self, id: PredId) -> u32 { self.pred_arity[(id.0.get() - 1) as usize] }
}

// ---- Clause-local variable scope (alpha-renaming -> 0-based ints) ----
struct VarScope {
    map: HashMap<Box<str>, u32>, // var name -> 0-based index
    next: u32,                   // next index to assign (also the size)
}
impl VarScope {
    fn new() -> Self { Self { map: HashMap::new(), next: 0 } }
    #[inline] fn fresh_ix(&mut self) -> u32 { let ix = self.next; self.next += 1; ix }
    fn get_ix(&mut self, v: &Box<str>) -> u32 {
        match self.map.entry(v.clone()) {
            Entry::Occupied(o) => *o.get(),
            Entry::Vacant(vac) => { let ix = self.next; self.next += 1; vac.insert(ix); ix }
        }
    }
    fn anon_ix(&mut self) -> u32 { self.fresh_ix() }
}

// ---- Errors (Box<str>, no String) ----
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KBError {
    WrongArity { predicate: Box<str>, expected: u32, found: u32 },
    HeadVarNotBound { predicate: Box<str>, arg_idx: usize },
    UnknownPredicateInQuery { predicate: Box<str> },
    UnknownConstantInQuery { predicate: Box<str>, constant: Box<str> },
}

impl fmt::Display for KBError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use KBError::*;
        match self {
            WrongArity { predicate, expected, found } =>
                write!(f, "Predicate '{}' arity mismatch: expected {}, found {}.", predicate, expected, found),
            HeadVarNotBound { predicate, arg_idx } =>
                write!(f, "Rule {}(...): head var at position {} not bound in body.", predicate, arg_idx),
            UnknownPredicateInQuery { predicate } =>
                write!(f, "Query references unknown predicate '{}'.", predicate),
            UnknownConstantInQuery { predicate, constant } =>
                write!(f, "Query on '{}' uses unknown constant '{}'.", predicate, constant),
        }
    }
}

// ---- Producers bucket per predicate ----
#[derive(Default, Clone)]
pub struct Producers {
    pub rules: Vec<RuleId>,
    pub ground: Vec<AtomId>,
    pub generic: Vec<AtomId>,
}
impl Producers {
    #[inline] fn push_rule(&mut self, r: RuleId)   { self.rules.push(r); }
    #[inline] fn push_ground(&mut self, a: AtomId) { self.ground.push(a); }
    #[inline] fn push_generic(&mut self, a: AtomId){ self.generic.push(a); }
}

// ---- KB (single owner) ----
#[derive(Default)]
pub struct KB {
    pub(crate) inter: Interner,

    // facts
    pub(crate) ground_facts: HashSet<AtomId>,
    pub(crate) generic_facts: HashSet<AtomId>,

    // rules (dedup set)
    pub(crate) rules: HashSet<RuleId>,

    // single lookup: pred -> Producers {rules, ground, generic}
    pub(crate) producers: HashMap<PredId, Producers>,
}

impl KB {
    pub fn new() -> Self { Self::default() }
    pub fn interner(&self) -> &Interner { &self.inter }

    pub fn add_statement(&mut self, st: &Statement) -> Result<(), KBError> {
        match st {
            Statement::Fact(a) => self.add_fact_or_generic(a),
            Statement::Rule(r) => self.add_rule(r),
            Statement::Query(a) => self.validate_query(a), // queries do not mutate
        }
    }

    #[inline]
    fn producers_mut(&mut self, p: PredId) -> &mut Producers {
        self.producers.entry(p).or_default()
    }

    // ----- atoms -----
    fn canon_atom(&mut self, a: &RawAtom, vs: &mut VarScope) -> Result<AtomId, KBError> {
        let pred = self.inter.intern_pred_with_arity(&a.predicate, a.args.len() as u32)?;
        let mut args_vec: Vec<Term32> = Vec::with_capacity(a.args.len());
        for t in &a.args {
            let enc = match t {
                RawTerm::Variable(v) if v.as_ref() == "_" => Term32::var(vs.anon_ix()),
                RawTerm::Variable(v) => Term32::var(vs.get_ix(v)),
                RawTerm::Constant(c) => Term32::from_const(self.inter.intern_const(c)),
            };
            args_vec.push(enc);
        }
        Ok(AtomId { pred, args: args_vec.into_boxed_slice() })
    }

    #[inline]
    fn is_ground_atom(a: &AtomId) -> bool {
        a.args.iter().all(|&t| t.is_const())
    }

    // Parser `Fact(_)` may contain variables → treat as generic fact.
    fn add_fact_or_generic(&mut self, a: &RawAtom) -> Result<(), KBError> {
        let mut vs = VarScope::new();
        let aid = self.canon_atom(a, &mut vs)?;

        if Self::is_ground_atom(&aid) {
            if self.ground_facts.insert(aid.clone()) {
                self.producers_mut(aid.pred).push_ground(aid);
            }
        } else {
            if self.generic_facts.insert(aid.clone()) {
                self.producers_mut(aid.pred).push_generic(aid);
            }
        }
        Ok(())
    }

    // Rules must have non-empty body; if body is empty, route to fact/generic.
    fn add_rule(&mut self, r: &RawRule) -> Result<(), KBError> {
        if r.body.is_empty() {
            return self.add_fact_or_generic(&r.head);
        }

        let mut vs = VarScope::new();
        let head = self.canon_atom(&r.head, &mut vs)?;

        // dedupe body atoms while preserving order
        let mut seen = HashSet::new();
        let mut body_vec = Vec::with_capacity(r.body.len());
        for a in &r.body {
            let ca = self.canon_atom(a, &mut vs)?;
            if seen.insert(ca.clone()) { body_vec.push(ca); }
        }

        // range-restriction: each head var must appear in body
        for (i, &t) in head.args.iter().enumerate() {
            if t.is_var() {
                let vh = t;
                let mut found = false;
                'outer: for b in &body_vec {
                    for &bt in b.args.iter() {
                        if bt == vh { found = true; break 'outer; }
                    }
                }
                if !found {
                    return Err(KBError::HeadVarNotBound { predicate: r.head.predicate.clone(), arg_idx: i });
                }
            }
        }

        let rule = RuleId { head: head.clone(), body: body_vec.into_boxed_slice() };

        // global O(1) dedupe; only on insert update producers bucket
        if self.rules.insert(rule.clone()) {
            self.producers_mut(head.pred).push_rule(rule);
        }
        Ok(())
    }

    // Queries must not add info: pred+arity must exist, constants must be known.
    fn validate_query(&mut self, a: &RawAtom) -> Result<(), KBError> {
        let (_pred, expect_arity) = self.inter
            .get_pred_if_known(&a.predicate)
            .ok_or_else(|| KBError::UnknownPredicateInQuery { predicate: a.predicate.clone() })?;
        let found_arity = a.args.len() as u32;
        if expect_arity != found_arity {
            return Err(KBError::WrongArity { predicate: a.predicate.clone(), expected: expect_arity, found: found_arity });
        }
        for t in &a.args {
            if let RawTerm::Constant(c) = t {
                if self.inter.get_const_if_known(c).is_none() {
                    return Err(KBError::UnknownConstantInQuery { predicate: a.predicate.clone(), constant: c.clone() });
                }
            }
        }
        Ok(())
    }
}



// ---- Tests for KB layer ----
#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::DatalogParser;

    fn build(input: &str) -> Result<KB, KBError> {
        let mut p = DatalogParser::new(input);
        let stmts = p.parse_all().unwrap();
        let mut kb = KB::new();
        for s in &stmts { kb.add_statement(s)?; }
        Ok(kb)
    }

	#[test]
	fn producers_split_and_rule_dedupe() {
	    let kb = build(
	        "parent(tom,bob).
	         parent(bob,alice).
	         ancestor(X,Z) :- parent(X,Z).
	         ancestor(X,Z) :- parent(X,Y), parent(Y,Z).
	         ancestor(A,B) :- parent(A,C), parent(C,B)."
	    ).unwrap();

	    // rules deduped by alpha-equivalence: last two are equivalent
	    assert_eq!(kb.rules.len(), 2);

	    // producers split:
	    let (pid, _) = kb.inter.get_pred_if_known(&"ancestor".into()).unwrap();
	    let prod = kb.producers.get(&pid).expect("ancestor producers exist");
	    assert_eq!(prod.ground.len(),  0, "no ground facts for ancestor");
	    assert_eq!(prod.generic.len(), 0, "no generic facts for ancestor");
	    assert_eq!(prod.rules.len(),   2, "two distinct ancestor rules");
	}

	#[test]
	fn fact_vs_generic_fact_buckets() {
	    let kb = build(
	        "p(a,b).
	         p(X,Y).        % generic fact (variables, no body)
	         q(a).          % ground fact
	         r(X) :- .      % empty body rule becomes generic fact
	        "
	    ).unwrap();

	    let (p_pid, _) = kb.inter.get_pred_if_known(&"p".into()).unwrap();
	    let (q_pid, _) = kb.inter.get_pred_if_known(&"q".into()).unwrap();
	    let (r_pid, _) = kb.inter.get_pred_if_known(&"r".into()).unwrap();

	    // p/2: both one ground and one generic
	    let p_prod = kb.producers.get(&p_pid).expect("p producers exist");
	    assert_eq!(p_prod.ground.len(),  1);
	    assert_eq!(p_prod.generic.len(), 1);
	    assert_eq!(p_prod.rules.len(),   0);

	    // q/1: only ground
	    let q_prod = kb.producers.get(&q_pid).expect("q producers exist");
	    assert_eq!(q_prod.ground.len(),  1);
	    assert_eq!(q_prod.generic.len(), 0);
	    assert_eq!(q_prod.rules.len(),   0);

	    // r/1: came from an empty-body rule -> generic fact
	    let r_prod = kb.producers.get(&r_pid).expect("r producers exist");
	    assert_eq!(r_prod.generic.len(), 1);
	    assert_eq!(r_prod.ground.len(),  0);
	    assert_eq!(r_prod.rules.len(),   0);
	}


    #[test]
    fn query_checks_dont_add_info() {
        // Introduce parent/2 and constants
        let mut p = DatalogParser::new(
            "parent(tom,bob).
             ?- parent(tom,Who).
             ?- parent(tom,alice)."
        );
        let stmts = p.parse_all().unwrap();
        let mut b = KB::new();

        // ground fact registers pred/arity and constants {tom,bob}
        b.add_statement(&stmts[0]).unwrap();

        // query with new variable OK
        b.add_statement(&stmts[1]).unwrap();

        // query with new constant 'alice' should fail
        match b.add_statement(&stmts[2]) {
            Err(KBError::UnknownConstantInQuery { predicate, constant }) => {
                assert_eq!(predicate.as_ref(), "parent");
                assert_eq!(constant.as_ref(), "alice");
            }
            other => panic!("expected UnknownConstantInQuery, got {:?}", other),
        }
    }

    #[test]
    fn head_var_not_bound_errors() {
        let mut p = DatalogParser::new("s(X) :- t(Y).");
        let stmts = p.parse_all().unwrap();
        let mut b = KB::new();
        match b.add_statement(&stmts[0]) {
            Err(KBError::HeadVarNotBound { predicate, arg_idx }) => {
                assert_eq!(predicate.as_ref(), "s");
                assert_eq!(arg_idx, 0);
            }
            other => panic!("expected HeadVarNotBound, got {:?}", other),
        }
    }

    #[test]
    fn body_duplicate_atoms_are_removed() {
        let kb = build("p(X) :- q(X), q(X), q(X).").unwrap();
        let body = &kb.rules.iter().next().unwrap().body;
        assert_eq!(body.len(), 1);
    }
}
