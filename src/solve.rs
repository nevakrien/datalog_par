use crate::parser::Statement;
use crate::parser::DatalogParser;
use crate::parser::ParseError;
use std::sync::RwLock;
use hashbrown::HashMap;
use hashbrown::HashSet;
use crate::unique::{ListId,Registery,Store};
use crate::parser::{Term as RawTerm,Atom as RawAtom, Rule as RawRule};

pub type Cond<'a,'b1,'b2> = ListId<'b2, Atom<'a,'b1>>;
pub type Terms<'a,'b> = ListId<'b, Term<'a>>;
pub type Name<'a> = ListId<'a, u8>;

#[derive(Debug,Clone,Eq,PartialEq,Copy,Hash)]
pub enum Term<'a>{
	Variable(Name<'a>),
    Constant(Name<'a>),
}

impl<'a> Term<'a>{
	pub fn new(raw:&RawTerm,preds:&'a Registery<u8>)->Self{
		match raw{
			RawTerm::Variable(s)=>Term::Variable(preds.get_unique(s.as_bytes())),
			RawTerm::Constant(s)=>Term::Constant(preds.get_unique(s.as_bytes())),
		}	
	}
}

#[derive(Debug, Clone, PartialEq,Eq,Hash)]
pub struct Atom<'a,'b> {
    pub predicate: Name<'a>,
    pub args: Terms<'a,'b>,
}

impl<'a,'b> Atom<'a, 'b>{
	pub fn new(raw:&RawAtom,preds:&'a Registery<u8>,terms:&'b Registery<Term<'a>>)->Self{
		let predicate = preds.get_unique(raw.predicate.as_bytes());
		let args : Vec<_>= raw.args.iter().map(|x| {
			Term::new(x,preds)
		}).collect();
		let args = terms.get_unique(&args);
		Self{
			predicate,
			args
		}
	}
}

#[derive(Debug, Clone, PartialEq,Eq,Hash)]
pub struct Rule<'a,'b1,'b2> {
    pub head: Atom<'a,'b1>,
    pub body: Cond<'a,'b1,'b2>,
}

impl<'a,'b1,'b2> Rule<'a,'b1,'b2>{
	pub fn new(raw:&RawRule,
		preds:&'a Registery<u8>,
		terms:&'b1 Registery<Term<'a>>,
		atoms:&'b2 Registery<Atom<'a,'b1>>
	)->Self{
		let head = Atom::new(&raw.head,preds,terms);
		let body :Vec<_>= raw.body.iter().map(|a|{
			Atom::new(a,preds,terms)
		}).collect();
		let body = atoms.get_unique(&*body);

		Self{
			head,
			body
		}
	}
}

#[test]
fn test_store_rule() {
	use crate::parser::DatalogParser;
	use crate::parser::Statement;

    let mut parser = DatalogParser::new("grandparent(X, Z) :- parent(X, Y), parent(Y, Z).");
    let statements = parser.parse_all().unwrap();
    
    assert_eq!(statements.len(), 1);
    let Statement::Rule(ref raw) = statements[0] else { panic!() };
    let preds = Registery::default();
    let terms = Registery::default();
    let atoms = Registery::default();
    let _rule = Rule::new(&raw,&preds,&terms,&atoms);
}

pub struct Regs<'a,'b1,'b2>{
	pub preds:&'a Registery<u8>,
	pub terms:&'b1 Registery<Term<'a>>,
	pub atoms:&'b2 Registery<Atom<'a,'b1>>
}

impl<'a,'b1,'b2> Regs<'a,'b1,'b2>{
	pub fn make_atom(&self,raw:&RawAtom)->Atom<'a,'b1>{
		Atom::new(raw,&self.preds,&self.terms)
	}
	pub fn make_rule(&self,raw:&RawRule)->Rule<'a,'b1,'b2>{
		Rule::new(raw,&self.preds,&self.terms,&self.atoms)
	}
}

pub struct Given<'a,'b1,'b2>{
	pub rules:HashMap<Name<'a>,Vec<Rule<'a,'b1,'b2>>>,
	pub generic_facts:HashMap<Name<'a>,Vec<Atom<'a,'b1>>>,
	pub true_facts:HashMap<Name<'a>,Vec<Atom<'a,'b1>>>,
	pub consts:HashSet<Name<'a>>,
	pub regs:Regs<'a,'b1,'b2>
}

impl<'a,'b1,'b2> Given<'a,'b1,'b2>{
	pub fn new(regs:Regs<'a,'b1,'b2>) -> Self{
		Self{
			rules:HashMap::new(),
			generic_facts:HashMap::new(),
			true_facts:HashMap::new(),
			consts:HashSet::new(),
			regs,
		}
	}
	pub fn add_names(&mut self,names:impl Iterator<Item=Term<'a>>){
		self.consts.extend(names.filter_map(|t|{
			match t {
				Term::Constant(c)=>Some(c.clone()),
				Term::Variable(_)=>None,
			}
		}))
	}
	pub fn add_rule(&mut self,rule:&RawRule){
		//we need to keep it so facts are the only "something from nothing"
		if rule.body.is_empty(){
			return self.add_fact(&rule.head);
		}


		let rule = self.regs.make_rule(rule);
		self.add_names(rule.body.iter().flat_map(|x| x.args.iter()).cloned());
		self.add_names(rule.head.args.iter().cloned());

		self.rules.entry(rule.head.predicate).or_default().push(rule);
	}
	pub fn add_fact(&mut self,atom:&RawAtom){
		let atom = self.regs.make_atom(atom);
		self.add_names(atom.args.iter().cloned());

		if atom.args.iter().all(|x|{matches!(x,Term::Constant(_))}){
			self.generic_facts.entry(atom.predicate).or_default().push(atom);

		}else{
			self.true_facts.entry(atom.predicate).or_default().push(atom);
		}
	}

	pub fn add_till_query(&mut self,parser:&mut DatalogParser<'a>)->Result<Option<Atom<'a,'b1>>, ParseError>{
		while let Some(stmt) = parser.parse_statement()?{
			match stmt {
			    Statement::Fact(f) => self.add_fact(&f), 
			    Statement::Rule(r) => self.add_rule(&r), 
			    Statement::Query(q) => {
			    	//TODO technically we need to add the names here...
			    	//but only to the query scope which is weird
			    	//this probably is best handle in  query but still
			    	return Ok(Some(self.regs.make_atom(&q)));
			    },
			}
		}
		Ok(None)
	}
}

// pub struct QuertMeta<'a,'b1>{
// 	query:Atom<'a,'b1>,
// 	cache:Store<Name<'a>,RwLock<HashSet<Terms<'a,'b1>>>>,
// 	relvent:HashSet<Name<'a>>
// }

// impl<'a,'b1>  QuertMeta<'a,'b1>{
// 	pub fn new(query:Atom<'a,'b1>,given:&Given<'a,'b1,'_>)->Self{

// 		todo!()
// 	}
// }