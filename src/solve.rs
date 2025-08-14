use crate::unique::{ListId,Registery};
use crate::parser::{Term,Atom as RawAtom, Rule as RawRule};

#[derive(Debug, Clone, PartialEq,Eq,Hash)]
pub struct Atom<'a> {
    pub predicate: ListId<'a, u8>,
    pub args: ListId<'a, Term>,
}

impl Copy for Atom<'_> {}

impl<'a> Atom<'a>{
	pub fn new(raw:&RawAtom,preds:&'a Registery<u8>,terms:&'a Registery<Term>)->Self{
		let predicate = preds.get_unique(raw.predicate.as_bytes());
		let args = terms.get_unique(&raw.args);
		Self{
			predicate,
			args
		}
	}
}


#[derive(Debug, Clone, PartialEq,Eq,Hash)]
pub struct Rule<'a,'b> {
    pub head: Atom<'a>,
    pub body: ListId<'b, Atom<'a>>,
}

impl Copy for Rule<'_, '_> {}


impl<'a,'b> Rule<'a,'b>{
	pub fn new(raw:&RawRule,
		preds:&'a Registery<u8>,
		terms:&'a Registery<Term>,
		atoms:&'b Registery<Atom<'a>>
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