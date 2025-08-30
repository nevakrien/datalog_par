pub mod compile;
pub mod magic;
pub mod parser;
pub mod solve;

use std::io;
use std::io::Write;
use hashbrown::HashSet;
use crate::parser::ConstId;
use crate::parser::Atom;
use crate::parser::Term;
use crate::parser::Statement;
use crate::compile::full_compile;
use std::fmt;
use crate::parser::KBError;
use crate::parser::ParseError;
use crate::parser::DatalogParser;
use crate::parser::KB;

#[derive(Debug,Clone,PartialEq)]
pub enum DatalogError{
	Parse(ParseError),
	KB(KBError),
}

impl fmt::Display for DatalogError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    	match self{
    		DatalogError::Parse(x)=>write!(f,"{}", x),
    		DatalogError::KB(x)=>write!(f,"{}", x),
    	}
    }
}

impl From<KBError> for DatalogError{
fn from(x: parser::KBError) -> Self { DatalogError::KB(x) }
}

impl From<ParseError> for DatalogError{
	fn from(x: parser::ParseError) -> Self { DatalogError::Parse(x) }
}

/// Unique variable names in left-to-right order.
/// Example: parent(X, Y, X) -> ["X", "Y"]
pub fn collect_var_names(atom: &Atom) -> Vec<&str> {
    let mut seen = hashbrown::HashSet::<&str>::new();
    let mut out = Vec::new();
    for t in &atom.args {
        if let Term::Variable(s) = t {
            let n: &str = &*s;
            if seen.insert(n) {
                out.push(n);
            }
        }
    }
    out
}

fn push_print_to_string(s:&mut String,var_names:&[&str],kb:&KB,ids:&[ConstId]){
	if ids.is_empty(){
		s.push_str("yes");
		return;
	}
	for (x,name) in ids.iter().zip(var_names.iter()){
		let r = kb.inter.const_name(*x);
		s.push_str(name);
		s.push_str(" = ");
		s.push_str(r);
	}
}

pub fn print_answers(var_names:&[&str],kb:&KB,set:&HashSet<Box<[ConstId]>>){
	// set.par_iter().for_each_init(|| String::new(),|s,ids|{
	let mut s = String::new();
	set.iter().for_each(|ids|{
		s.clear();	
		push_print_to_string(&mut s,&*var_names,&kb,ids);
		println!("{s}");
		io::stdout().flush().unwrap();
	});
}

pub fn run_and_print<I:Iterator<Item=char>>(parser:&mut DatalogParser<I>,kb:&mut KB) -> Result<(),DatalogError>{
	while let Some(statment) = parser.parse_statement()?{
		if let Some(query) = kb.add_statement(&statment)?{
			let Statement::Query(atom) = statment else {panic!();};
			let var_names = collect_var_names(&atom);

			let mut q = full_compile(&kb, query);
			print_answers(&*var_names,&kb,q.current());
			
			while let Some(x) = q.solve_round(){
				print_answers(&*var_names,&kb,x);
			}
		}
	}

	io::stdout().flush().unwrap();

	Ok(())
}

// lib.rs (add this near your other pub fns)
use std::{fs, process, path::Path};

/// Run all given files through the provided KB, printing results as they are found.
/// On the first error (I/O, parse, or KB error), print to stderr and exit(1).
/// Intended for CLI use;
pub fn run_many_or_exit<P: AsRef<Path>>(kb: &mut KB, files: &[P]) {
    for path in files {
        let path = path.as_ref();
        let src = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{}: {}", path.display(), e);
                process::exit(1);
            }
        };

        let mut parser = DatalogParser::new(&src);
        if let Err(e) = run_and_print(&mut parser, kb) {
            eprintln!("{}: {}", path.display(), e);
            process::exit(1);
        }
    }
}
