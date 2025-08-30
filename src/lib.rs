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
	debug_assert!(!var_names.is_empty(), "bindings printer called for ground query");
	debug_assert!(var_names.len()==ids.len());
	let end = var_names.len()-1;
	for (i,(x,name)) in ids.iter().zip(var_names.iter()).enumerate(){
		let r = kb.inter.const_name(*x);
		s.reserve(name.len()+r.len()+3);

		s.push_str(name);
		s.push_str(" = ");
		s.push_str(r);

		if i!=end{
			s.push_str(", ")
		}
	}
}

pub fn print_answers_to<F:FnMut(&str)>(mut callback:F,first:&mut bool,var_names:&[&str],kb:&KB,set:&HashSet<Box<[ConstId]>>){
	let mut s = String::new();
	set.iter().for_each(|ids|{
		s.clear();	
		if *first {
			*first = false;
		}else{
			s.push_str(";\n");
		}
		push_print_to_string(&mut s,&*var_names,&kb,ids);
		callback(&mut s);
		io::stdout().flush().unwrap();
	});
}

pub fn run_and_print_to<F:FnMut(&str),I:Iterator<Item=char>>(mut callback:F,parser:&mut DatalogParser<I>,kb:&mut KB) -> Result<(),DatalogError>{
	'outer:while let Some(statment) = parser.parse_statement()?{
		if let Some(query) = kb.add_statement(&statment)?{
			let Statement::Query(atom) = statment else {panic!();};
			let var_names = collect_var_names(&atom);

			let mut q = full_compile(&kb, query);

			let mut first = true;
			if var_names.is_empty(){
				if !q.current().is_empty(){
					callback("true.\n");
					continue 'outer; 
				}
			}else{
				print_answers_to(&mut callback,&mut first,&*var_names,&kb,q.current());
			}
			
			while let Some(x) = q.solve_round(){
				if var_names.is_empty(){
					if !x.is_empty(){
						callback("true.\n");
						continue 'outer; 
					}
				}else{
					print_answers_to(&mut callback,&mut first,&*var_names,&kb,x);
				}
			}

			if var_names.is_empty(){
				callback("false.\n");
			}else{
				if !first{
					callback(".\n");
				}else{
					callback("false.\n")
				}
			}
		}
	}

	io::stdout().flush().unwrap();

	Ok(())
}

pub fn run_and_print<I:Iterator<Item=char>>(parser:&mut DatalogParser<I>,kb:&mut KB) -> Result<(),DatalogError>{
	run_and_print_to(|x| print!("{x}"),parser,kb)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn run_capture(src: &str) -> String {
        let mut out = String::new();
        let mut kb = KB::new();
        let mut parser = DatalogParser::new(src);
        run_and_print_to(|s| out.push_str(s), &mut parser, &mut kb).unwrap();
        out
    }

    /// Split the entire engine output into per-query blocks, in order.
    /// A block is either "true.\n", "false.\n", or a non-ground listing ending with ".\n".
    fn split_queries(out: &str) -> Vec<String> {
        let mut blocks = Vec::<String>::new();
        let mut cur = String::new();
        let mut i = 0;
        let bytes = out.as_bytes();

        while i < bytes.len() {
            // fast-path check for "true.\n" / "false.\n" / ".\n"
            if out[i..].starts_with("true.\n") {
                blocks.push("true.\n".to_string());
                i += "true.\n".len();
                continue;
            }
            if out[i..].starts_with("false.\n") {
                blocks.push("false.\n".to_string());
                i += "false.\n".len();
                continue;
            }
            if out[i..].starts_with(".\n") {
                cur.push_str(".\n");
                blocks.push(std::mem::take(&mut cur));
                i += 2;
                continue;
            }

            // otherwise, consume one UTF-8 char into current block
            let ch = out[i..].chars().next().unwrap();
            cur.push(ch);
            i += ch.len_utf8();
        }

        // defensive: no dangling partial block
        assert!(cur.is_empty(), "unterminated query block: {cur:?}");
        blocks
    }

    /// Parse a non-ground block (must end with ".\n") into a Vec of binding lines,
    /// ignoring ordering. The block format is: first line (no trailing \n),
    /// then zero or more of ";\n<line>", and finally ".\n".
    fn bindings_from_block(block: &str) -> Vec<String> {
        assert!(block.ends_with(".\n"), "non-ground block must end with .\\n: {block:?}");
        let body = &block[..block.len() - 2]; // strip ".\n"
        body.split(";\n").map(|s| s.to_string()).collect()
    }

    #[test]
    fn full_end_to_end_iso_compliance() {
        let program = r#"
        % --- Facts ---
        parent(alice, bob).
        parent(bob, charlie).
        parent(alice, dana).
        parent(dana, elaine).

        % --- Rules ---
        grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        ancestor(X, Z) :- parent(X, Z).
        ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

        % --- Queries ---
        ?- grandparent(alice, charlie).
        ?- grandparent(alice, elaine).
        ?- ancestor(alice, elaine).
        ?- ancestor(bob, elaine).
        ?- ancestor(bob, X).
        ?- ancestor(X, Y).
        ?- ancestor(alice, alice).
        ?- parent(charlie, X).
        ?- ancestor(X, X).
        "#;

        let output = run_capture(program);
        let blocks = split_queries(&output);

        // We expect 9 query outputs, in this fixed order.
        assert_eq!(blocks.len(), 9, "unexpected number of query outputs: {blocks:#?}");

        // 1 grandparent(alice, charlie).  -> true.
        assert_eq!(blocks[0], "true.\n");

        // 2 grandparent(alice, elaine).   -> true.
        assert_eq!(blocks[1], "true.\n");

        // 3 ancestor(alice, elaine).      -> true.
        assert_eq!(blocks[2], "true.\n");

        // 4 ancestor(bob, elaine).        -> false.
        assert_eq!(blocks[3], "false.\n");

        // 5 ancestor(bob, X).             -> single solution
        assert_eq!(blocks[4], "X = charlie.\n");

        // 6 ancestor(X, Y).               -> multiple solutions, order not guaranteed
        let got = bindings_from_block(&blocks[5]);
        let got_set: std::collections::BTreeSet<_> = got.into_iter().collect(); // order-insensitive
        let expected: std::collections::BTreeSet<_> = [
            "X = alice, Y = bob",
            "X = alice, Y = dana",
            "X = dana, Y = elaine",
            "X = bob, Y = charlie",
            "X = alice, Y = elaine",
            "X = alice, Y = charlie",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect();

        assert_eq!(got_set, expected, "ancestor(X,Y) bindings mismatch");

        // Additionally ensure separators are correct: each non-final binding preceded by ';\n'
        // (That is implied by the block structure we parsed; still, assert at least 5 separators)
        assert!(
            blocks[5].matches(";\n").count() >= 5,
            "expected at least 5 ';\n' separators in multi-solution block: {}",
            blocks[5]
        );
        assert!(blocks[5].ends_with(".\n"));

        // 7 ancestor(alice, alice).       -> false.
        assert_eq!(blocks[6], "false.\n");

        // 8 parent(charlie, X).           -> false.
        assert_eq!(blocks[7], "false.\n");

        // 9 ancestor(X, X).               -> false.
        assert_eq!(blocks[8], "false.\n");
    }
}
