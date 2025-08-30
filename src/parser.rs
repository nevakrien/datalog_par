//!standard https://www.swi-prolog.org/pldoc/man?section=isosyntax
use std::fmt;
use std::str::Chars;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term {
    Variable(Box<str>),
    Constant(Box<str>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Atom {
    pub predicate: Box<str>,
    pub args: Vec<Term>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Rule {
    pub head: Atom,
    pub body: Vec<Atom>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Statement {
    Fact(Atom),
    Rule(Rule),
    Query(Atom),
}

#[derive(Debug, PartialEq, Clone)]
pub enum ParseError {
    UnexpectedToken(Box<str>),
    UnexpectedEof,
    InvalidSyntax(Box<str>),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ParseError::UnexpectedToken(token) => write!(f, "Unexpected token: {}", token),
            ParseError::UnexpectedEof => write!(f, "Unexpected end of input"),
            ParseError::InvalidSyntax(msg) => write!(f, "Invalid syntax: {}", msg),
        }
    }
}

/// A simple two-char lookahead without requiring `Clone`.
#[derive(Debug,Clone)]
pub struct Lookahead2<I: Iterator<Item = char>> {
    iter: I,
    buf0: Option<char>, // next()
    buf1: Option<char>, // next-after-next
}

impl<I: Iterator<Item = char>> Lookahead2<I> {
    pub fn new(iter: I) -> Self {
        Self { iter, buf0: None, buf1: None }
    }

    #[inline]
    fn fill0(&mut self) {
        if self.buf0.is_none() {
            self.buf0 = self.iter.next();
        }
    }
    #[inline]
    fn fill1(&mut self) {
        self.fill0();
        if self.buf0.is_some() && self.buf1.is_none() {
            self.buf1 = self.iter.next();
        }
    }

    /// Peek the next character (like Peekable::peek)
    #[inline]
    pub fn peek(&mut self) -> Option<char> {
        self.fill0();
        self.buf0
    }

    /// Peek the character after next; fills both lookahead slots as needed.
    #[inline]
    pub fn peek2(&mut self) -> Option<char> {
        self.fill1();
        self.buf1
    }

    /// Consume and return the next character.
    #[inline]
    pub fn next(&mut self) -> Option<char> {
        self.fill0();
        let out = self.buf0.take();
        if let Some(c1) = self.buf1.take() {
            // shift second lookahead down to first
            self.buf0 = Some(c1);
        }
        out
    }
}

pub struct DatalogParser<I:Iterator<Item = char>> {
    chars: Lookahead2<I>,
    current_token: Option<String>,
}

impl<'a> DatalogParser<Chars<'a>> {
    pub fn new(input: &'a str) -> Self {
        let mut parser = DatalogParser {
            chars: Lookahead2::new(input.chars()),
            current_token: None,
        };
        parser.advance_token(); // Initialize with first token
        parser
    }
}
impl<I:Iterator<Item = char>> DatalogParser<I>{
    fn advance_token(&mut self) {
        self.current_token = self.next_token();
    }

    #[inline]
    fn skip_line_comment(&mut self) {
        while let Some(ch) = self.chars.peek() {
            if ch == '\n' || ch == '\r' {
                break;
            }
            self.chars.next();
        }
    }

    #[inline]
    fn skip_block_comment(&mut self) {
        // Non-nesting: consume until the next "*/" or EOF
        let mut prev = '\0';
        while let Some(ch) = self.chars.next() {
            if prev == '*' && ch == '/' {
                return; // closed
            }
            prev = ch;
        }
        // EOF: treat as closed (parser will likely hit UnexpectedEof later if needed)
    }

    fn skip_ws_and_comments(&mut self) {
        loop {
            // skip whitespace
            while let Some(ch) = self.chars.peek() {
                if ch.is_whitespace() {
                    self.chars.next();
                } else {
                    break;
                }
            }

            // check for comments: %, //, /* ... */
            match self.chars.peek() {
                Some('%') => {
                    // consume '%' and the rest of the line
                    self.chars.next();
                    self.skip_line_comment();
                    continue;
                }
                Some('/') => {
                    match self.chars.peek2() {
                        Some('/') => {
                            // consume both slashes, then rest of line
                            self.chars.next();
                            self.chars.next();
                            self.skip_line_comment();
                            continue;
                        }
                        Some('*') => {
                            // consume "/*" and skip until "*/"
                            self.chars.next();
                            self.chars.next();
                            self.skip_block_comment();
                            continue;
                        }
                        _ => { /* not a comment */ }
                    }
                }
                _ => { /* not a comment */ }
            }
            break;
        }
    }

    fn next_token(&mut self) -> Option<String> {
        // Skip whitespace and comments repeatedly
        self.skip_ws_and_comments();

        // End of input?
        let first_char = self.chars.next()?;

        match first_char {
            '(' | ')' | ',' | '.' => Some(first_char.to_string()),
            ':' => {
                if self.chars.peek() == Some('-') {
                    self.chars.next(); // consume '-'
                    Some(":-".to_string())
                } else {
                    // ':' is part of an identifier, collect the full token
                    let mut token = String::new();
                    token.push(':');
                    self.collect_identifier_chars(&mut token)
                }
            }
            '?' => {
                if self.chars.peek() == Some('-') {
                    self.chars.next(); // consume '-'
                    Some("?-".to_string())
                } else {
                    let mut token = String::new();
                    token.push('?');
                    self.collect_identifier_chars(&mut token)
                }
            }
            _ => {
                let mut token = String::new();
                token.push(first_char);
                self.collect_identifier_chars(&mut token)
            }
        }
    }

    fn collect_identifier_chars(&mut self, token: &mut String) -> Option<String> {
        while let Some(ch) = self.chars.peek() {
            match ch {
                ' ' | '\t' | '\n' | '\r' | '(' | ')' | ',' | '.' => break,
                ':' | '?' => {
                    // Check if this starts a special token (:- or ?-)
                    if let Some(next_ch) = self.chars.peek() {
                        if next_ch == '-' {
                            break;
                        }
                    }
                    token.push(ch);
                    self.chars.next();
                }
                _ => {
                    token.push(ch);
                    self.chars.next();
                }
            }
        }
        Some(token.clone())
    }

    fn current_token(&self) -> Option<&str> {
        self.current_token.as_deref()
    }

    fn advance(&mut self) -> Option<&str> {
        self.advance_token();
        self.current_token()
    }

    fn expect_token(&mut self, expected: &str) -> Result<(), ParseError> {
        match self.current_token() {
            Some(token) if token == expected => {
                self.advance();
                Ok(())
            }
            Some(token) => Err(ParseError::UnexpectedToken(token.into())),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn parse_term(&mut self) -> Result<Term, ParseError> {
        match self.current_token() {
            Some(token) => {
                let first = token.chars().next().unwrap();
                let is_var = first == '_' || first.is_uppercase();
                let term = if is_var {
                    Term::Variable(token.into())
                } else {
                    Term::Constant(token.into())
                };
                self.advance();
                Ok(term)
            }
            None => Err(ParseError::UnexpectedEof),
        }
    }


    fn parse_atom(&mut self) -> Result<Atom, ParseError> {
        // Read functor / atom name
        let predicate = match self.current_token() {
            Some(token) => {
                // ISO: atoms (predicate symbols) start with lowercase (or are quoted).
                // Your lexer doesn't support quoted atoms yet, so check lowercase here.
                let first = token.chars().next().unwrap_or('\0');
                if first.is_uppercase() {
                    return Err(ParseError::InvalidSyntax(
                        format!("Predicate/atom must start with lowercase (ISO): got '{token}'")
                            .into(),
                    ));
                }
                let pred: Box<str> = token.into();
                self.advance();
                pred
            }
            None => return Err(ParseError::UnexpectedEof),
        };

        // If no '(' follows, this is a nullary atom: ISO form `p` (not `p()`).
        match self.current_token() {
            Some(tok) if tok == "(" => { /* parse non-empty arg list below */ }
            _ => {
                return Ok(Atom {
                    predicate,
                    args: Vec::new(),
                });
            }
        }

        // We saw '(' → parse >= 1 arguments; `p()` is NOT ISO.
        self.advance(); // consume '('

        // Reject `p()` per ISO (compound terms require arity >= 1)
        if let Some(tok) = self.current_token() {
            if tok == ")" {
                return Err(ParseError::InvalidSyntax(
                    "Nullary predicate must be written without parentheses: use `p`, not `p()` (ISO)".into(),
                ));
            }
        } else {
            return Err(ParseError::UnexpectedEof);
        }

        // Parse first argument
        let mut args = Vec::new();
        args.push(self.parse_term()?);

        // Parse remaining arguments until ')'
        while let Some(tok) = self.current_token() {
            if tok == ")" {
                self.advance();
                break;
            } else if tok == "," {
                self.advance();
                args.push(self.parse_term()?);
            } else {
                return Err(ParseError::UnexpectedToken(tok.into()));
            }
        }

        Ok(Atom { predicate, args })
    }

    pub fn parse_statement(&mut self) -> Result<Option<Statement>, ParseError> {
        if self.current_token().is_none() {
            return Ok(None);
        }

        // Check for query
        if let Some(token) = self.current_token() {
            if token == "?-" {
                self.advance();
                let query_atom = self.parse_atom()?;
                self.expect_token(".")?;
                return Ok(Some(Statement::Query(query_atom)));
            }
        }

        // Parse atom (could be fact or head of rule)
        let atom = self.parse_atom()?;

        // What follows?
        match self.current_token() {
            Some(token) if token == "." => {
                self.advance();
                Ok(Some(Statement::Fact(atom)))
            }
            Some(token) if token == ":-" => {
                self.advance();

                // Allow EMPTY BODY: ":- ."
                if let Some(tok) = self.current_token() {
                    if tok == "." {
                        self.advance();
                        return Ok(Some(Statement::Rule(Rule {
                            head: atom,
                            body: Vec::new(),
                        })));
                    }
                }

                // Otherwise, parse one-or-more atoms separated by commas, terminated by '.'
                let mut body = Vec::new();
                body.push(self.parse_atom()?);

                while let Some(token) = self.current_token() {
                    if token == "." {
                        self.advance();
                        break;
                    } else if token == "," {
                        self.advance();
                        body.push(self.parse_atom()?);
                    } else {
                        return Err(ParseError::UnexpectedToken(token.into()));
                    }
                }

                Ok(Some(Statement::Rule(Rule { head: atom, body })))
            }
            Some(token) => Err(ParseError::UnexpectedToken(token.into())),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    pub fn parse_all(&mut self) -> Result<Vec<Statement>, ParseError> {
        let mut statements = Vec::new();

        while let Some(statement) = self.parse_statement()? {
            statements.push(statement);
        }

        Ok(statements)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_underscore_is_variable_in_body() {
        let mut p = DatalogParser::new("ok :- parent(_, bob).");
        let v = p.parse_all().unwrap();
        match &v[0] {
            Statement::Rule(r) => {
                assert_eq!(r.body[0].predicate.as_ref(), "parent");
                match &r.body[0].args[0] {
                    Term::Variable(name) => assert_eq!(name.as_ref(), "_"),
                    other => panic!("first arg should be Variable(\"_\"), got {:?}", other),
                }
                match &r.body[0].args[1] {
                    Term::Constant(name) => assert_eq!(name.as_ref(), "bob"),
                    other => panic!("second arg should be Constant(\"bob\"), got {:?}", other),
                }
            }
            _ => panic!("expected rule"),
        }
    }

    #[test]
    fn iso_named_underscore_is_variable() {
        let mut p = DatalogParser::new("ok :- parent(_Kid, bob).");
        let v = p.parse_all().unwrap();
        match &v[0] {
            Statement::Rule(r) => {
                match &r.body[0].args[0] {
                    Term::Variable(name) => assert_eq!(name.as_ref(), "_Kid"),
                    other => panic!("expected Variable(\"_Kid\"), got {:?}", other),
                }
            }
            _ => panic!("expected rule"),
        }
    }

    #[test]
    fn iso_multiple_underscores_are_variables() {
        let mut p = DatalogParser::new("p(_, _).");
        let v = p.parse_all().unwrap();
        match &v[0] {
            Statement::Fact(a) => {
                for i in 0..2 {
                    match &a.args[i] {
                        Term::Variable(name) => assert_eq!(name.as_ref(), "_"),
                        other => panic!("arg {} should be Variable(\"_\"), got {:?}", i, other),
                    }
                }
            }
            _ => panic!("expected fact"),
        }
    }

    #[test]
    fn iso_query_with_underscore_variable() {
        let mut p = DatalogParser::new("?- parent(_, Who).");
        let v = p.parse_all().unwrap();
        match &v[0] {
            Statement::Query(a) => {
                match &a.args[0] {
                    Term::Variable(name) => assert_eq!(name.as_ref(), "_"),
                    other => panic!("first arg should be Variable(\"_\"), got {:?}", other),
                }
                match &a.args[1] {
                    Term::Variable(name) => assert_eq!(name.as_ref(), "Who"),
                    other => panic!("second arg should be Variable(\"Who\"), got {:?}", other),
                }
            }
            _ => panic!("expected query"),
        }
    }

    #[test]
    fn iso_underscore_digits_and_names_are_variables() {
        let mut p = DatalogParser::new("t(_1, _Tmp, X).");
        let v = p.parse_all().unwrap();
        match &v[0] {
            Statement::Fact(a) => {
                // _1
                match &a.args[0] {
                    Term::Variable(name) => assert_eq!(name.as_ref(), "_1"),
                    other => panic!("_1 should be Variable, got {:?}", other),
                }
                // _Tmp
                match &a.args[1] {
                    Term::Variable(name) => assert_eq!(name.as_ref(), "_Tmp"),
                    other => panic!("_Tmp should be Variable, got {:?}", other),
                }
                // X
                match &a.args[2] {
                    Term::Variable(name) => assert_eq!(name.as_ref(), "X"),
                    other => panic!("X should be Variable, got {:?}", other),
                }
            }
            _ => panic!("expected fact"),
        }
    }

    #[test]
    fn iso_nullary_head_and_nullaries_in_body() {
        let src = "ok :- p, r, thing(X,Y).";
        let mut p = DatalogParser::new(src);
        let v = p.parse_all().unwrap();
        match &v[0] {
            Statement::Rule(r) => {
                assert_eq!(r.head.predicate.as_ref(), "ok");
                assert!(r.head.args.is_empty(), "nullary head");

                assert_eq!(r.body.len(), 3);
                assert_eq!(r.body[0].predicate.as_ref(), "p");
                assert!(r.body[0].args.is_empty(), "p is nullary");

                assert_eq!(r.body[1].predicate.as_ref(), "r");
                assert!(r.body[1].args.is_empty(), "r is nullary");

                assert_eq!(r.body[2].predicate.as_ref(), "thing");
                assert_eq!(r.body[2].args.len(), 2);
            }
            _ => panic!("expected a rule"),
        }
    }

    #[test]
    fn iso_nullary_fact_and_query() {
        let src = "p.\n?- p.";
        let mut p = DatalogParser::new(src);
        let v = p.parse_all().unwrap();
        assert_eq!(v.len(), 2);

        match &v[0] {
            Statement::Fact(a) => {
                assert_eq!(a.predicate.as_ref(), "p");
                assert!(a.args.is_empty());
            }
            _ => panic!("expected nullary fact"),
        }
        match &v[1] {
            Statement::Query(a) => {
                assert_eq!(a.predicate.as_ref(), "p");
                assert!(a.args.is_empty());
            }
            _ => panic!("expected nullary query"),
        }
    }

    #[test]
    fn iso_reject_paren_form_for_nullary() {
        for src in &["p().", "?- p().", "ok() :- thing(X,Y)."] {
            let mut p = DatalogParser::new(src);
            let err = p.parse_all().unwrap_err();
            match err {
                ParseError::InvalidSyntax(msg) => {
                    assert!(msg.contains("Nullary predicate"), "got: {msg}");
                }
                other => panic!("expected InvalidSyntax for '{}', got {:?}", src, other),
            }
        }
    }

    #[test]
    fn iso_predicates_must_start_lowercase() {
        let mut p = DatalogParser::new("Ok :- p.");
        let err = p.parse_all().unwrap_err();
        match err {
            ParseError::InvalidSyntax(msg) => {
                assert!(msg.contains("lowercase"), "got: {msg}");
            }
            other => panic!("expected InvalidSyntax for uppercase atom, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_with_percent_and_slash_comments() {
        let src = r#"
            % a line comment
            parent(tom, bob). // trailing comment

            /* block
               comment */
            grandparent(X, Z) /* inline block */ :- parent(X, Y), // mid-line //
                                parent(Y, Z).  % end
            ?- parent(tom, Who).  // query
        "#;

        let mut parser = DatalogParser::new(src);
        let statements = parser.parse_all().unwrap();

        assert_eq!(statements.len(), 3);

        match &statements[0] {
            Statement::Fact(atom) => {
                assert_eq!(atom.predicate.as_ref(), "parent");
                assert_eq!(atom.args.len(), 2);
            }
            _ => panic!("expected fact"),
        }

        match &statements[1] {
            Statement::Rule(rule) => {
                assert_eq!(rule.head.predicate.as_ref(), "grandparent");
                assert_eq!(rule.body.len(), 2);
            }
            _ => panic!("expected rule"),
        }

        match &statements[2] {
            Statement::Query(atom) => {
                assert_eq!(atom.predicate.as_ref(), "parent");
                assert_eq!(atom.args.len(), 2);
            }
            _ => panic!("expected query"),
        }
    }

    #[test]
    fn test_block_comment_at_eof_is_ok() {
        // Unterminated block just ends input; parser will stop cleanly.
        let src = "p(a,b). /* not closed";
        let mut p = DatalogParser::new(src);
        let v = p.parse_all().unwrap();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_parse_fact() {
        let mut parser = DatalogParser::new("parent(tom, bob).");
        let statements = parser.parse_all().unwrap();

        assert_eq!(statements.len(), 1);
        match &statements[0] {
            Statement::Fact(atom) => {
                assert_eq!(atom.predicate.as_ref(), "parent");
                assert_eq!(atom.args.len(), 2);
                assert_eq!(atom.args[0], Term::Constant("tom".into()));
                assert_eq!(atom.args[1], Term::Constant("bob".into()));
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_parse_rule() {
        let mut parser = DatalogParser::new("grandparent(X, Z) :- parent(X, Y), parent(Y, Z).");
        let statements = parser.parse_all().unwrap();

        assert_eq!(statements.len(), 1);
        match &statements[0] {
            Statement::Rule(rule) => {
                assert_eq!(rule.head.predicate.as_ref(), "grandparent");
                assert_eq!(rule.head.args[0], Term::Variable("X".into()));
                assert_eq!(rule.head.args[1], Term::Variable("Z".into()));
                assert_eq!(rule.body.len(), 2);
                assert_eq!(rule.body[0].predicate.as_ref(), "parent");
            }
            _ => panic!("Expected rule"),
        }
    }

    #[test]
    fn test_parse_query() {
        let mut parser = DatalogParser::new("?- parent(tom, Who).");
        let statements = parser.parse_all().unwrap();

        assert_eq!(statements.len(), 1);
        match &statements[0] {
            Statement::Query(atom) => {
                assert_eq!(atom.predicate.as_ref(), "parent");
                assert_eq!(atom.args[0], Term::Constant("tom".into()));
                assert_eq!(atom.args[1], Term::Variable("Who".into()));
            }
            _ => panic!("Expected query"),
        }
    }

    #[test]
    fn test_variable_vs_constant() {
        let mut parser = DatalogParser::new("test(X, bob, Y, alice).");
        let statements = parser.parse_all().unwrap();

        match &statements[0] {
            Statement::Fact(atom) => {
                assert_eq!(atom.args[0], Term::Variable("X".into()));
                assert_eq!(atom.args[1], Term::Constant("bob".into()));
                assert_eq!(atom.args[2], Term::Variable("Y".into()));
                assert_eq!(atom.args[3], Term::Constant("alice".into()));
            }
            _ => panic!("Expected fact"),
        }
    }

    #[test]
    fn test_empty_body_rule_ok() {
        let mut p = DatalogParser::new("r(X) :- .");
        let v = p.parse_all().unwrap();
        assert_eq!(v.len(), 1);
        match &v[0] {
            Statement::Rule(rule) => {
                assert_eq!(rule.head.predicate.as_ref(), "r");
                assert!(rule.body.is_empty());
            }
            _ => panic!("expected rule"),
        }
    }
}

// ================== kb builder / deduper ==================
use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::num::NonZeroU32;

// ---- Non-zero IDs (separate namespaces) ----
#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)]
pub struct PredId(pub(crate) NonZeroU32);
#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)]
pub struct ConstId(pub(crate) NonZeroU32);

impl PredId {
    pub const fn sentinal() -> Self {
        unsafe { Self(NonZeroU32::new_unchecked(u32::MAX)) }
    }
}

// ---- Compact term representation ----
// Vars:  0,1,2,...  (clause-local index, 0-based)
// Const: -id         (id is NonZeroU32, so negative never overlaps vars)
#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)]
pub struct Term32(pub i32);

impl Term32 {
    #[inline]
    pub const fn var(ix0: u32) -> Self {
        Self(ix0 as i32)
    }
    #[inline]
    pub const fn from_const(cid: ConstId) -> Self {
        Self(-(cid.0.get() as i32))
    }

    #[inline]
    pub const fn is_var(self) -> bool {
        self.0 >= 0
    }
    #[inline]
    pub const fn is_const(self) -> bool {
        self.0 < 0
    }

    #[inline]
    pub const fn var_index(self) -> u32 {
        debug_assert!(self.is_var(), "var_index on non-var");
        self.0 as u32
    }

    #[inline]
    pub const unsafe fn const_id_unchecked(self) -> ConstId {
        debug_assert!(self.is_const(), "const_id on non-const");
        let raw = (-self.0) as u32;
        debug_assert!(raw >= 1, "encoded const id must be >= 1");
        unsafe { ConstId(NonZeroU32::new_unchecked(raw)) }
    }

    #[inline]
    pub const fn term(self) -> TermId {
        if self.is_var() {
            TermId::Var(self.var_index())
        } else {
            unsafe { TermId::Const(self.const_id_unchecked()) }
        }
    }

    #[inline]
    pub const fn try_const(self) -> Option<ConstId> {
        if self.is_var() {
            None
        } else {
            unsafe { Some(self.const_id_unchecked()) }
        }
    }

    #[inline]
    pub const fn try_var(self) -> Option<u32> {
        if self.is_var() {
            Some(self.var_index())
        } else {
            None
        }
    }
}

// Public ergonomic view (optional)
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum TermId {
    Var(u32),       // 0-based
    Const(ConstId), // interned
}
impl From<TermId> for Term32 {
    #[inline]
    fn from(t: TermId) -> Self {
        match t {
            TermId::Var(ix) => Term32::var(ix),
            TermId::Const(c) => Term32::from_const(c),
        }
    }
}
impl From<Term32> for TermId {
    #[inline]
    fn from(t: Term32) -> Self {
        t.term()
    }
}

// ---- Canonicalized structures ----
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct AtomId {
    pub pred: PredId,
    pub args: Box<[Term32]>, // immutable, arity is fixed
}

impl AtomId {
    pub fn var_count(&self) -> usize {
        let vars = self
            .args
            .iter()
            .filter(|i| i.is_var())
            .cloned()
            .collect::<HashSet<_>>();
        vars.len()
    }

    pub fn is_canon(&self) -> bool {
        let mut next_var = 0;
        for id in self.args.iter().filter_map(|i| i.try_var()) {
            if id > next_var {
                return false;
            }

            if id == next_var {
                next_var += 1;
            }
        }

        true
    }

    pub fn canonize(&mut self) -> std::collections::HashMap<u32, u32> {
        let mut next_var = 0;
        let mut vars = HashMap::new();
        for a in self.args.iter_mut() {
            let TermId::Var(v) = a.term() else {
                continue;
            };

            let id = vars.entry(v).or_insert_with(|| {
                next_var += 1;
                next_var - 1
            });

            *a = TermId::Var(*id).into();
        }

        vars
    }
}

// Total order for canonical sorting/dedup in rule bodies
impl Ord for AtomId {
    fn cmp(&self, other: &Self) -> Ordering {
        let ans = self.args.as_ref().cmp(other.args.as_ref());
        if ans == Ordering::Equal {
            self.pred.cmp(&other.pred)
        } else {
            ans
        }
    }
}
impl PartialOrd for AtomId {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct RuleId {
    pub head: AtomId,
    pub body: Box<[AtomId]>, // immutable, CANONICALLY SORTED (see add_rule)
}

impl RuleId {
    pub fn var_count(&self) -> usize {
        self.body
            .iter()
            .chain(Some(&self.head))
            .map(|a| a.var_count())
            .max()
            .unwrap_or(0)
    }
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
    fn nz(n: u32) -> NonZeroU32 {
        NonZeroU32::new(n).unwrap()
    }

    pub fn intern_pred_with_arity(
        &mut self,
        name: &Box<str>,
        arity: u32,
    ) -> Result<PredId, KBError> {
        if let Some(&id) = self.pred_map.get(name) {
            let ix = (id.0.get() - 1) as usize;
            let a = self.pred_arity[ix];
            if a == arity {
                Ok(id)
            } else {
                Err(KBError::WrongArity {
                    predicate: self.pred_names[ix].clone(),
                    expected: a,
                    found: arity,
                })
            }
        } else {
            let id = PredId(Self::nz(self.pred_names.len() as u32 + 1));
            self.pred_names.push(name.clone());
            self.pred_map
                .insert(self.pred_names.last().unwrap().clone(), id);
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
        if let Some(&id) = self.const_map.get(name) {
            return id;
        }
        let id = ConstId(Self::nz(self.const_names.len() as u32 + 1));
        self.const_names.push(name.clone());
        self.const_map
            .insert(self.const_names.last().unwrap().clone(), id);
        id
    }

    pub fn get_const_if_known(&self, name: &Box<str>) -> Option<ConstId> {
        self.const_map.get(name).copied()
    }

    pub fn pred_name(&self, id: PredId) -> &str {
        self.pred_names[(id.0.get() - 1) as usize].as_ref()
    }
    pub fn const_name(&self, id: ConstId) -> &str {
        self.const_names[(id.0.get() - 1) as usize].as_ref()
    }
    pub fn pred_arity_of(&self, id: PredId) -> u32 {
        self.pred_arity[(id.0.get() - 1) as usize]
    }
}

// ---- Clause-local variable scope (alpha-renaming -> 0-based ints) ----
struct VarScope {
    map: HashMap<Box<str>, u32>, // var name -> 0-based index
    names: Vec<Box<str>>,
    next: u32, // next index to assign (also the size)
}
impl VarScope {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            names: Vec::new(),
            next: 0,
        }
    }
    #[inline]
    fn get_ix(&mut self, v: &str) -> u32 {
        match self.map.entry(v.into()) {
            Entry::Occupied(o) => *o.get(),
            Entry::Vacant(vac) => {
                let ix = self.next;
                self.next += 1;
                vac.insert(ix);
                self.names.push(v.into());
                ix
            }
        }
    }
    #[inline]
    fn anon_ix(&mut self) -> u32 {
        let ix = self.next;
        self.next += 1;
        ix
    }
}

// ---- Errors (Box<str>, no String) ----
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KBError {
    WrongArity {
        predicate: Box<str>,
        expected: u32,
        found: u32,
    },
    UnboundVar {
        var: Box<str>,
    },
    HeadVarNotBound {
        predicate: Box<str>,
        arg: Box<str>,
        arg_idx: usize,
    },
    UnknownPredicateInQuery {
        predicate: Box<str>,
    },
    UnknownConstantInQuery {
        predicate: Box<str>,
        constant: Box<str>,
    },
}

impl fmt::Display for KBError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use KBError::*;
        match self {
            WrongArity {
                predicate,
                expected,
                found,
            } => write!(
                f,
                "Predicate '{}' arity mismatch: expected {}, found {}.",
                predicate, expected, found
            ),
            HeadVarNotBound {
                predicate,
                arg,
                arg_idx,
            } => write!(
                f,
                "Rule {predicate}(...): head var '{arg}' at position {arg_idx} not bound in body.",
            ),
            UnknownPredicateInQuery { predicate } => {
                write!(f, "Query references unknown predicate '{}'.", predicate)
            }
            UnknownConstantInQuery {
                predicate,
                constant,
            } => write!(
                f,
                "Query on '{}' uses unknown constant '{}'.",
                predicate, constant
            ),
            KBError::UnboundVar { var } => write!(f, "Unbound Variable '{var}'"),
        }
    }
}

// ---- Producers bucket per predicate ----
#[derive(Default, Clone)]
pub struct Producers {
    pub rules: Vec<RuleId>,
    pub facts: Vec<AtomId>,
}
impl Producers {
    #[inline]
    fn push_rule(&mut self, r: RuleId) {
        self.rules.push(r);
    }
    #[inline]
    fn push_fact(&mut self, a: AtomId) {
        self.facts.push(a);
    }
}

// ---- KB (single owner) ----
#[derive(Default)]
pub struct KB {
    pub(crate) inter: Interner,

    // facts
    pub(crate) facts: HashSet<AtomId>,

    // rules (dedup set)
    pub(crate) rules: HashSet<RuleId>,

    // single lookup: pred -> Producers {rules, ground, generic}
    pub(crate) producers: HashMap<PredId, Producers>,
}

impl KB {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn interner(&self) -> &Interner {
        &self.inter
    }

    pub fn add_statement(&mut self, st: &Statement) -> Result<Option<AtomId>, KBError> {
        match st {
            Statement::Fact(a) => {
                self.add_fact_or_generic(a)?;
                Ok(None)
            }
            Statement::Rule(r) => {
                self.add_rule(r)?;
                Ok(None)
            }
            Statement::Query(a) => {
                // Non-mutating: just return the encoded query
                let enc = self.get_query_encoding(a)?;
                Ok(Some(enc))
            }
        }
    }

    #[inline]
    fn producers_mut(&mut self, p: PredId) -> &mut Producers {
        self.producers.entry(p).or_default()
    }

    // ----- atoms -----
    fn canon_atom(&mut self, a: &Atom, vs: &mut VarScope) -> Result<AtomId, KBError> {
        let pred = self
            .inter
            .intern_pred_with_arity(&a.predicate, a.args.len() as u32)?;
        let mut args_vec: Vec<Term32> = Vec::with_capacity(a.args.len());
        for t in &a.args {
            let enc = match t {
                Term::Variable(v) if v.as_ref() == "_" => Term32::var(vs.anon_ix()),
                Term::Variable(v) => Term32::var(vs.get_ix(v)),
                Term::Constant(c) => Term32::from_const(self.inter.intern_const(c)),
            };
            args_vec.push(enc);
        }
        Ok(AtomId {
            pred,
            args: args_vec.into_boxed_slice(),
        })
    }

    // #[inline]
    // fn is_ground_atom(a: &AtomId) -> bool {
    //     a.args.iter().all(|&t| t.is_const())
    // }

    // Parser `Fact(_)` may contain variables → treat as generic fact.
    fn add_fact_or_generic(&mut self, a: &Atom) -> Result<(), KBError> {
        let mut vs = VarScope::new();
        let aid = self.canon_atom(a, &mut vs)?;

        for t in &aid.args {
            match t.term() {
                TermId::Var(i) => {
                    return Err(KBError::UnboundVar {
                        var: vs.names[i as usize].clone(),
                    });
                }
                _ => {}
            }
        }

        if self.facts.insert(aid.clone()) {
            self.producers_mut(aid.pred).push_fact(aid);
        }
        Ok(())
    }

    // Rules must have non-empty body; if body is empty, route to fact/generic.
    fn add_rule(&mut self, r: &Rule) -> Result<(), KBError> {
        if r.body.is_empty() {
            return self.add_fact_or_generic(&r.head);
        }

        let mut vs = VarScope::new();
        let head = self.canon_atom(&r.head, &mut vs)?;

        // collect, then canonical sort + dedup; kills permutations & duplicates
        let mut body_vec: Vec<AtomId> = Vec::with_capacity(r.body.len());
        for a in &r.body {
            body_vec.push(self.canon_atom(a, &mut vs)?);
        }
        body_vec.sort_unstable(); // uses Ord on AtomId
        body_vec.dedup(); // adjacent duplicates removed

        // range-restriction: each head var must appear in (deduped, sorted) body
        for (i, &t) in head.args.iter().enumerate() {
            if t.is_var() {
                let vh = t;
                let mut found = false;
                'outer: for b in &body_vec {
                    for &bt in b.args.iter() {
                        if bt == vh {
                            found = true;
                            break 'outer;
                        }
                    }
                }
                if !found {
                    return Err(KBError::HeadVarNotBound {
                        predicate: r.head.predicate.clone(),
                        arg_idx: i,
                        arg: vs.names[t.var_index() as usize].clone(),
                    });
                }
            }
        }

        let rule = RuleId {
            head: head.clone(),
            body: body_vec.into_boxed_slice(),
        };

        // global O(1) dedupe; only on insert update producers bucket
        if self.rules.insert(rule.clone()) {
            self.producers_mut(head.pred).push_rule(rule);
        }
        Ok(())
    }

    /// Queries must not add info: pred+arity must exist, constants must be known.
    /// Returns the canonical query encoding (variables canonically 0-based).
    pub fn get_query_encoding(&self, a: &Atom) -> Result<AtomId, KBError> {
        // predicate must already be known; arity must match
        let (pred, expect_arity) = self.inter.get_pred_if_known(&a.predicate).ok_or_else(|| {
            KBError::UnknownPredicateInQuery {
                predicate: a.predicate.clone(),
            }
        })?;
        let found_arity = a.args.len() as u32;
        if expect_arity != found_arity {
            return Err(KBError::WrongArity {
                predicate: a.predicate.clone(),
                expected: expect_arity,
                found: found_arity,
            });
        }

        // build clause-local variable numbering; constants must already be interned
        let mut vs = VarScope::new();
        let mut args_vec: Vec<Term32> = Vec::with_capacity(a.args.len());
        for t in &a.args {
            match t {
                Term::Variable(v) if v.as_ref() == "_" => {
                    args_vec.push(Term32::var(vs.anon_ix()));
                }
                Term::Variable(v) => {
                    args_vec.push(Term32::var(vs.get_ix(v)));
                }
                Term::Constant(c) => {
                    if let Some(cid) = self.inter.get_const_if_known(c) {
                        args_vec.push(Term32::from_const(cid));
                    } else {
                        return Err(KBError::UnknownConstantInQuery {
                            predicate: a.predicate.clone(),
                            constant: c.clone(),
                        });
                    }
                }
            }
        }

        Ok(AtomId {
            pred,
            args: args_vec.into_boxed_slice(),
        })
    }

    pub fn trace_pred(&self, pred: PredId) -> HashSet<PredId> {
        let mut ans = HashSet::new();
        ans.insert(pred);

        let mut stack = Vec::new();
        stack.push(pred);

        while let Some(id) = stack.pop() {
            for r in &self.producers[&id].rules {
                for a in &r.body {
                    if ans.insert(a.pred) {
                        stack.push(a.pred);
                    }
                }
            }
        }
        ans
    }
}

// ---- Tests for KB layer ----
#[cfg(test)]
mod tests_kb {
    use super::*;
    use crate::parser::DatalogParser;

    fn build(input: &str) -> Result<KB, KBError> {
        let mut p = DatalogParser::new(input);
        let stmts = p.parse_all().unwrap();
        let mut kb = KB::new();
        for s in &stmts {
            kb.add_statement(s)?;
        }
        Ok(kb)
    }

    #[test]
    fn producers_split_and_rule_dedupe() {
        let kb = build(
            "parent(tom,bob).
             parent(bob,alice).
             ancestor(X,Z) :- parent(X,Z).
             ancestor(X,Z) :- parent(X,Y), parent(Y,Z).
             ancestor(A,B) :- parent(A,C), parent(C,B).",
        )
        .unwrap();

        // rules deduped by alpha-equivalence: last two are equivalent
        assert_eq!(kb.rules.len(), 2);

        // producers split:
        let (pid, _) = kb.inter.get_pred_if_known(&"ancestor".into()).unwrap();
        let prod = kb.producers.get(&pid).expect("ancestor producers exist");
        assert_eq!(prod.facts.len(), 0, "no facts for ancestor");
        assert_eq!(prod.rules.len(), 2, "two distinct ancestor rules");
    }

    #[test]
    fn query_checks_dont_add_info() {
        // Introduce parent/2 and constants
        let mut p = DatalogParser::new(
            "parent(tom,bob).
             ?- parent(tom,Who).
             ?- parent(tom,alice).",
        );
        let stmts = p.parse_all().unwrap();
        let mut b = KB::new();

        // ground fact registers pred/arity and constants {tom,bob}
        b.add_statement(&stmts[0]).unwrap();

        // query with new variable OK
        b.add_statement(&stmts[1]).unwrap();

        // query with new constant 'alice' should fail
        match b.add_statement(&stmts[2]) {
            Err(KBError::UnknownConstantInQuery {
                predicate,
                constant,
            }) => {
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
            Err(KBError::HeadVarNotBound {
                predicate,
                arg_idx,
                arg: _,
            }) => {
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

    #[test]
    fn body_ordering_partitions_ground_before_non_ground_and_dedupes() {
        use crate::parser::DatalogParser;

        // Mixed body: duplicates + permutation; two clauses that are logically the same
        let mut p = DatalogParser::new(
            "p(X) :- s(Y), r(c1), q(X), t(c2), r(c1), q(X).
             p(U) :- t(c2), q(U), s(V), r(c1).",
        );
        let stmts = p.parse_all().unwrap();

        // Build KB
        let mut kb = super::KB::new();
        for s in &stmts {
            kb.add_statement(s).unwrap();
        }

        // Fetch producers for head predicate p/1
        let (ppred, _) = kb.inter.get_pred_if_known(&"p".into()).unwrap();
        let prod = kb.producers.get(&ppred).expect("no producers for p/1");

        // Only ONE canonical rule should remain (permutation & dup collapse)
        assert_eq!(
            prod.rules.len(),
            1,
            "rules should dedupe by canonical body order"
        );

        let body = &prod.rules[0].body;

        // We expect these unique atoms in the body: r(c1), t(c2), q(X), s(Y) => 4
        assert_eq!(body.len(), 4, "body should be deduped to 4 unique atoms");

        // Check the partition: all ground atoms first, then all with variables
        let mut first_non_ground = body.len();
        for (i, atom) in body.iter().enumerate() {
            let is_ground = atom.args.iter().all(|t| t.is_const());
            if !is_ground {
                first_non_ground = i;
                break;
            }
        }
        // If there are any non-ground atoms, ensure everything before is ground
        for i in 0..first_non_ground {
            assert!(
                body[i].args.iter().all(|t| t.is_const()),
                "expected ground atom before the first non-ground atom at index {}",
                first_non_ground
            );
        }
        // And everything from that point on is non-ground
        for i in first_non_ground..body.len() {
            assert!(
                body[i].args.iter().any(|t| t.is_var()),
                "expected non-ground atom at or after the partition index {}",
                first_non_ground
            );
        }

        // Also verify there are no accidental duplicates post-sort/dedup
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for a in body.iter() {
            assert!(
                seen.insert(a),
                "duplicate atom remained after canonicalization: {:?}",
                a
            );
        }
    }
}
