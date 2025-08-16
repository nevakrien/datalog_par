// made by claude4 + comments
use std::fmt;
use std::iter::Peekable;
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

pub struct DatalogParser<'a> {
    chars: Peekable<Chars<'a>>,
    current_token: Option<String>,
}

impl<'a> DatalogParser<'a> {
    pub fn new(input: &'a str) -> Self {
        let mut parser = DatalogParser {
            chars: input.chars().peekable(),
            current_token: None,
        };
        parser.advance_token(); // Initialize with first token
        parser
    }

    fn advance_token(&mut self) {
        self.current_token = self.next_token();
    }

    #[inline]
    fn skip_line_comment(&mut self) {
        while let Some(&ch) = self.chars.peek() {
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
            while let Some(&ch) = self.chars.peek() {
                if ch.is_whitespace() {
                    self.chars.next();
                } else {
                    break;
                }
            }

            // check for comments: %, //, /* ... */
            let mut cloned = self.chars.clone();
            match cloned.next() {
                Some('%') => {
                    // consume '%' and the rest of the line
                    self.chars.next();
                    self.skip_line_comment();
                    continue;
                }
                Some('/') => {
                    match cloned.next() {
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
                if self.chars.peek() == Some(&'-') {
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
                if self.chars.peek() == Some(&'-') {
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
        while let Some(&ch) = self.chars.peek() {
            match ch {
                ' ' | '\t' | '\n' | '\r' | '(' | ')' | ',' | '.' => break,
                ':' | '?' => {
                    // Check if this starts a special token (:- or ?-)
                    if let Some(&next_ch) = self.chars.peek() {
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
                let term = if token.chars().next().unwrap().is_uppercase() {
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
        let predicate = match self.current_token() {
            Some(token) => {
                let pred: Box<str> = token.into();
                self.advance();
                pred
            }
            None => return Err(ParseError::UnexpectedEof),
        };

        self.expect_token("(")?;

        let mut args = Vec::new();

        // Handle empty argument list
        if let Some(token) = self.current_token() {
            if token == ")" {
                self.advance();
                return Ok(Atom { predicate, args });
            }
        }

        // Parse first argument
        args.push(self.parse_term()?);

        // Parse remaining arguments
        while let Some(token) = self.current_token() {
            if token == ")" {
                self.advance();
                break;
            } else if token == "," {
                self.advance();
                args.push(self.parse_term()?);
            } else {
                return Err(ParseError::UnexpectedToken(token.into()));
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
