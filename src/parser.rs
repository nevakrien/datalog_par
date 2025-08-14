//made by claude4
use std::fmt;

#[derive(Debug, Clone, PartialEq,Eq,Hash)]
pub enum Term {
    Variable(Box<str>),
    Constant(Box<str>),
}

#[derive(Debug, Clone, PartialEq,Eq,Hash)]
pub struct Atom {
    pub predicate: Box<str>,
    pub args: Vec<Term>,
}

#[derive(Debug, Clone, PartialEq,Eq,Hash)]
pub struct Rule {
    pub head: Atom,
    pub body: Vec<Atom>,
}

#[derive(Debug, Clone, PartialEq,Eq,Hash)]
pub enum Statement {
    Fact(Atom),
    Rule(Rule),
    Query(Atom),
}

#[derive(Debug,PartialEq,Clone)]
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

pub struct DatalogParser {
    tokens: Vec<Box<str>>,
    position: usize,
}

impl DatalogParser {
    pub fn new(input: &str) -> Self {
        let tokens = Self::tokenize(input);
        DatalogParser { tokens, position: 0 }
    }

    fn tokenize(input: &str) -> Vec<Box<str>> {
        let mut tokens = Vec::new();
        let mut current_token = std::string::String::new();
        let mut chars = input.chars().peekable();

        while let Some(ch) = chars.next() {
            match ch {
                ' ' | '\t' | '\n' | '\r' => {
                    if !current_token.is_empty() {
                        tokens.push(current_token.clone().into());
                        current_token.clear();
                    }
                }
                '(' | ')' | ',' | '.' => {
                    if !current_token.is_empty() {
                        tokens.push(current_token.clone().into());
                        current_token.clear();
                    }
                    tokens.push(ch.to_string().into());
                }
                ':' => {
                    if chars.peek() == Some(&'-') {
                        chars.next(); // consume '-'
                        if !current_token.is_empty() {
                            tokens.push(current_token.as_str().into());
                            current_token.clear();
                        }
                        tokens.push(":-".into());
                    } else {
                        current_token.push(ch);
                    }
                }
                '?' => {
                    if chars.peek() == Some(&'-') {
                        chars.next(); // consume '-'
                        if !current_token.is_empty() {
                            tokens.push(current_token.as_str().into());
                            current_token.clear();
                        }
                        tokens.push("?-".into());
                    } else {
                        current_token.push(ch);
                    }
                }
                _ => current_token.push(ch),
            }
        }

        if !current_token.is_empty() {
            tokens.push(current_token.into());
        }

        tokens
    }

    fn current_token(&self) -> Option<&str> {
        self.tokens.get(self.position).map(|s| s.as_ref())
    }

    fn advance(&mut self) -> Option<&str> {
        self.position += 1;
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
        if self.position >= self.tokens.len() {
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

        // Check what follows
        match self.current_token() {
            Some(token) if token == "." => {
                self.advance();
                Ok(Some(Statement::Fact(atom)))
            }
            Some(token) if token == ":-" => {
                self.advance();
                
                // Parse body atoms
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
}