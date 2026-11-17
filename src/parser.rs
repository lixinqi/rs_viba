//! The .viba lexer and recursive-descent parser (was viba/parser.py).

use pyo3::exceptions::PySyntaxError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};

use crate::pyast::{self, Nodes};

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    ClassName(String),
    Tagged(String),
    Int(String),
    Float(String),
    Str(String),
    Bool(bool),
    Nil,
    Never,
    Any,
    Ellipsis,
    HintBlock(String),
    Assign,
    Sum,
    Prod,
    Exp,
    Apply,
    Import,
    As,
    Pattern,
    LBracket,
    RBracket,
    LParen,
    RParen,
    Comma,
    Dot,
    Eof,
}

#[derive(Clone, Debug)]
struct Token {
    tok: Tok,
    line: usize,
}

struct Lexer<'py> {
    py: Python<'py>,
    chars: Vec<char>,
    pos: usize,
    line: usize,
}

impl<'py> Lexer<'py> {
    fn new(py: Python<'py>, source: &str) -> Self {
        Self {
            py,
            chars: source.chars().collect(),
            pos: 0,
            line: 1,
        }
    }

    fn at_end(&self) -> bool {
        self.pos >= self.chars.len()
    }

    fn cur(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn at(&self, offset: usize) -> Option<char> {
        self.chars.get(self.pos + offset).copied()
    }

    fn push(&mut self, tok: Tok, out: &mut Vec<Token>) {
        out.push(Token {
            tok,
            line: self.line,
        });
    }

    fn error(&self, message: String) -> PyErr {
        PySyntaxError::new_err(message)
    }

    fn illegal(&self) -> PyErr {
        let c = self.cur().unwrap_or('\0');
        let repr = pyast::repr_of(self.py, PyString::new(self.py, &c.to_string()).as_any());
        self.error(format!(
            "Viba parse error: illegal character {} at line {}",
            repr, self.line
        ))
    }

    fn tokenize(mut self) -> PyResult<Vec<Token>> {
        let mut out = Vec::new();
        while let Some(c) = self.cur() {
            match c {
                ' ' | '\t' | '\r' => {
                    self.pos += 1;
                }
                '\n' => {
                    self.line += 1;
                    self.pos += 1;
                }
                '#' => {
                    while let Some(c) = self.cur() {
                        if c == '\n' {
                            break;
                        }
                        self.pos += 1;
                    }
                }
                '{' => {
                    let (code, line) = self.hint_block()?;
                    if code.is_none() {
                        return Err(self.error(format!(
                            "Viba parse error: unterminated hint block at line {}",
                            line
                        )));
                    }
                    self.push(Tok::HintBlock(code.unwrap()), &mut out);
                }
                '"' | '\'' => {
                    let tok = self.string(c)?;
                    self.push(tok, &mut out);
                }
                '-' | '0'..='9' | '.' => {
                    if c == '-' && !self.number_ahead() {
                        return Err(self.illegal());
                    }
                    if c == '.' {
                        if self.at(1) == Some('.') && self.at(2) == Some('.') {
                            self.pos += 3;
                            self.push(Tok::Ellipsis, &mut out);
                        } else if matches!(self.at(1), Some(d) if d.is_ascii_digit()) {
                            let tok = self.number()?;
                            self.push(tok, &mut out);
                        } else if self.dot_allowed() {
                            self.pos += 1;
                            self.push(Tok::Dot, &mut out);
                        } else {
                            return Err(self.illegal());
                        }
                    } else {
                        let tok = self.number()?;
                        self.push(tok, &mut out);
                    }
                }
                '$' => {
                    self.pos += 1;
                    let start = self.pos;
                    while matches!(self.cur(), Some(c) if word_char(c)) {
                        self.pos += 1;
                    }
                    if self.pos == start {
                        return Err(self.illegal());
                    }
                    let tag: String = self.chars[start..self.pos].iter().collect();
                    self.push(Tok::Tagged(format!("${}", tag)), &mut out);
                }
                '<' => {
                    if self.at(1) == Some('<') {
                        self.pos += 2;
                        self.push(Tok::Apply, &mut out);
                    } else if self.at(1) == Some('-') {
                        self.pos += 2;
                        self.push(Tok::Exp, &mut out);
                    } else {
                        return Err(self.illegal());
                    }
                }
                '=' => {
                    self.pos += 1;
                    self.push(Tok::Assign, &mut out);
                }
                '|' => {
                    self.pos += 1;
                    self.push(Tok::Sum, &mut out);
                }
                '*' => {
                    self.pos += 1;
                    self.push(Tok::Prod, &mut out);
                }
                '[' => {
                    self.pos += 1;
                    self.push(Tok::LBracket, &mut out);
                }
                ']' => {
                    self.pos += 1;
                    self.push(Tok::RBracket, &mut out);
                }
                '(' => {
                    self.pos += 1;
                    self.push(Tok::LParen, &mut out);
                }
                ')' => {
                    self.pos += 1;
                    self.push(Tok::RParen, &mut out);
                }
                ',' => {
                    self.pos += 1;
                    self.push(Tok::Comma, &mut out);
                }
                _ if word_char(c) => {
                    let start = self.pos;
                    while matches!(self.cur(), Some(c) if word_char(c)) {
                        self.pos += 1;
                    }
                    let word: String = self.chars[start..self.pos].iter().collect();
                    let tok = match word.as_str() {
                        "import" => Tok::Import,
                        "as" => Tok::As,
                        "pattern" => Tok::Pattern,
                        "true" => Tok::Bool(true),
                        "false" => Tok::Bool(false),
                        "nil" | "void" | "None" => Tok::Nil,
                        "never" => Tok::Never,
                        "Any" => Tok::Any,
                        _ => Tok::ClassName(word),
                    };
                    self.push(tok, &mut out);
                }
                _ => return Err(self.illegal()),
            }
        }
        out.push(Token {
            tok: Tok::Eof,
            line: self.line,
        });
        Ok(out)
    }

    fn dot_allowed(&self) -> bool {
        match self.at(1) {
            Some(c) if c.is_ascii_digit() || c == '.' => false,
            _ => true,
        }
    }

    fn number_ahead(&self) -> bool {
        matches!(self.at(1), Some(c) if c.is_ascii_digit() || c == '.')
            && !(self.at(1) == Some('.') && !matches!(self.at(2), Some(d) if d.is_ascii_digit()))
    }

    fn number(&mut self) -> PyResult<Tok> {
        let start = self.pos;
        if self.cur() == Some('-') {
            self.pos += 1;
        }
        let mut is_float = false;
        if self.cur() == Some('.') {
            // -.5 / .5
            is_float = true;
            self.pos += 1;
            while matches!(self.cur(), Some(c) if c.is_ascii_digit()) {
                self.pos += 1;
            }
        } else {
            let digits_start = self.pos;
            while matches!(self.cur(), Some(c) if c.is_ascii_digit()) {
                self.pos += 1;
            }
            if self.pos == digits_start {
                return Err(self.illegal());
            }
            if self.cur() == Some('.') {
                is_float = true;
                self.pos += 1;
                while matches!(self.cur(), Some(c) if c.is_ascii_digit()) {
                    self.pos += 1;
                }
            }
        }
        let text: String = self.chars[start..self.pos].iter().collect();
        Ok(if is_float {
            Tok::Float(text)
        } else {
            Tok::Int(text)
        })
    }

    fn string(&mut self, quote: char) -> PyResult<Tok> {
        let triple = quote == '\'' && self.at(1) == Some('\'') && self.at(2) == Some('\'');
        if triple {
            self.pos += 3;
            let start = self.pos;
            while self.pos + 2 < self.chars.len() {
                if self.chars[self.pos] == '\''
                    && self.chars[self.pos + 1] == '\''
                    && self.chars[self.pos + 2] == '\''
                {
                    let text: String = self.chars[start..self.pos].iter().collect();
                    self.pos += 3;
                    return Ok(Tok::Str(text));
                }
                if self.chars[self.pos] == '\n' {
                    self.line += 1;
                }
                self.pos += 1;
            }
            return Err(self.illegal());
        }
        self.pos += 1;
        let start = self.pos;
        while let Some(c) = self.cur() {
            if c == '\n' {
                return Err(self.illegal());
            }
            if c == '\\' {
                // The regex takes a backslash and the character after it.
                if self.at(1).is_none() || self.at(1) == Some('\n') {
                    return Err(self.illegal());
                }
                self.pos += 2;
                continue;
            }
            if c == quote {
                let text: String = self.chars[start..self.pos].iter().collect();
                self.pos += 1;
                return Ok(Tok::Str(text));
            }
            self.pos += 1;
        }
        Err(self.illegal())
    }

    fn hint_block(&mut self) -> PyResult<(Option<String>, usize)> {
        // self.cur() == '{'
        self.pos += 1;
        let start = self.pos;
        let mut depth = 1usize;
        let line = self.line;
        while let Some(c) = self.cur() {
            if c == '\n' {
                self.line += 1;
            } else if c == '{' {
                depth += 1;
            } else if c == '}' {
                depth -= 1;
                if depth == 0 {
                    let code: String = self.chars[start..self.pos].iter().collect();
                    self.pos += 1;
                    return Ok((Some(code), line));
                }
            }
            self.pos += 1;
        }
        Ok((None, self.line))
    }
}

fn word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

struct Parser<'py> {
    py: Python<'py>,
    nodes: Nodes<'py>,
    tokens: Vec<Token>,
    pos: usize,
}

impl<'py> Parser<'py> {
    fn new(py: Python<'py>, nodes: Nodes<'py>, tokens: Vec<Token>) -> Self {
        Self {
            py,
            nodes,
            tokens,
            pos: 0,
        }
    }

    fn peek(&self) -> &Tok {
        &self.tokens[self.pos].tok
    }

    fn peek_at(&self, offset: usize) -> &Tok {
        let index = (self.pos + offset).min(self.tokens.len() - 1);
        &self.tokens[index].tok
    }

    fn line(&self) -> usize {
        self.tokens[self.pos].line
    }

    fn bump(&mut self) -> Token {
        let token = self.tokens[self.pos].clone();
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        token
    }

    fn error_at(&self) -> PyErr {
        let token = &self.tokens[self.pos];
        if token.tok == Tok::Eof {
            PySyntaxError::new_err("Viba parse error: unexpected EOF")
        } else {
            PySyntaxError::new_err(format!(
                "Viba parse error: unexpected {} at line {}",
                self.repr_token(&token.tok),
                token.line
            ))
        }
    }

    fn repr_token(&self, tok: &Tok) -> String {
        let text = match tok {
            Tok::ClassName(s) | Tok::Tagged(s) | Tok::Str(s) => s.clone(),
            Tok::Int(s) => {
                let value = self
                    .py
                    .import("builtins")
                    .and_then(|b| b.getattr("int"))
                    .and_then(|f| f.call1((s.as_str(),)))
                    .ok();
                return value
                    .map(|v| pyast::repr_of(self.py, &v))
                    .unwrap_or_else(|| s.clone());
            }
            Tok::Float(s) => {
                let value = self
                    .py
                    .import("builtins")
                    .and_then(|b| b.getattr("float"))
                    .and_then(|f| f.call1((s.as_str(),)))
                    .ok();
                return value
                    .map(|v| pyast::repr_of(self.py, &v))
                    .unwrap_or_else(|| s.clone());
            }
            Tok::Bool(b) => {
                if *b {
                    "True".into()
                } else {
                    "False".into()
                }
            }
            Tok::Nil => "nil".into(),
            Tok::Never => "never".into(),
            Tok::Any => "Any".into(),
            Tok::Ellipsis => "...".into(),
            Tok::HintBlock(_) => "{...}".into(),
            Tok::Assign => "=".into(),
            Tok::Sum => "|".into(),
            Tok::Prod => "*".into(),
            Tok::Exp => "<-".into(),
            Tok::Apply => "<<".into(),
            Tok::Import => "import".into(),
            Tok::As => "as".into(),
            Tok::Pattern => "pattern".into(),
            Tok::LBracket => "[".into(),
            Tok::RBracket => "]".into(),
            Tok::LParen => "(".into(),
            Tok::RParen => ")".into(),
            Tok::Comma => ",".into(),
            Tok::Dot => ".".into(),
            Tok::Eof => return "EOF".into(),
        };
        pyast::repr_of(self.py, PyString::new(self.py, &text).as_any())
    }

    fn expect(&mut self, tok: Tok) -> PyResult<Token> {
        if *self.peek() == tok {
            Ok(self.bump())
        } else {
            Err(self.error_at())
        }
    }

    fn expect_class_name(&mut self) -> PyResult<(String, usize)> {
        let line = self.line();
        match self.peek().clone() {
            Tok::ClassName(name) => {
                self.bump();
                Ok((name, line))
            }
            _ => Err(self.error_at()),
        }
    }

    fn program(mut self) -> PyResult<Py<PyAny>> {
        let mut body = Vec::new();
        while *self.peek() != Tok::Eof {
            body.push(self.statement()?);
        }
        self.refuse_repeated_names(&body)?;
        for item in &body {
            self.name_the_definitions(item)?;
        }
        let mut folded = Vec::new();
        for item in &body {
            folded.push(self.fold_tagged_symbols(item)?);
        }
        self.check_definition_names(&folded)?;
        let items = pyast::pylist(self.py, folded)?;
        self.nodes.make("Module", &[("body", items)])
    }

    fn statement(&mut self) -> PyResult<Py<PyAny>> {
        match self.peek().clone() {
            Tok::Import => self.import_stmt(),
            Tok::Pattern => self.pattern_stmt(),
            Tok::ClassName(_) => {
                // A generic definition is a name, `[`, a name.  A definition is
                // a name, `=`.  Anything else is not a statement.
                if *self.peek_at(1) == Tok::LBracket {
                    self.generic_definition()
                } else {
                    self.type_definition()
                }
            }
            _ => Err(self.error_at()),
        }
    }

    fn type_definition(&mut self) -> PyResult<Py<PyAny>> {
        let (name, line) = self.expect_class_name()?;
        self.expect(Tok::Assign)?;
        let body = self.partial_expr()?;
        let node = self.nodes.make(
            "TypeDefinition",
            &[("name", pyast::pystr(self.py, &name)), ("body", body)],
        )?;
        pyast::set_lineno(self.py, node.bind(self.py), line)?;
        Ok(node)
    }

    fn generic_definition(&mut self) -> PyResult<Py<PyAny>> {
        let (name, line) = self.expect_class_name()?;
        self.expect(Tok::LBracket)?;
        let (first, _) = self.expect_class_name()?;
        let mut params = vec![first];
        while *self.peek() == Tok::Comma {
            self.bump();
            let (param, _) = self.expect_class_name()?;
            params.push(param);
        }
        self.expect(Tok::RBracket)?;
        self.expect(Tok::Assign)?;
        let body = self.partial_expr()?;
        let param_values = pyast::pylist(
            self.py,
            params
                .iter()
                .map(|param| pyast::pystr(self.py, param))
                .collect(),
        )?;
        let node = self.nodes.make(
            "GenericDefinition",
            &[
                ("name", pyast::pystr(self.py, &name)),
                ("generic_params", param_values),
                ("body", body),
            ],
        )?;
        pyast::set_lineno(self.py, node.bind(self.py), line)?;
        Ok(node)
    }

    fn import_stmt(&mut self) -> PyResult<Py<PyAny>> {
        let line = self.line();
        self.expect(Tok::Import)?;
        let (mut module, _) = self.expect_class_name()?;
        while *self.peek() == Tok::Dot {
            self.bump();
            let (part, _) = self.expect_class_name()?;
            module.push('.');
            module.push_str(&part);
        }
        let alias = if *self.peek() == Tok::As {
            self.bump();
            let (alias, _) = self.expect_class_name()?;
            pyast::pystr(self.py, &alias)
        } else {
            self.py.None()
        };
        let node = self.nodes.make(
            "Import",
            &[("module", pyast::pystr(self.py, &module)), ("alias", alias)],
        )?;
        pyast::set_lineno(self.py, node.bind(self.py), line)?;
        Ok(node)
    }

    fn pattern_stmt(&mut self) -> PyResult<Py<PyAny>> {
        let line = self.line();
        self.expect(Tok::Pattern)?;
        let pattern = self.partial_expr()?;
        let node = self.nodes.make("Pattern", &[("pattern", pattern)])?;
        pyast::set_lineno(self.py, node.bind(self.py), line)?;
        Ok(node)
    }

    fn partial_expr(&mut self) -> PyResult<Py<PyAny>> {
        let member_head = matches!(self.peek(), Tok::Tagged(_)) && *self.peek_at(1) == Tok::Apply;
        let mut left = if member_head {
            let token = self.bump();
            let tag = match token.tok {
                Tok::Tagged(tag) => tag,
                _ => unreachable!(),
            };
            self.nodes
                .make("Member", &[("tag", pyast::pystr(self.py, &tag))])?
        } else {
            self.adt_expr()?
        };
        while *self.peek() == Tok::Apply {
            let op_line = self.line();
            self.bump();
            let right = self.adt_expr()?;
            let node = self
                .nodes
                .make("Partial", &[("function", left), ("argument", right)])?;
            pyast::set_lineno(self.py, node.bind(self.py), op_line)?;
            let function = node.bind(self.py).getattr("function")?;
            pyast::set_lineno(self.py, &function, op_line)?;
            left = node;
        }
        Ok(left)
    }

    fn adt_expr(&mut self) -> PyResult<Py<PyAny>> {
        let mut left = self.product_expr()?;
        while *self.peek() == Tok::Sum {
            self.bump();
            let right = self.product_expr()?;
            left = self
                .nodes
                .make("Sum", &[("left", left), ("right", right)])?;
        }
        Ok(left)
    }

    fn product_expr(&mut self) -> PyResult<Py<PyAny>> {
        let mut left = self.exponent_expr()?;
        while *self.peek() == Tok::Prod {
            self.bump();
            let right = self.exponent_expr()?;
            left = self
                .nodes
                .make("Product", &[("left", left), ("right", right)])?;
        }
        Ok(left)
    }

    fn exponent_expr(&mut self) -> PyResult<Py<PyAny>> {
        let mut left = self.unary_expr()?;
        while *self.peek() == Tok::Exp {
            self.bump();
            let right = self.unary_expr()?;
            left = self
                .nodes
                .make("Exponent", &[("result", left), ("argument", right)])?;
        }
        Ok(left)
    }

    fn unary_expr(&mut self) -> PyResult<Py<PyAny>> {
        if matches!(self.peek(), Tok::Tagged(_)) {
            let mut tag = match self.bump().tok {
                Tok::Tagged(tag) => tag,
                _ => unreachable!(),
            };
            while *self.peek() == Tok::Dot && matches!(self.peek_at(1), Tok::ClassName(_)) {
                self.bump();
                let (part, _) = self.expect_class_name()?;
                tag.push('.');
                tag.push_str(&part);
            }
            let body = self.member_expr()?;
            let tag_value = pyast::pystr(self.py, &tag);
            return self
                .nodes
                .make("Tagged", &[("tag", tag_value), ("type", body)]);
        }
        self.member_expr()
    }

    fn member_expr(&mut self) -> PyResult<Py<PyAny>> {
        let mut left = self.type_app_expr()?;
        while *self.peek() == Tok::Dot {
            self.bump();
            let (name, _) = self.expect_class_name()?;
            left = self.nodes.make(
                "MemberTaken",
                &[("owner", left), ("name", pyast::pystr(self.py, &name))],
            )?;
        }
        Ok(left)
    }

    fn type_app_expr(&mut self) -> PyResult<Py<PyAny>> {
        if matches!(self.peek(), Tok::ClassName(_)) {
            let (mut name, _) = self.expect_class_name()?;
            while *self.peek() == Tok::Dot && matches!(self.peek_at(1), Tok::ClassName(_)) {
                self.bump();
                let (part, _) = self.expect_class_name()?;
                name.push('.');
                name.push_str(&part);
            }
            if *self.peek() == Tok::LBracket {
                self.bump();
                let mut args = Vec::new();
                if *self.peek() != Tok::RBracket {
                    args.push(self.partial_expr()?);
                    while *self.peek() == Tok::Comma {
                        self.bump();
                        args.push(self.partial_expr()?);
                    }
                }
                self.expect(Tok::RBracket)?;
                let args = pyast::pylist(self.py, args)?;
                let constructor = pyast::pystr(self.py, &name);
                return self
                    .nodes
                    .make("TypeApp", &[("constructor", constructor), ("args", args)]);
            }
            return self
                .nodes
                .make("TypeRef", &[("name", pyast::pystr(self.py, &name))]);
        }
        self.primary_expr()
    }

    fn primary_expr(&mut self) -> PyResult<Py<PyAny>> {
        let token = (*self.peek()).clone();
        match token {
            Tok::Int(text) => {
                self.bump();
                let value = self.number_value(&text, false)?;
                self.nodes.make("Constant", &[("value", value)])
            }
            Tok::Float(text) => {
                self.bump();
                let value = self.number_value(&text, true)?;
                self.nodes.make("Constant", &[("value", value)])
            }
            Tok::Str(text) => {
                self.bump();
                let value = pyast::pystr(self.py, &text);
                self.nodes.make("Constant", &[("value", value)])
            }
            Tok::Bool(value) => {
                self.bump();
                let value = pyast::pybool(self.py, value);
                self.nodes.make("Constant", &[("value", value)])
            }
            Tok::Nil => {
                self.bump();
                self.nodes.make0("Nil")
            }
            Tok::Never => {
                self.bump();
                self.nodes.make0("Never")
            }
            Tok::Any => {
                self.bump();
                self.nodes.make0("Any")
            }
            Tok::Ellipsis => {
                self.bump();
                self.nodes.make0("Ellipsis")
            }
            Tok::HintBlock(code) => {
                self.bump();
                self.nodes
                    .make("HintBlock", &[("code", pyast::pystr(self.py, &code))])
            }
            Tok::LParen => {
                self.bump();
                if *self.peek() == Tok::RParen {
                    self.bump();
                    return self
                        .nodes
                        .make("Tuple", &[("elements", pyast::pylist(self.py, vec![])?)]);
                }
                let first = self.partial_expr()?;
                if *self.peek() != Tok::Comma {
                    self.expect(Tok::RParen)?;
                    return Ok(first);
                }
                self.bump();
                let mut elements = vec![first];
                if *self.peek() != Tok::RParen {
                    loop {
                        elements.push(self.partial_expr()?);
                        if *self.peek() != Tok::Comma {
                            break;
                        }
                        self.bump();
                        if *self.peek() == Tok::RParen {
                            break;
                        }
                    }
                }
                self.expect(Tok::RParen)?;
                let elements = pyast::pylist(self.py, elements)?;
                self.nodes.make("Tuple", &[("elements", elements)])
            }
            _ => Err(self.error_at()),
        }
    }

    fn number_value(&self, text: &str, float: bool) -> PyResult<Py<PyAny>> {
        let builtins = self.py.import("builtins")?;
        let name = if float { "float" } else { "int" };
        Ok(builtins.getattr(name)?.call1((text,))?.unbind())
    }

    fn refuse_repeated_names(&self, definitions: &[Py<PyAny>]) -> PyResult<()> {
        let mut seen = std::collections::HashSet::new();
        for definition in definitions {
            let node = definition.bind(self.py);
            let class = pyast::class_name(node)?;
            if class == "TypeDefinition" || class == "GenericDefinition" {
                let name = pyast::field_string(node, "name")?;
                if !seen.insert(name.clone()) {
                    return Err(PySyntaxError::new_err(format!(
                        "Viba parse error: {} is defined twice: a module defines each name once",
                        name
                    )));
                }
            }
        }
        Ok(())
    }

    fn name_the_definitions(&self, node: &Py<PyAny>) -> PyResult<()> {
        let bound = node.bind(self.py);
        let class = pyast::class_name(bound)?;
        if class == "TypeDefinition" || class == "GenericDefinition" {
            let name = pyast::field_string(bound, "name")?;
            let body = bound.getattr("body")?;
            let replaced = self.self_named(&body, &name)?;
            bound.setattr("body", replaced)?;
        }
        Ok(())
    }

    fn self_named(&self, node: &Bound<'_, PyAny>, name: &str) -> PyResult<Py<PyAny>> {
        let py = self.py;
        if !self.is_ast(node)? {
            return Ok(node.clone().unbind());
        }
        let class = pyast::class_name(node)?;
        if class == "TypeApp" && pyast::field_string(node, "constructor")? == "tagged" {
            let args = pyast::field_list(node, "args")?;
            let mut folded = Vec::new();
            for arg in &args {
                folded.push(self.self_named(arg, name)?);
            }
            if !folded.is_empty() {
                let first = folded[0].bind(py);
                if pyast::class_name(first)? == "TypeRef"
                    && pyast::field_string(first, "name")? == "__type__"
                {
                    let constant = self
                        .nodes
                        .make("Constant", &[("value", pyast::pystr(py, name))])?;
                    folded[0] = constant;
                }
            }
            let args = pyast::pylist(py, folded)?;
            let constructor = pyast::field(node, "constructor")?;
            return self.nodes.make(
                "TypeApp",
                &[("constructor", constructor.unbind()), ("args", args)],
            );
        }
        let fields = node.getattr("_fields")?;
        for field in fields.try_iter()? {
            let field: String = field?.extract()?;
            let value = node.getattr(field.as_str())?;
            if value.is_instance_of::<PyList>() {
                let mut changed = Vec::new();
                let mut any = false;
                for item in value.try_iter()? {
                    let item = item?;
                    let replaced = self.self_named(&item, name)?;
                    if !replaced.is(&item) {
                        any = true;
                    }
                    changed.push(replaced);
                }
                if any {
                    let list = pyast::pylist(py, changed)?;
                    node.setattr(field.as_str(), list)?;
                }
            } else if self.is_ast(&value)? {
                let replaced = self.self_named(&value, name)?;
                if !replaced.is(&value) {
                    node.setattr(field.as_str(), replaced)?;
                }
            }
        }
        Ok(node.clone().unbind())
    }

    fn fold_tagged_symbols(&self, node: &Py<PyAny>) -> PyResult<Py<PyAny>> {
        let py = self.py;
        let bound = node.bind(py);
        if !self.is_ast(bound)? {
            return Ok(node.clone_ref(py));
        }
        let class = pyast::class_name(bound)?;
        if class == "TypeApp" && pyast::field_string(bound, "constructor")? == "tagged" {
            let args = pyast::field_list(bound, "args")?;
            let mut folded = Vec::new();
            for arg in &args {
                let owned = arg.clone().unbind();
                folded.push(self.fold_tagged_symbols(&owned)?);
            }
            let problem = self.tagged_problem(&folded)?;
            if let Some(problem) = problem {
                return Err(PySyntaxError::new_err(format!(
                    "Viba parse error: {}",
                    problem
                )));
            }
            let symbol = self.literal_symbol(folded.first().map(|v| v.bind(py)));
            if symbol.is_none() || folded.len() == 1 {
                let args = pyast::pylist(py, folded)?;
                let constructor = pyast::field(bound, "constructor")?;
                return self.nodes.make(
                    "TypeApp",
                    &[("constructor", constructor.unbind()), ("args", args)],
                );
            }
            let symbol = symbol.unwrap();
            let tag = format!("${}", symbol);
            if folded.len() == 1 {
                return self
                    .nodes
                    .make("Member", &[("tag", pyast::pystr(py, &tag))]);
            }
            let type_value = folded.remove(1);
            self.nodes.make(
                "Tagged",
                &[("tag", pyast::pystr(py, &tag)), ("type", type_value)],
            )
        } else {
            let fields = bound.getattr("_fields")?;
            for field in fields.try_iter()? {
                let field: String = field?.extract()?;
                let value = bound.getattr(field.as_str())?;
                if value.is_instance_of::<PyList>() {
                    let mut changed = Vec::new();
                    for item in value.try_iter()? {
                        changed.push(self.fold_tagged_symbols(&item?.unbind())?);
                    }
                    let list = pyast::pylist(py, changed)?;
                    bound.setattr(field.as_str(), list)?;
                } else if self.is_ast(&value)? {
                    let folded = self.fold_tagged_symbols(&value.unbind())?;
                    bound.setattr(field.as_str(), folded)?;
                }
            }
            Ok(node.clone_ref(py))
        }
    }

    fn tagged_problem(&self, arguments: &[Py<PyAny>]) -> PyResult<Option<String>> {
        let py = self.py;
        if arguments.len() != 1 && arguments.len() != 2 {
            return Ok(Some(format!(
                "tagged takes one argument (the symbol) or two (the symbol and the type it marks), not {}",
                arguments.len()
            )));
        }
        let first = arguments[0].bind(py);
        if pyast::class_name(first)? == "Constant" {
            let value = first.getattr("value")?;
            if self.symbol_of(&value).is_none() {
                let repr = pyast::repr_of(py, &value);
                return Ok(Some(format!("{} is no symbol for a tag", repr)));
            }
        }
        Ok(None)
    }

    fn literal_symbol(&self, node: Option<&Bound<'_, PyAny>>) -> Option<String> {
        let node = node?;
        if pyast::class_name(node).ok()? != "Constant" {
            return None;
        }
        let value = node.getattr("value").ok()?;
        self.symbol_of(&value)
    }

    fn symbol_of(&self, value: &Bound<'_, PyAny>) -> Option<String> {
        let text: String = value.extract().ok()?;
        let name = text.strip_prefix('$').unwrap_or(&text);
        if name.is_empty() {
            return None;
        }
        let mut chars = name.chars();
        let first = chars.next()?;
        if !(first.is_alphabetic() || first == '_') {
            return None;
        }
        if !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return None;
        }
        Some(name.to_string())
    }

    fn is_ast(&self, node: &Bound<'_, PyAny>) -> PyResult<bool> {
        let ast = self.nodes.class("AST")?;
        node.is_instance(&ast)
    }

    fn check_definition_names(&self, definitions: &[Py<PyAny>]) -> PyResult<()> {
        for definition in definitions {
            let node = definition.bind(self.py);
            let name = pyast::class_name(node)?;
            if name == "TypeDefinition" || name == "GenericDefinition" {
                check_builtin_name(&pyast::field_string(node, "name")?, "a definition name")?;
                if name == "GenericDefinition" {
                    for param in pyast::field_list(node, "generic_params")? {
                        check_builtin_name(&param.extract::<String>()?, "a generic parameter")?;
                    }
                }
            }
        }
        Ok(())
    }
}

fn builtin_description(name: &str) -> Option<&'static str> {
    match name {
        "list" | "set" | "dict" => Some("a builtin container"),
        "ListLiteral" | "SetLiteral" | "DictLiteral" => Some("a builtin literal constructor"),
        "tagged" => Some("a builtin tag constructor"),
        _ => None,
    }
}

fn check_builtin_name(name: &str, position: &str) -> PyResult<()> {
    if let Some(description) = builtin_description(name) {
        return Err(PySyntaxError::new_err(format!(
            "Viba parse error: {} is {}; it cannot be {}",
            name, description, position
        )));
    }
    Ok(())
}

pub fn parse(py: Python<'_>, source: &str) -> PyResult<Py<PyAny>> {
    let lexer = Lexer::new(py, source);
    let tokens = lexer.tokenize()?;
    let nodes = Nodes::new(py)?;
    Parser::new(py, nodes, tokens).program()
}
