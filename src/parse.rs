// Copyright 2016 The Fancy Regex Authors.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

//! A regex parser yielding an AST.

use compact_str::CompactString;
use regex_syntax::escape;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::collections::BTreeMap;
use std::str::FromStr;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use crate::codepoint_len;
use crate::CompileError;
use crate::Error;
use crate::ParseError;
use crate::Result;
use crate::MAX_RECURSION;

const FLAG_CASEI: u32 = 1;
const FLAG_MULTI: u32 = 1 << 1;
const FLAG_DOTNL: u32 = 1 << 2;
const FLAG_SWAP_GREED: u32 = 1 << 3;
const FLAG_IGNORE_SPACE: u32 = 1 << 4;
const FLAG_UNICODE: u32 = 1 << 5;

/// Regular expression AST. This is public for now but may change.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Expr {
    /// An empty expression, e.g. the last branch in `(a|b|)`
    Empty,
    /// Any character, regex `.`
    Any {
        /// Whether it also matches newlines or not
        newline: bool,
        /// CRLF mode
        crlf: bool,
    },
    /// An assertion
    Assertion(Assertion),
    /// The string as a literal, e.g. `a`
    Literal {
        /// The string to match
        val: CompactString,
        /// Whether match is case-insensitive or not
        casei: bool,
    },
    /// Concatenation of multiple expressions, must match in order, e.g. `a.` is a concatenation of
    /// the literal `a` and `.` for any character
    Concat(Vec<Expr>),
    /// Alternative of multiple expressions, one of them must match, e.g. `a|b` is an alternative
    /// where either the literal `a` or `b` must match
    Alt(Vec<Expr>),
    /// Capturing group of expression, e.g. `(a.)` matches `a` and any character and "captures"
    /// (remembers) the match
    Group(Box<Expr>),
    /// Look-around (e.g. positive/negative look-ahead or look-behind) with an expression, e.g.
    /// `(?=a)` means the next character must be `a` (but the match is not consumed)
    LookAround(Box<Expr>, LookAround),
    /// Repeat of an expression, e.g. `a*` or `a+` or `a{1,3}`
    Repeat {
        /// The expression that is being repeated
        child: Box<Expr>,
        /// The minimum number of repetitions
        lo: usize,
        /// The maximum number of repetitions (or `usize::MAX`)
        hi: usize,
        /// Greedy means as much as possible is matched, e.g. `.*b` would match all of `abab`.
        /// Non-greedy means as little as possible, e.g. `.*?b` would match only `ab` in `abab`.
        greedy: bool,
    },
    /// Delegate a regex to the regex crate. This is used as a simplification so that we don't have
    /// to represent all the expressions in the AST, e.g. character classes.
    Delegate {
        /// The regex
        inner: CompactString,
        /// How many characters the regex matches
        size: usize, // TODO: move into analysis result
        /// Whether the matching is case-insensitive or not
        casei: bool,
    },
    /// Back reference to a capture group, e.g. `\1` in `(abc|def)\1` references the captured group
    /// and the whole regex matches either `abcabc` or `defdef`.
    Backref(usize),
    /// Atomic non-capturing group, e.g. `(?>ab|a)` in text that contains `ab` will match `ab` and
    /// never backtrack and try `a`, even if matching fails after the atomic group.
    AtomicGroup(Box<Expr>),
    /// Keep matched text so far out of overall match
    KeepOut,
    /// Anchor to match at the position where the previous match ended
    ContinueFromPreviousMatchEnd,
    /// Conditional expression based on whether the numbered capture group matched or not
    BackrefExistsCondition(usize),
    /// If/Then/Else Condition. If there is no Then/Else, these will just be empty expressions.
    Conditional {
        /// The conditional expression to evaluate
        condition: Box<Expr>,
        /// What to execute if the condition is true
        true_branch: Box<Expr>,
        /// What to execute if the condition is false
        false_branch: Box<Expr>,
    },
}

/// Type of look-around assertion as used for a look-around expression.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LookAround {
    /// Look-ahead assertion, e.g. `(?=a)`
    LookAhead,
    /// Negative look-ahead assertion, e.g. `(?!a)`
    LookAheadNeg,
    /// Look-behind assertion, e.g. `(?<=a)`
    LookBehind,
    /// Negative look-behind assertion, e.g. `(?<!a)`
    LookBehindNeg,
}

/// Type of assertions
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Assertion {
    /// Start of input text
    StartText,
    /// End of input text
    EndText,
    /// Start of a line
    StartLine {
        /// CRLF mode
        crlf: bool,
    },
    /// End of a line
    EndLine {
        /// CRLF mode
        crlf: bool,
    },
    /// Left word boundary
    LeftWordBoundary,
    /// Right word boundary
    RightWordBoundary,
    /// Both word boundaries
    WordBoundary,
    /// Not word boundary
    NotWordBoundary,
}

impl Assertion {
    pub(crate) fn is_hard(&self) -> bool {
        use Assertion::*;
        matches!(
            self,
            // these will make regex-automata use PikeVM
            LeftWordBoundary | RightWordBoundary | WordBoundary | NotWordBoundary
        )
    }
}

impl Expr {
    /// Parse the regex and return an expression (AST) and a bit set with the indexes of groups
    /// that are referenced by backrefs.
    pub fn parse_tree(re: &str) -> Result<ExprTree> {
        ExprTree::parse(re)
    }

    fn to_ast(&self, capture_index: &mut u32) -> Result<regex_syntax::ast::Ast> {
        use regex_syntax::ast::*;
        // XXX: implement span?
        let span = Span::splat(Position::new(0, 0, 0));

        let with_flag = |flag, ast| {
            if let Some(flag) = flag {
                Ast::Group(Box::new(Group {
                    span,
                    kind: GroupKind::NonCapturing(Flags {
                        span,
                        items: vec![FlagsItem {
                            span,
                            kind: FlagsItemKind::Flag(flag),
                        }],
                    }),
                    ast: Box::new(ast),
                }))
            } else {
                ast
            }
        };

        let mut fetch_add_capture_index = || {
            let index = *capture_index;
            *capture_index += 1;
            index
        };

        Ok(match self {
            Expr::Empty => Ast::Empty(Box::new(span)),
            Expr::Any { newline, crlf } => with_flag(
                newline.then_some(Flag::DotMatchesNewLine),
                with_flag(crlf.then_some(Flag::CRLF), Ast::Dot(Box::new(span))),
            ),
            Expr::Literal { val, casei } => with_flag(
                casei.then_some(Flag::CaseInsensitive),
                Ast::Concat(Box::new(Concat {
                    span,
                    asts: val
                        .chars()
                        .map(|c| {
                            Ast::Literal(Box::new(Literal {
                                span,
                                kind: LiteralKind::Verbatim, // does not matter
                                c,
                            }))
                        })
                        .collect(),
                })),
            ),
            Expr::Assertion(assertion) => match assertion {
                self::Assertion::StartText => Ast::Assertion(Box::new(Assertion {
                    span,
                    kind: AssertionKind::StartText,
                })),
                self::Assertion::EndText => Ast::Assertion(Box::new(Assertion {
                    span,
                    kind: AssertionKind::EndText,
                })),
                self::Assertion::StartLine { crlf } => with_flag(
                    Some(Flag::MultiLine),
                    with_flag(
                        crlf.then_some(Flag::CRLF),
                        Ast::Assertion(Box::new(Assertion {
                            span,
                            kind: AssertionKind::StartLine,
                        })),
                    ),
                ),
                self::Assertion::EndLine { crlf } => with_flag(
                    Some(Flag::MultiLine),
                    with_flag(
                        crlf.then_some(Flag::CRLF),
                        Ast::Assertion(Box::new(Assertion {
                            span,
                            kind: AssertionKind::EndLine,
                        })),
                    ),
                ),
                _ => panic!("word boundaries are considered hard"),
            },
            Expr::Concat(children) => Ast::Concat(Box::new(Concat {
                span,
                asts: try_collect(
                    children
                        .into_iter()
                        .map(|child| child.to_ast(capture_index)),
                )?,
            })),
            Expr::Alt(children) => Ast::Alternation(Box::new(Alternation {
                span,
                asts: try_collect(
                    children
                        .into_iter()
                        .map(|child| child.to_ast(capture_index)),
                )?,
            })),
            Expr::Group(child) => Ast::Group(Box::new(Group {
                span,
                kind: GroupKind::CaptureIndex(fetch_add_capture_index()),
                ast: Box::new(child.to_ast(capture_index)?),
            })),
            Expr::Repeat {
                child,
                lo,
                hi,
                greedy,
            } => Ast::Repetition(Box::new(Repetition {
                span,
                op: RepetitionOp {
                    span,
                    kind: match (*lo, *hi) {
                        (0, 1) => RepetitionKind::ZeroOrOne,
                        (0, usize::MAX) => RepetitionKind::ZeroOrMore,
                        (1, usize::MAX) => RepetitionKind::OneOrMore,
                        (lo, hi) if lo == hi => {
                            RepetitionKind::Range(RepetitionRange::Exactly(lo.try_into().unwrap()))
                        }
                        (lo, usize::MAX) => {
                            RepetitionKind::Range(RepetitionRange::AtLeast(lo.try_into().unwrap()))
                        }
                        (lo, hi) => RepetitionKind::Range(RepetitionRange::Bounded(
                            lo.try_into().unwrap(),
                            hi.try_into().unwrap(),
                        )),
                    },
                },
                greedy: *greedy,
                ast: Box::new(child.to_ast(capture_index)?),
            })),
            Expr::Delegate { inner, casei, .. } => with_flag(
                casei.then_some(Flag::CaseInsensitive),
                parse_ast(&inner, capture_index)?,
            ),
            _ => panic!("attempting to convert hard expr"),
        })
    }

    /// Convert expression to a [`regex_syntax::hir::Hir`].
    pub fn to_hir<E: Borrow<Expr>, I: IntoIterator<Item = E>>(
        exprs: I,
    ) -> Result<regex_syntax::hir::Hir> {
        let mut capture_index = 1;
        let mut translator = regex_syntax::hir::translate::Translator::new();
        Ok(regex_syntax::hir::Hir::concat(try_collect(
            exprs
                .into_iter()
                .map(|expr| expr.borrow().to_ast(&mut capture_index))
                .map(|ast| {
                    // XXX: using empty pattern; this will make error info useless
                    translator.translate("", &ast?).map_err(|e| {
                        Error::CompileError(CompileError::InnerSyntaxError(Box::new(
                            regex_syntax::Error::Translate(e),
                        )))
                    })
                }),
        )?))
    }

    /// Convert expression to a string
    pub fn to_str(&self) -> Result<String> {
        Ok(format!("{}", Expr::to_hir([self])?))
    }
}

fn offset_capture_index(ast: &mut regex_syntax::ast::Ast, capture_index: u32) -> u32 {
    use regex_syntax::ast::*;

    let recur = |ast| offset_capture_index(ast, capture_index);

    match ast {
        Ast::Alternation(children) => children.asts.iter_mut().map(recur).max(),
        Ast::Concat(children) => children.asts.iter_mut().map(recur).max(),
        Ast::Repetition(child) => Some(recur(&mut child.ast)),
        Ast::Group(child) => match &mut child.kind {
            GroupKind::CaptureIndex(index) => {
                *index += capture_index;
                Some(*index + 1)
            }
            _ => None,
        },
        Ast::Assertion(_)
        | Ast::ClassBracketed(_)
        | Ast::ClassPerl(_)
        | Ast::ClassUnicode(_)
        | Ast::Dot(_)
        | Ast::Empty(_)
        | Ast::Flags(_)
        | Ast::Literal(_) => None,
    }
    .unwrap_or(capture_index)
}

fn parse_ast(pattern: &str, capture_index: &mut u32) -> Result<regex_syntax::ast::Ast> {
    let mut ast = regex_syntax::ast::parse::Parser::new()
        .parse(pattern)
        .map_err(|e| {
            Error::CompileError(CompileError::InnerSyntaxError(Box::new(
                regex_syntax::Error::Parse(e),
            )))
        })?;
    *capture_index = offset_capture_index(&mut ast, *capture_index);
    Ok(ast)
}

fn try_collect<T>(iter: impl IntoIterator<Item = Result<T>>) -> Result<Vec<T>> {
    let iter = iter.into_iter();
    let mut vec = Vec::with_capacity(iter.size_hint().0);
    for item in iter {
        vec.push(item?);
    }
    Ok(vec)
}

pub(crate) type NamedGroups = BTreeMap<CompactString, usize>;

/// Regular expression AST Tree.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExprTree {
    /// The expr
    pub expr: Expr,
    /// A mapping from group name to group index
    pub named_groups: NamedGroups,
}

impl ExprTree {
    /// Parse the regex and return an expression (AST) and a bit set with the indexes of groups
    /// that are referenced by backrefs.
    pub fn parse(re: &str) -> Result<ExprTree> {
        Parser::parse(re)
    }
}

#[derive(Debug)]
pub(crate) struct Parser<'a> {
    re: &'a str, // source
    flags: u32,
    named_groups: NamedGroups,
    numeric_backrefs: bool,
    curr_group: usize, // need to keep track of which group number we're parsing
}

impl<'a> Parser<'a> {
    /// Parse the regex and return an expression (AST) and a bit set with the indexes of groups
    /// that are referenced by backrefs.
    pub(crate) fn parse(re: &str) -> Result<ExprTree> {
        let mut p = Parser::new(re);
        let (ix, expr) = p.parse_re(0, 0)?;
        if ix < re.len() {
            return Err(Error::ParseError(
                ix,
                ParseError::GeneralParseError("end of string not reached".into()),
            ));
        }
        let expr = p.optimize(expr);
        Ok(ExprTree {
            expr,
            named_groups: p.named_groups,
        })
    }

    fn new(re: &str) -> Parser<'_> {
        Parser {
            re,
            named_groups: Default::default(),
            numeric_backrefs: false,
            flags: FLAG_UNICODE,
            curr_group: 0,
        }
    }

    fn parse_re(&mut self, ix: usize, depth: usize) -> Result<(usize, Expr)> {
        let (ix, child) = self.parse_branch(ix, depth)?;
        let mut ix = self.optional_whitespace(ix)?;
        if self.re[ix..].starts_with('|') {
            let mut children = vec![child];
            while self.re[ix..].starts_with('|') {
                ix += 1;
                let (next, child) = self.parse_branch(ix, depth)?;
                children.push(child);
                ix = self.optional_whitespace(next)?;
            }
            return Ok((ix, Expr::Alt(children)));
        }
        // can't have numeric backrefs and named backrefs
        if self.numeric_backrefs && !self.named_groups.is_empty() {
            return Err(Error::CompileError(CompileError::NamedBackrefOnly));
        }
        Ok((ix, child))
    }

    fn parse_branch(&mut self, ix: usize, depth: usize) -> Result<(usize, Expr)> {
        let mut children = Vec::new();
        let mut ix = ix;
        while ix < self.re.len() {
            let (next, child) = self.parse_piece(ix, depth)?;
            if next == ix {
                break;
            }
            if child != Expr::Empty {
                children.push(child);
            }
            ix = next;
        }
        match children.len() {
            0 => Ok((ix, Expr::Empty)),
            1 => Ok((ix, children.pop().unwrap())),
            _ => Ok((ix, Expr::Concat(children))),
        }
    }

    fn parse_piece(&mut self, ix: usize, depth: usize) -> Result<(usize, Expr)> {
        let (ix, child) = self.parse_atom(ix, depth)?;
        let mut ix = self.optional_whitespace(ix)?;
        if ix < self.re.len() {
            // fail when child is empty?
            let (lo, hi) = match self.re.as_bytes()[ix] {
                b'?' => (0, 1),
                b'*' => (0, usize::MAX),
                b'+' => (1, usize::MAX),
                b'{' => {
                    match self.parse_repeat(ix) {
                        Ok((next, lo, hi)) => {
                            ix = next - 1;
                            (lo, hi)
                        }
                        Err(_) => {
                            // Invalid repeat syntax, which results in `{` being treated as a literal
                            return Ok((ix, child));
                        }
                    }
                }
                _ => return Ok((ix, child)),
            };
            if !self.is_repeatable(&child) {
                return Err(Error::ParseError(ix, ParseError::TargetNotRepeatable));
            }
            ix += 1;
            ix = self.optional_whitespace(ix)?;
            let mut greedy = true;
            if ix < self.re.len() && self.re.as_bytes()[ix] == b'?' {
                greedy = false;
                ix += 1;
            }
            greedy ^= self.flag(FLAG_SWAP_GREED);
            let mut node = Expr::Repeat {
                child: Box::new(child),
                lo,
                hi,
                greedy,
            };
            if ix < self.re.len() && self.re.as_bytes()[ix] == b'+' {
                ix += 1;
                node = Expr::AtomicGroup(Box::new(node));
            }
            return Ok((ix, node));
        }
        Ok((ix, child))
    }

    fn is_repeatable(&self, child: &Expr) -> bool {
        match child {
            Expr::LookAround(_, _) => false,
            Expr::Empty => false,
            Expr::Assertion(_) => false,
            _ => true,
        }
    }

    // ix, lo, hi
    fn parse_repeat(&self, ix: usize) -> Result<(usize, usize, usize)> {
        let ix = self.optional_whitespace(ix + 1)?; // skip opening '{'
        let bytes = self.re.as_bytes();
        if ix == self.re.len() {
            return Err(Error::ParseError(ix, ParseError::InvalidRepeat));
        }
        let mut end = ix;
        let lo = if bytes[ix] == b',' {
            0
        } else if let Some((next, lo)) = parse_decimal(self.re, ix) {
            end = next;
            lo
        } else {
            return Err(Error::ParseError(ix, ParseError::InvalidRepeat));
        };
        let ix = self.optional_whitespace(end)?; // past lo number
        if ix == self.re.len() {
            return Err(Error::ParseError(ix, ParseError::InvalidRepeat));
        }
        end = ix;
        let hi = match bytes[ix] {
            b'}' => lo,
            b',' => {
                end = self.optional_whitespace(ix + 1)?; // past ','
                if let Some((next, hi)) = parse_decimal(self.re, end) {
                    end = next;
                    hi
                } else {
                    usize::MAX
                }
            }
            _ => return Err(Error::ParseError(ix, ParseError::InvalidRepeat)),
        };
        let ix = self.optional_whitespace(end)?; // past hi number
        if ix == self.re.len() || bytes[ix] != b'}' {
            return Err(Error::ParseError(ix, ParseError::InvalidRepeat));
        }
        Ok((ix + 1, lo, hi))
    }

    fn parse_atom(&mut self, ix: usize, depth: usize) -> Result<(usize, Expr)> {
        let ix = self.optional_whitespace(ix)?;
        if ix == self.re.len() {
            return Ok((ix, Expr::Empty));
        }
        match self.re.as_bytes()[ix] {
            b'.' => Ok((
                ix + 1,
                Expr::Any {
                    newline: self.flag(FLAG_DOTNL),
                    crlf: false,
                },
            )),
            b'^' => Ok((
                ix + 1,
                if self.flag(FLAG_MULTI) {
                    // TODO: support crlf flag
                    Expr::Assertion(Assertion::StartLine { crlf: false })
                } else {
                    Expr::Assertion(Assertion::StartText)
                },
            )),
            b'$' => Ok((
                ix + 1,
                if self.flag(FLAG_MULTI) {
                    // TODO: support crlf flag
                    Expr::Assertion(Assertion::EndLine { crlf: false })
                } else {
                    Expr::Assertion(Assertion::EndText)
                },
            )),
            b'(' => self.parse_group(ix, depth),
            b'\\' => {
                let (next, expr) = self.parse_escape(ix, false)?;
                Ok((next, expr))
            }
            b'+' | b'*' | b'?' | b'|' | b')' => Ok((ix, Expr::Empty)),
            b'[' => self.parse_class(ix),
            b => {
                // TODO: maybe want to match multiple codepoints?
                let next = ix + codepoint_len(b);
                Ok((
                    next,
                    Expr::Literal {
                        val: CompactString::from(&self.re[ix..next]),
                        casei: self.flag(FLAG_CASEI),
                    },
                ))
            }
        }
    }

    fn parse_named_backref(&self, ix: usize, open: &str, close: &str) -> Result<(usize, Expr)> {
        if let Some((id, skip)) = parse_id(&self.re[ix..], open, close) {
            let group = if let Some(group) = self.named_groups.get(id) {
                Some(*group)
            } else if let Ok(group) = id.parse::<isize>() {
                if group > 0 {
                    Some(group as usize)
                } else if group == 0 {
                    // XXX
                    return Err(Error::ParseError(ix, ParseError::InvalidBackref));
                } else {
                    self.curr_group.checked_add_signed(group + 1)
                }
            } else {
                None
            };
            if let Some(group) = group {
                return Ok((ix + skip, Expr::Backref(group)));
            }
            // here the name is parsed but it is invalid
            Err(Error::ParseError(
                ix,
                ParseError::InvalidGroupNameBackref(id.into()),
            ))
        } else {
            // in this case the name can't be parsed
            Err(Error::ParseError(ix, ParseError::InvalidGroupName))
        }
    }

    fn parse_numbered_backref(&mut self, ix: usize) -> Result<(usize, Expr)> {
        if let Some((end, group)) = parse_decimal(self.re, ix) {
            // protect BitSet against unreasonably large value
            if cfg!(feature = "partial_parse") || group < self.re.len() / 2 {
                self.numeric_backrefs = true;
                return Ok((end, Expr::Backref(group)));
            }
        }
        return Err(Error::ParseError(ix, ParseError::InvalidBackref));
    }

    // ix points to \ character
    fn parse_escape(&mut self, ix: usize, in_class: bool) -> Result<(usize, Expr)> {
        let bytes = self.re.as_bytes();
        let Some(b) = bytes.get(ix + 1).copied() else {
            return Err(Error::ParseError(ix, ParseError::TrailingBackslash));
        };
        let end = ix + 1 + codepoint_len(b);
        Ok(if is_digit(b) {
            return self.parse_numbered_backref(ix + 1);
        } else if matches!(b, b'k' | b'g') {
            // Named backref: \k<name>
            if bytes.get(end).copied() == Some(b'\'') {
                return self.parse_named_backref(end, "'", "'");
            } else {
                return self.parse_named_backref(end, "<", ">");
            }
        } else if b == b'A' && !in_class {
            (end, Expr::Assertion(Assertion::StartText))
        } else if b == b'z' && !in_class {
            (end, Expr::Assertion(Assertion::EndText))
        } else if b == b'b' && !in_class {
            if bytes.get(end).copied() == Some(b'{') {
                // Support for \b{...} is not implemented yet
                return Err(Error::ParseError(
                    ix,
                    ParseError::InvalidEscape(format!("\\{}", &self.re[ix + 1..end])),
                ));
            }
            (end, Expr::Assertion(Assertion::WordBoundary))
        } else if b == b'B' && !in_class {
            if bytes.get(end).copied() == Some(b'{') {
                // Support for \b{...} is not implemented yet
                return Err(Error::ParseError(
                    ix,
                    ParseError::InvalidEscape(format!("\\{}", &self.re[ix + 1..end])),
                ));
            }
            (end, Expr::Assertion(Assertion::NotWordBoundary))
        } else if b == b'<' && !in_class {
            (end, Expr::Assertion(Assertion::LeftWordBoundary))
        } else if b == b'>' && !in_class {
            (end, Expr::Assertion(Assertion::RightWordBoundary))
        } else if matches!(b | 32, b'd' | b's' | b'w') {
            (
                end,
                Expr::Delegate {
                    inner: CompactString::from(&self.re[ix..end]),
                    size: 1,
                    casei: self.flag(FLAG_CASEI),
                },
            )
        } else if (b | 32) == b'h' {
            let s = if b == b'h' {
                "[0-9A-Fa-f]"
            } else {
                "[^0-9A-Fa-f]"
            };
            (
                end,
                Expr::Delegate {
                    inner: CompactString::from(s),
                    size: 1,
                    casei: false,
                },
            )
        } else if b == b'x' {
            return self.parse_hex(end, 2);
        } else if b == b'u' {
            return self.parse_hex(end, 4);
        } else if b == b'U' {
            return self.parse_hex(end, 8);
        } else if (b | 32) == b'p' && end != bytes.len() {
            let mut end = end;
            let b = bytes[end];
            end += codepoint_len(b);
            if b == b'{' {
                loop {
                    if end == self.re.len() {
                        return Err(Error::ParseError(ix, ParseError::UnclosedUnicodeName));
                    }
                    let b = bytes[end];
                    if b == b'}' {
                        end += 1;
                        break;
                    }
                    end += codepoint_len(b);
                }
            }
            (
                end,
                Expr::Delegate {
                    inner: CompactString::from(&self.re[ix..end]),
                    size: 1,
                    casei: self.flag(FLAG_CASEI),
                },
            )
        } else if b == b'K' {
            (end, Expr::KeepOut)
        } else if b == b'G' {
            (end, Expr::ContinueFromPreviousMatchEnd)
        } else {
            // printable ASCII (including space, see issue #29)
            (
                end,
                make_literal(match b {
                    b'a' => "\x07", // BEL
                    b'b' => "\x08", // BS
                    b'f' => "\x0c", // FF
                    b'n' => "\n",   // LF
                    b'r' => "\r",   // CR
                    b't' => "\t",   // TAB
                    b'v' => "\x0b", // VT
                    b'e' => "\x1b", // ESC
                    b' ' => " ",
                    b => {
                        let s = &self.re[ix + 1..end];
                        // we shall be permissive in production
                        if cfg!(debug_assertions) && b.is_ascii_alphanumeric() {
                            return Err(Error::ParseError(
                                ix,
                                ParseError::InvalidEscape(format!("\\{}", s)),
                            ));
                        } else {
                            s
                        }
                    }
                }),
            )
        })
    }

    // ix points after '\x', eg to 'A0' or '{12345}', or after `\u` or `\U`
    fn parse_hex(&self, ix: usize, digits: usize) -> Result<(usize, Expr)> {
        if ix >= self.re.len() {
            // Incomplete escape sequence
            return Err(Error::ParseError(ix, ParseError::InvalidHex));
        }
        let bytes = self.re.as_bytes();
        let b = bytes[ix];
        let (end, s) = if ix + digits <= self.re.len()
            && bytes[ix..ix + digits].iter().all(|&b| is_hex_digit(b))
        {
            let end = ix + digits;
            (end, &self.re[ix..end])
        } else if b == b'{' {
            let starthex = ix + 1;
            let mut endhex = starthex;
            loop {
                if endhex == self.re.len() {
                    return Err(Error::ParseError(ix, ParseError::InvalidHex));
                }
                let b = bytes[endhex];
                if endhex > starthex && b == b'}' {
                    break;
                }
                if is_hex_digit(b) && endhex < starthex + 8 {
                    endhex += 1;
                } else {
                    return Err(Error::ParseError(ix, ParseError::InvalidHex));
                }
            }
            (endhex + 1, &self.re[starthex..endhex])
        } else {
            return Err(Error::ParseError(ix, ParseError::InvalidHex));
        };
        let codepoint = u32::from_str_radix(s, 16).unwrap();
        if let Some(c) = ::std::char::from_u32(codepoint) {
            let mut inner = CompactString::default();
            inner.push(c);
            Ok((
                end,
                Expr::Literal {
                    val: inner,
                    casei: self.flag(FLAG_CASEI),
                },
            ))
        } else {
            Err(Error::ParseError(ix, ParseError::InvalidCodepointValue))
        }
    }

    fn parse_class(&mut self, ix: usize) -> Result<(usize, Expr)> {
        let bytes = self.re.as_bytes();
        let mut ix = ix + 1; // skip opening '['
        let mut class = CompactString::default();
        let mut nest = 1;
        class.push('[');

        // Negated character class
        if bytes.get(ix).copied() == Some(b'^') {
            class.push('^');
            ix += 1;
        }

        // `]` does not have to be escaped after opening `[` or `[^`
        if bytes.get(ix).copied() == Some(b']') {
            class.push(']');
            ix += 1;
        }

        loop {
            if ix == self.re.len() {
                return Err(Error::ParseError(ix, ParseError::InvalidClass));
            }
            let end = match bytes[ix] {
                b'\\' => {
                    // We support more escapes than regex, so parse it ourselves before delegating.
                    let (end, expr) = self.parse_escape(ix, true)?;
                    match expr {
                        Expr::Literal { val, .. } => {
                            debug_assert_eq!(val.chars().count(), 1);
                            class.push_str(&escape(&val));
                        }
                        Expr::Delegate { inner, .. } => {
                            class.push_str(&inner);
                        }
                        _ => {
                            return Err(Error::ParseError(ix, ParseError::InvalidClass));
                        }
                    }
                    end
                }
                b'[' => {
                    nest += 1;
                    class.push('[');
                    ix + 1
                }
                b']' => {
                    nest -= 1;
                    class.push(']');
                    if nest == 0 {
                        break;
                    }
                    ix + 1
                }
                b => {
                    let end = ix + codepoint_len(b);
                    class.push_str(&self.re[ix..end]);
                    end
                }
            };
            ix = end;
        }
        let class = Expr::Delegate {
            inner: class,
            size: 1,
            casei: self.flag(FLAG_CASEI),
        };
        let ix = ix + 1; // skip closing ']'
        Ok((ix, class))
    }

    fn parse_group(&mut self, ix: usize, depth: usize) -> Result<(usize, Expr)> {
        use LookAround::*;

        let depth = depth + 1;
        if depth >= MAX_RECURSION {
            return Err(Error::ParseError(ix, ParseError::RecursionExceeded));
        }
        let ix = self.optional_whitespace(ix + 1)?;
        let (la, skip) = if self.re[ix..].starts_with("?=") {
            (Some(LookAhead), 2)
        } else if self.re[ix..].starts_with("?!") {
            (Some(LookAheadNeg), 2)
        } else if self.re[ix..].starts_with("?<=") {
            (Some(LookBehind), 3)
        } else if self.re[ix..].starts_with("?<!") {
            (Some(LookBehindNeg), 3)
        } else if self.re[ix..].starts_with("?<") {
            // Named capture group using Oniguruma syntax: (?<name>...)
            self.curr_group += 1;
            if let Some((id, skip)) = parse_id(&self.re[ix + 1..], "<", ">") {
                self.named_groups.insert(id.into(), self.curr_group);
                (None, skip + 1)
            } else {
                return Err(Error::ParseError(ix, ParseError::InvalidGroupName));
            }
        } else if self.re[ix..].starts_with("?P<") {
            // Named capture group using Python syntax: (?P<name>...)
            self.curr_group += 1; // this is a capture group
            if let Some((id, skip)) = parse_id(&self.re[ix + 2..], "<", ">") {
                self.named_groups.insert(id.into(), self.curr_group);
                (None, skip + 2)
            } else {
                return Err(Error::ParseError(ix, ParseError::InvalidGroupName));
            }
        } else if self.re[ix..].starts_with("?P=") {
            // Backref using Python syntax: (?P=name)
            return self.parse_named_backref(ix + 3, "", ")");
        } else if self.re[ix..].starts_with("?>") {
            (None, 2)
        } else if self.re[ix..].starts_with("?(") {
            return self.parse_conditional(ix + 2, depth);
        } else if self.re[ix..].starts_with('?') {
            return self.parse_flags(ix, depth);
        } else {
            self.curr_group += 1; // this is a capture group
            (None, 0)
        };
        let ix = ix + skip;
        let (ix, child) = self.parse_re(ix, depth)?;
        let ix = self.check_for_close_paren(ix)?;
        let result = match (la, skip) {
            (Some(la), _) => Expr::LookAround(Box::new(child), la),
            (None, 2) => Expr::AtomicGroup(Box::new(child)),
            _ => Expr::Group(Box::new(child)),
        };
        Ok((ix, result))
    }

    fn check_for_close_paren(&self, ix: usize) -> Result<usize> {
        let ix = self.optional_whitespace(ix)?;
        if ix == self.re.len() {
            return Err(Error::ParseError(ix, ParseError::UnclosedOpenParen));
        } else if self.re.as_bytes()[ix] != b')' {
            return Err(Error::ParseError(
                ix,
                ParseError::GeneralParseError("expected close paren".into()),
            ));
        }
        Ok(ix + 1)
    }

    // ix points to `?` in `(?`
    fn parse_flags(&mut self, ix: usize, depth: usize) -> Result<(usize, Expr)> {
        let start = ix + 1;

        fn unknown_flag(re: &str, start: usize, end: usize) -> Error {
            let after_end = end + codepoint_len(re.as_bytes()[end]);
            let s = format!("(?{}", &re[start..after_end]);
            Error::ParseError(start, ParseError::UnknownFlag(s))
        }

        let mut ix = start;
        let mut neg = false;
        let oldflags = self.flags;
        loop {
            ix = self.optional_whitespace(ix)?;
            if ix == self.re.len() {
                return Err(Error::ParseError(ix, ParseError::UnclosedOpenParen));
            }
            let b = self.re.as_bytes()[ix];
            match b {
                b'i' => self.update_flag(FLAG_CASEI, neg),
                b'm' => self.update_flag(FLAG_MULTI, neg),
                b's' => self.update_flag(FLAG_DOTNL, neg),
                b'U' => self.update_flag(FLAG_SWAP_GREED, neg),
                b'x' => self.update_flag(FLAG_IGNORE_SPACE, neg),
                b'u' => {
                    if neg {
                        return Err(Error::ParseError(ix, ParseError::NonUnicodeUnsupported));
                    }
                }
                b'-' => {
                    if neg {
                        return Err(unknown_flag(self.re, start, ix));
                    }
                    neg = true;
                }
                b')' => {
                    if ix == start || neg && ix == start + 1 {
                        return Err(unknown_flag(self.re, start, ix));
                    }
                    return Ok((ix + 1, Expr::Empty));
                }
                b':' => {
                    if neg && ix == start + 1 {
                        return Err(unknown_flag(self.re, start, ix));
                    }
                    ix += 1;
                    let (ix, child) = self.parse_re(ix, depth)?;
                    if ix == self.re.len() {
                        return Err(Error::ParseError(ix, ParseError::UnclosedOpenParen));
                    } else if self.re.as_bytes()[ix] != b')' {
                        return Err(Error::ParseError(
                            ix,
                            ParseError::GeneralParseError("expected close paren".into()),
                        ));
                    };
                    self.flags = oldflags;
                    return Ok((ix + 1, child));
                }
                _ => return Err(unknown_flag(self.re, start, ix)),
            }
            ix += 1;
        }
    }

    // ix points to after the last ( in (?(
    fn parse_conditional(&mut self, ix: usize, depth: usize) -> Result<(usize, Expr)> {
        if ix >= self.re.len() {
            return Err(Error::ParseError(ix, ParseError::UnclosedOpenParen));
        }
        let bytes = self.re.as_bytes();
        // get the character after the open paren
        let b = bytes[ix];
        let (mut next, condition) = if is_digit(b) {
            self.parse_numbered_backref(ix)?
        } else if b == b'\'' {
            self.parse_named_backref(ix, "'", "'")?
        } else if b == b'<' {
            self.parse_named_backref(ix, "<", ">")?
        } else {
            self.parse_re(ix, depth)?
        };
        next = self.check_for_close_paren(next)?;
        let (end, child) = self.parse_re(next, depth)?;
        if end == next {
            // Backreference validity checker
            if let Expr::Backref(group) = condition {
                return Ok((end + 1, Expr::BackrefExistsCondition(group)));
            } else {
                return Err(Error::ParseError(
                    end,
                    ParseError::GeneralParseError(
                        "expected conditional to be a backreference or at least an expression for when the condition is true".into()
                    )
                ));
            }
        }
        let if_true: Expr;
        let mut if_false: Expr = Expr::Empty;
        if let Expr::Alt(mut alternatives) = child {
            // the truth branch will be the first alternative
            if_true = alternatives.remove(0);
            // if there is only one alternative left, take it out the Expr::Alt
            if alternatives.len() == 1 {
                if_false = alternatives.pop().expect("expected 2 alternatives");
            } else {
                // otherwise the remaining branches become the false branch
                if_false = Expr::Alt(alternatives);
            }
        } else {
            // there is only one branch - the truth branch. i.e. "if" without "else"
            if_true = child;
        }
        let inner_condition = if let Expr::Backref(group) = condition {
            Expr::BackrefExistsCondition(group)
        } else {
            condition
        };

        Ok((
            end + 1,
            if if_true == Expr::Empty && if_false == Expr::Empty {
                inner_condition
            } else {
                Expr::Conditional {
                    condition: Box::new(inner_condition),
                    true_branch: Box::new(if_true),
                    false_branch: Box::new(if_false),
                }
            },
        ))
    }

    fn flag(&self, flag: u32) -> bool {
        (self.flags & flag) != 0
    }

    fn update_flag(&mut self, flag: u32, neg: bool) {
        if neg {
            self.flags &= !flag;
        } else {
            self.flags |= flag;
        }
    }

    fn optional_whitespace(&self, mut ix: usize) -> Result<usize> {
        let bytes = self.re.as_bytes();
        loop {
            if ix == self.re.len() {
                return Ok(ix);
            }
            match bytes[ix] {
                b'#' if self.flag(FLAG_IGNORE_SPACE) => {
                    match bytes[ix..].iter().position(|&c| c == b'\n') {
                        Some(x) => ix += x + 1,
                        None => return Ok(self.re.len()),
                    }
                }
                b' ' | b'\r' | b'\n' | b'\t' if self.flag(FLAG_IGNORE_SPACE) => ix += 1,
                b'(' if bytes[ix..].starts_with(b"(?#") => {
                    ix += 3;
                    loop {
                        if ix >= self.re.len() {
                            return Err(Error::ParseError(ix, ParseError::UnclosedOpenParen));
                        }
                        match bytes[ix] {
                            b')' => {
                                ix += 1;
                                break;
                            }
                            b'\\' => ix += 2,
                            _ => ix += 1,
                        }
                    }
                }
                _ => return Ok(ix),
            }
        }
    }

    fn optimize(&self, mut expr: Expr) -> Expr {
        loop {
            let (new_expr, changed) = self.optimize_expr_pass(expr);
            expr = new_expr;
            if !changed {
                break expr;
            }
        }
    }

    fn optimize_expr_pass(&self, expr: Expr) -> (Expr, bool) {
        let changed = AtomicBool::new(false); // fuck Rust
        macro_rules! mark_change {
            ($expr:expr) => {{
                changed.fetch_or(true, Ordering::Relaxed);
                $expr
            }};
        }
        let recur = |expr| {
            let (expr, subchanged) = self.optimize_expr_pass(expr);
            changed.fetch_or(subchanged, Ordering::Relaxed);
            expr
        };
        (
            match expr {
                Expr::Concat(mut children) => {
                    children = children
                        .into_iter()
                        .map(recur)
                        .flat_map(|child| match child {
                            // flatten concat in concat
                            Expr::Concat(descents) => mark_change!(descents),
                            // eliminate empty literal
                            Expr::Literal { val, .. } if val.is_empty() => mark_change!(vec![]),
                            // no change
                            e => vec![e],
                        })
                        .fold(vec![], |mut children, item| {
                            let item = match (children.last_mut(), item) {
                                // merge literals
                                (
                                    Some(Expr::Literal {
                                        val: lval,
                                        casei: lcase,
                                    }),
                                    Expr::Literal {
                                        val: rval,
                                        casei: ref rcase,
                                    },
                                ) if lcase == rcase => mark_change! {{
                                    lval.push_str(&rval);
                                    None
                                }},
                                // merge delegate
                                (
                                    Some(Expr::Delegate {
                                        inner: linner,
                                        size: lsize,
                                        casei: lcase,
                                    }),
                                    Expr::Delegate {
                                        inner: rinner,
                                        size: rsize,
                                        casei: ref rcase,
                                    },
                                ) if lcase == rcase => mark_change! {{
                                    linner.push_str(&rinner);
                                    *lsize += rsize;
                                    None
                                }},
                                // no change
                                (_, item) => Some(item),
                            };
                            children.extend(item.into_iter());
                            children
                        });
                    if children.len() == 1 {
                        mark_change!(children.into_iter().next().unwrap())
                    } else {
                        Expr::Concat(children)
                    }
                }
                Expr::Alt(mut children) => {
                    children = children
                        .into_iter()
                        .map(recur)
                        .flat_map(|child| match child {
                            // flatten alt in alt
                            Expr::Alt(descents) => {
                                debug_assert!(!descents.is_empty());
                                mark_change!(descents)
                            }
                            // eliminate empty literal
                            Expr::Literal { val, .. } if val.is_empty() => {
                                mark_change!(vec![Expr::Empty])
                            }
                            // no change
                            e => vec![e],
                        })
                        .collect();
                    if children.len() == 1 {
                        mark_change!(children.into_iter().next().unwrap())
                    } else {
                        Expr::Alt(children)
                    }
                }
                Expr::Repeat {
                    child,
                    lo,
                    hi,
                    greedy,
                } => {
                    match recur(*child) {
                        // fold repeated literal
                        Expr::Literal { val, casei } if lo == hi && hi != usize::MAX => {
                            mark_change!(Expr::Literal {
                                val: {
                                    let len = val.len() * lo;
                                    [val].iter().cycle().take(lo).fold(
                                        CompactString::with_capacity(len),
                                        |mut acc, item| {
                                            acc.push_str(&item);
                                            acc
                                        },
                                    )
                                },
                                casei
                            })
                        }
                        // no change
                        child => Expr::Repeat {
                            child: Box::new(child),
                            lo,
                            hi,
                            greedy,
                        },
                    }
                }
                Expr::Group(child) => Expr::Group(Box::new(recur(*child))),
                Expr::LookAround(child, lookahead) => {
                    Expr::LookAround(Box::new(recur(*child)), lookahead)
                }
                Expr::AtomicGroup(child) => Expr::AtomicGroup(Box::new(recur(*child))),
                Expr::Conditional {
                    condition,
                    true_branch,
                    false_branch,
                } => Expr::Conditional {
                    condition: Box::new(recur(*condition)),
                    true_branch: Box::new(recur(*true_branch)),
                    false_branch: Box::new(recur(*false_branch)),
                },
                e => e,
            },
            changed.load(Ordering::Relaxed),
        )
    }
}

// return (ix, value)
pub(crate) fn parse_decimal(s: &str, ix: usize) -> Option<(usize, usize)> {
    let mut end = ix;
    while end < s.len() && is_digit(s.as_bytes()[end]) {
        end += 1;
    }
    usize::from_str(&s[ix..end]).ok().map(|val| (end, val))
}

/// Attempts to parse an identifier between the specified opening and closing
/// delimiters.  On success, returns `Some((id, skip))`, where `skip` is how much
/// of the string was used.
pub(crate) fn parse_id<'a>(s: &'a str, open: &'_ str, close: &'_ str) -> Option<(&'a str, usize)> {
    debug_assert!(!close.starts_with(is_id_char));

    if !s.starts_with(open) {
        return None;
    }

    let id_start = open.len();
    let id_len = match s[id_start..].find(|c: char| !is_id_char(c)) {
        Some(id_len) if s[id_start + id_len..].starts_with(close) => Some(id_len),
        None if close.is_empty() => Some(s.len()),
        _ => None,
    };
    match id_len {
        Some(0) => None,
        Some(id_len) => {
            let id_end = id_start + id_len;
            Some((&s[id_start..id_end], id_end + close.len()))
        }
        _ => None,
    }
}

fn is_id_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '-'
}

fn is_digit(b: u8) -> bool {
    b'0' <= b && b <= b'9'
}

fn is_hex_digit(b: u8) -> bool {
    is_digit(b) || (b'a' <= (b | 32) && (b | 32) <= b'f')
}

pub(crate) fn make_literal(s: &str) -> Expr {
    Expr::Literal {
        val: CompactString::from(s),
        casei: false,
    }
}

#[cfg(test)]
mod tests {
    use compact_str::CompactString;

    use crate::parse::{make_literal, parse_id};
    use crate::LookAround::*;
    use crate::{Assertion, Expr};

    fn p(s: &str) -> Expr {
        Expr::parse_tree(s).unwrap().expr
    }

    #[cfg_attr(feature = "track_caller", track_caller)]
    fn fail(s: &str) {
        assert!(Expr::parse_tree(s).is_err());
    }

    #[cfg_attr(feature = "track_caller", track_caller)]
    fn assert_error(re: &str, expected_error: &str) {
        let result = Expr::parse_tree(re);
        assert!(result.is_err());
        assert_eq!(&format!("{}", result.err().unwrap()), expected_error);
    }

    #[test]
    fn empty() {
        assert_eq!(p(""), Expr::Empty);
    }

    #[test]
    fn any() {
        assert_eq!(
            p("."),
            Expr::Any {
                newline: false,
                crlf: false
            }
        );
        assert_eq!(
            p("(?s:.)"),
            Expr::Any {
                newline: true,
                crlf: false
            }
        );
    }

    #[test]
    fn start_text() {
        assert_eq!(p("^"), Expr::Assertion(Assertion::StartText));
    }

    #[test]
    fn end_text() {
        assert_eq!(p("$"), Expr::Assertion(Assertion::EndText));
    }

    #[test]
    fn literal() {
        assert_eq!(p("a"), make_literal("a"));
    }

    #[test]
    fn literal_special() {
        assert_eq!(p("}"), make_literal("}"));
        assert_eq!(p("]"), make_literal("]"));
    }

    #[test]
    fn parse_id_test() {
        assert_eq!(parse_id("foo.", "", ""), Some(("foo", 3)));
        assert_eq!(parse_id("{foo}", "{", "}"), Some(("foo", 5)));
        assert_eq!(parse_id("{foo.", "{", "}"), None);
        assert_eq!(parse_id("{foo", "{", "}"), None);
        assert_eq!(parse_id("{}", "{", "}"), None);
        assert_eq!(parse_id("", "", ""), None);
    }

    #[test]
    fn literal_unescaped_opening_curly() {
        // `{` in position where quantifier is not allowed results in literal `{`
        assert_eq!(p("{"), make_literal("{"));
        assert_eq!(p("({)"), Expr::Group(Box::new(make_literal("{"),)));
        assert_eq!(
            p("a|{"),
            Expr::Alt(vec![make_literal("a"), make_literal("{"),])
        );
        assert_eq!(p("{{2}"), make_literal("{{"),);
    }

    #[test]
    fn literal_escape() {
        assert_eq!(p("\\'"), make_literal("'"));
        assert_eq!(p("\\\""), make_literal("\""));
        assert_eq!(p("\\ "), make_literal(" "));
        assert_eq!(p("\\xA0"), make_literal("\u{A0}"));
        assert_eq!(p("\\x{1F4A9}"), make_literal("\u{1F4A9}"));
        assert_eq!(p("\\x{000000B7}"), make_literal("\u{B7}"));
        assert_eq!(p("\\u21D2"), make_literal("\u{21D2}"));
        assert_eq!(p("\\u{21D2}"), make_literal("\u{21D2}"));
        assert_eq!(p("\\u21D2x"), p("\u{21D2}x"));
        assert_eq!(p("\\U0001F60A"), make_literal("\u{1F60A}"));
        assert_eq!(p("\\U{0001F60A}"), make_literal("\u{1F60A}"));
    }

    #[test]
    fn hex_escape() {
        assert_eq!(
            p("\\h"),
            Expr::Delegate {
                inner: CompactString::from("[0-9A-Fa-f]"),
                size: 1,
                casei: false
            }
        );
        assert_eq!(
            p("\\H"),
            Expr::Delegate {
                inner: CompactString::from("[^0-9A-Fa-f]"),
                size: 1,
                casei: false
            }
        );
    }

    #[test]
    fn invalid_escape() {
        assert_error(
            "\\",
            "Parsing error at position 0: Backslash without following character",
        );
        assert_error("\\q", "Parsing error at position 0: Invalid escape: \\q");
        assert_error("\\xAG", "Parsing error at position 2: Invalid hex escape");
        assert_error("\\xA", "Parsing error at position 2: Invalid hex escape");
        assert_error("\\x{}", "Parsing error at position 2: Invalid hex escape");
        assert_error("\\x{AG}", "Parsing error at position 2: Invalid hex escape");
        assert_error("\\x{42", "Parsing error at position 2: Invalid hex escape");
        assert_error(
            "\\x{D800}",
            "Parsing error at position 2: Invalid codepoint for hex or unicode escape",
        );
        assert_error(
            "\\x{110000}",
            "Parsing error at position 2: Invalid codepoint for hex or unicode escape",
        );
        assert_error("\\u123", "Parsing error at position 2: Invalid hex escape");
        assert_error("\\u123x", "Parsing error at position 2: Invalid hex escape");
        assert_error("\\u{}", "Parsing error at position 2: Invalid hex escape");
        assert_error(
            "\\U1234567",
            "Parsing error at position 2: Invalid hex escape",
        );
        assert_error("\\U{}", "Parsing error at position 2: Invalid hex escape");
    }

    #[test]
    fn concat() {
        assert_eq!(p("ab"), make_literal("ab"),);
    }

    #[test]
    fn alt() {
        assert_eq!(
            p("a|b"),
            Expr::Alt(vec![make_literal("a"), make_literal("b"),])
        );
    }

    #[test]
    fn group() {
        assert_eq!(p("(a)"), Expr::Group(Box::new(make_literal("a"),)));
    }

    #[test]
    fn group_repeat() {
        assert_eq!(
            p("(a){2}"),
            Expr::Repeat {
                child: Box::new(Expr::Group(Box::new(make_literal("a")))),
                lo: 2,
                hi: 2,
                greedy: true
            }
        );
    }

    #[test]
    fn repeat() {
        assert_eq!(
            p("a{2,42}"),
            Expr::Repeat {
                child: Box::new(make_literal("a")),
                lo: 2,
                hi: 42,
                greedy: true
            }
        );
        assert_eq!(
            p("a{2,}"),
            Expr::Repeat {
                child: Box::new(make_literal("a")),
                lo: 2,
                hi: usize::MAX,
                greedy: true
            }
        );
        assert_eq!(p("a{2}"), make_literal("aa"),);
        assert_eq!(
            p("a{,2}"),
            Expr::Repeat {
                child: Box::new(make_literal("a")),
                lo: 0,
                hi: 2,
                greedy: true
            }
        );

        assert_eq!(
            p("a{2,42}?"),
            Expr::Repeat {
                child: Box::new(make_literal("a")),
                lo: 2,
                hi: 42,
                greedy: false
            }
        );
        assert_eq!(
            p("a{2,}?"),
            Expr::Repeat {
                child: Box::new(make_literal("a")),
                lo: 2,
                hi: usize::MAX,
                greedy: false
            }
        );
        assert_eq!(p("a{2}?"), make_literal("aa"),);
        assert_eq!(
            p("a{,2}?"),
            Expr::Repeat {
                child: Box::new(make_literal("a")),
                lo: 0,
                hi: 2,
                greedy: false
            }
        );
    }

    #[test]
    fn invalid_repeat() {
        // Invalid repeat syntax results in literal
        assert_eq!(p("a{"), make_literal("a{"));
        assert_eq!(p("a{6"), make_literal("a{6"));
        assert_eq!(p("a{6,"), make_literal("a{6,"));
    }

    #[test]
    fn delegate_zero() {
        assert_eq!(p("\\b"), Expr::Assertion(Assertion::WordBoundary),);
        assert_eq!(p("\\B"), Expr::Assertion(Assertion::NotWordBoundary),);
    }

    #[test]
    fn delegate_named_group() {
        assert_eq!(
            p("\\p{Greek}"),
            Expr::Delegate {
                inner: CompactString::from("\\p{Greek}"),
                size: 1,
                casei: false
            }
        );
        assert_eq!(
            p("\\pL"),
            Expr::Delegate {
                inner: CompactString::from("\\pL"),
                size: 1,
                casei: false
            }
        );
        assert_eq!(
            p("\\P{Greek}"),
            Expr::Delegate {
                inner: CompactString::from("\\P{Greek}"),
                size: 1,
                casei: false
            }
        );
        assert_eq!(
            p("\\PL"),
            Expr::Delegate {
                inner: CompactString::from("\\PL"),
                size: 1,
                casei: false
            }
        );
        assert_eq!(
            p("(?i)\\p{Ll}"),
            Expr::Delegate {
                inner: CompactString::from("\\p{Ll}"),
                size: 1,
                casei: true
            }
        );
    }

    #[test]
    fn backref() {
        assert_eq!(
            p("(.)\\1"),
            Expr::Concat(vec![
                Expr::Group(Box::new(Expr::Any {
                    newline: false,
                    crlf: false
                })),
                Expr::Backref(1),
            ])
        );
    }

    #[test]
    fn named_backref() {
        assert_eq!(
            p("(?<i>.)\\k<i>"),
            Expr::Concat(vec![
                Expr::Group(Box::new(Expr::Any {
                    newline: false,
                    crlf: false
                })),
                Expr::Backref(1),
            ])
        );
    }

    #[test]
    fn lookaround() {
        assert_eq!(
            p("(?=a)"),
            Expr::LookAround(Box::new(make_literal("a")), LookAhead)
        );
        assert_eq!(
            p("(?!a)"),
            Expr::LookAround(Box::new(make_literal("a")), LookAheadNeg)
        );
        assert_eq!(
            p("(?<=a)"),
            Expr::LookAround(Box::new(make_literal("a")), LookBehind)
        );
        assert_eq!(
            p("(?<!a)"),
            Expr::LookAround(Box::new(make_literal("a")), LookBehindNeg)
        );
    }

    #[test]
    fn shy_group() {
        assert_eq!(p("(?:ab)c"), make_literal("abc"),);
    }

    #[test]
    fn flag_state() {
        assert_eq!(
            p("(?s)."),
            Expr::Any {
                newline: true,
                crlf: false
            }
        );
        assert_eq!(
            p("(?s:(?-s:.))"),
            Expr::Any {
                newline: false,
                crlf: false
            }
        );
        assert_eq!(
            p("(?s:.)."),
            Expr::Concat(vec![
                Expr::Any {
                    newline: true,
                    crlf: false
                },
                Expr::Any {
                    newline: false,
                    crlf: false
                },
            ])
        );
        assert_eq!(
            p("(?:(?s).)."),
            Expr::Concat(vec![
                Expr::Any {
                    newline: true,
                    crlf: false
                },
                Expr::Any {
                    newline: false,
                    crlf: false
                },
            ])
        );
    }

    #[test]
    fn flag_multiline() {
        assert_eq!(p("^"), Expr::Assertion(Assertion::StartText));
        assert_eq!(
            p("(?m:^)"),
            Expr::Assertion(Assertion::StartLine { crlf: false })
        );
        assert_eq!(p("$"), Expr::Assertion(Assertion::EndText));
        assert_eq!(
            p("(?m:$)"),
            Expr::Assertion(Assertion::EndLine { crlf: false })
        );
    }

    #[test]
    fn flag_swap_greed() {
        assert_eq!(p("a*"), p("(?U:a*?)"));
        assert_eq!(p("a*?"), p("(?U:a*)"));
    }

    #[test]
    fn invalid_flags() {
        assert!(Expr::parse_tree("(?").is_err());
        assert!(Expr::parse_tree("(?)").is_err());
        assert!(Expr::parse_tree("(?-)").is_err());
        assert!(Expr::parse_tree("(?-:a)").is_err());
        assert!(Expr::parse_tree("(?q:a)").is_err());
    }

    #[test]
    fn lifetime() {
        assert_eq!(
            p("\\'[a-zA-Z_][a-zA-Z0-9_]*(?!\\')\\b"),
            Expr::Concat(vec![
                make_literal("'"),
                Expr::Delegate {
                    inner: CompactString::from("[a-zA-Z_]"),
                    size: 1,
                    casei: false
                },
                Expr::Repeat {
                    child: Box::new(Expr::Delegate {
                        inner: CompactString::from("[a-zA-Z0-9_]"),
                        size: 1,
                        casei: false
                    }),
                    lo: 0,
                    hi: usize::MAX,
                    greedy: true
                },
                Expr::LookAround(Box::new(make_literal("'")), LookAheadNeg),
                Expr::Assertion(Assertion::WordBoundary),
            ])
        );
    }

    #[test]
    fn ignore_whitespace() {
        assert_eq!(p("(?x: )"), p(""));
        assert_eq!(p("(?x) | "), p("|"));
        assert_eq!(p("(?x: a )"), p("a"));
        assert_eq!(p("(?x: a # ) bobby tables\n b )"), p("ab"));
        assert_eq!(p("(?x: a | b )"), p("a|b"));
        assert_eq!(p("(?x: ( a b ) )"), p("(ab)"));
        assert_eq!(p("(?x: a + )"), p("a+"));
        assert_eq!(p("(?x: a {2} )"), p("a{2}"));
        assert_eq!(p("(?x: a { 2 } )"), p("a{2}"));
        assert_eq!(p("(?x: a { 2 , } )"), p("a{2,}"));
        assert_eq!(p("(?x: a { , 2 } )"), p("a{,2}"));
        assert_eq!(p("(?x: a { 2 , 3 } )"), p("a{2,3}"));
        assert_eq!(p("(?x: a { 2 , 3 } ? )"), p("a{2,3}?"));
        assert_eq!(p("(?x: ( ? i : . ) )"), p("(?i:.)"));
        assert_eq!(p("(?x: ( ?= a ) )"), p("(?=a)"));
        assert_eq!(p("(?x: [ ] )"), p("[ ]"));
        assert_eq!(p("(?x: [ ^] )"), p("[ ^]"));
        assert_eq!(p("(?x: [a - z] )"), p("[a - z]"));
        assert_eq!(p("(?x: [ \\] \\\\] )"), p("[ \\] \\\\]"));
        assert_eq!(p("(?x: a\\ b )"), p("a b"));
        assert_eq!(p("(?x: a (?-x:#) b )"), p("a#b"));
    }

    #[test]
    fn comments() {
        assert_eq!(p(r"ab(?# comment)"), p("ab"));
        assert_eq!(p(r"ab(?#)"), p("ab"));
        assert_eq!(p(r"(?# comment 1)(?# comment 2)ab"), p("ab"));
        assert_eq!(p(r"ab(?# comment \))c"), p("abc"));
        assert_eq!(p(r"ab(?# comment \\)c"), p("abc"));
        assert_eq!(p(r"ab(?# comment ()c"), p("abc"));
        assert_eq!(p(r"ab(?# comment)*"), p("ab*"));
        fail(r"ab(?# comment");
        fail(r"ab(?# comment\");
    }

    #[test]
    fn atomic_group() {
        assert_eq!(p("(?>a)"), Expr::AtomicGroup(Box::new(make_literal("a"))));
    }

    #[test]
    fn possessive() {
        assert_eq!(
            p("a++"),
            Expr::AtomicGroup(Box::new(Expr::Repeat {
                child: Box::new(make_literal("a")),
                lo: 1,
                hi: usize::MAX,
                greedy: true
            }))
        );
        assert_eq!(
            p("a*+"),
            Expr::AtomicGroup(Box::new(Expr::Repeat {
                child: Box::new(make_literal("a")),
                lo: 0,
                hi: usize::MAX,
                greedy: true
            }))
        );
        assert_eq!(
            p("a?+"),
            Expr::AtomicGroup(Box::new(Expr::Repeat {
                child: Box::new(make_literal("a")),
                lo: 0,
                hi: 1,
                greedy: true
            }))
        );
    }

    #[test]
    fn invalid_backref() {
        // only syntactic tests; see similar test in analyze module
        fail(".\\12345678"); // unreasonably large number
        fail(".\\c"); // not decimal
    }

    #[test]
    fn invalid_group_name_backref() {
        assert_error(
            "\\k<id>(?<id>.)",
            "Parsing error at position 2: Invalid group name in back reference: id",
        );
    }

    #[test]
    fn named_backref_only() {
        assert_error("(?<id>.)\\1", "Error compiling regex: Numbered backref/call not allowed because named group was used, use a named backref instead");
        assert_error("(a)\\1(?<name>b)", "Error compiling regex: Numbered backref/call not allowed because named group was used, use a named backref instead");
    }

    #[test]
    fn invalid_group_name() {
        assert_error(
            "(?<id)",
            "Parsing error at position 1: Could not parse group name",
        );
        assert_error(
            "(?<>)",
            "Parsing error at position 1: Could not parse group name",
        );
        assert_error(
            "(?<#>)",
            "Parsing error at position 1: Could not parse group name",
        );
        assert_error(
            "\\kxxx<id>",
            "Parsing error at position 2: Could not parse group name",
        );
    }

    #[test]
    fn unknown_flag() {
        assert_error(
            "(?-:a)",
            "Parsing error at position 2: Unknown group flag: (?-:",
        );
        assert_error(
            "(?)",
            "Parsing error at position 2: Unknown group flag: (?)",
        );
        assert_error(
            "(?--)",
            "Parsing error at position 2: Unknown group flag: (?--",
        );
        // Check that we don't split on char boundary
        assert_error(
            "(?\u{1F60A})",
            "Parsing error at position 2: Unknown group flag: (?\u{1F60A}",
        );
    }

    #[test]
    fn no_quantifiers_on_lookarounds() {
        assert_error(
            "(?=hello)+",
            "Parsing error at position 9: Target of repeat operator is invalid",
        );
        assert_error(
            "(?<!hello)*",
            "Parsing error at position 10: Target of repeat operator is invalid",
        );
        assert_error(
            "(?<=hello){2,3}",
            "Parsing error at position 14: Target of repeat operator is invalid",
        );
        assert_error(
            "(?!hello)?",
            "Parsing error at position 9: Target of repeat operator is invalid",
        );
        assert_error(
            "^?",
            "Parsing error at position 1: Target of repeat operator is invalid",
        );
        assert_error(
            "${2}",
            "Parsing error at position 3: Target of repeat operator is invalid",
        );
        assert_error(
            "(?m)^?",
            "Parsing error at position 5: Target of repeat operator is invalid",
        );
        assert_error(
            "(?m)${2}",
            "Parsing error at position 7: Target of repeat operator is invalid",
        );
        assert_error(
            "(a|b|?)",
            "Parsing error at position 5: Target of repeat operator is invalid",
        );
    }

    #[test]
    fn keepout() {
        assert_eq!(
            p("a\\Kb"),
            Expr::Concat(vec![make_literal("a"), Expr::KeepOut, make_literal("b"),])
        );
    }

    #[test]
    fn backref_exists_condition() {
        assert_eq!(
            p("(h)?(?(1))"),
            Expr::Concat(vec![
                Expr::Repeat {
                    child: Box::new(Expr::Group(Box::new(make_literal("h")))),
                    lo: 0,
                    hi: 1,
                    greedy: true
                },
                Expr::BackrefExistsCondition(1)
            ])
        );
        assert_eq!(
            p("(?<h>h)?(?('h'))"),
            Expr::Concat(vec![
                Expr::Repeat {
                    child: Box::new(Expr::Group(Box::new(make_literal("h")))),
                    lo: 0,
                    hi: 1,
                    greedy: true
                },
                Expr::BackrefExistsCondition(1)
            ])
        );
    }

    #[test]
    fn conditional_non_backref_validity_check_without_branches() {
        assert_error(
            "(?(foo))",
            "Parsing error at position 7: General parsing error: expected conditional to be a backreference or at least an expression for when the condition is true",
        );
    }

    #[test]
    fn conditional_invalid_target_of_repeat_operator() {
        assert_error(
            r"(?(?=\d)\w|!)",
            "Parsing error at position 3: Target of repeat operator is invalid",
        );
    }

    #[test]
    fn backref_condition_with_one_two_or_three_branches() {
        assert_eq!(
            p("(h)?(?(1)i|x)"),
            Expr::Concat(vec![
                Expr::Repeat {
                    child: Box::new(Expr::Group(Box::new(make_literal("h")))),
                    lo: 0,
                    hi: 1,
                    greedy: true
                },
                Expr::Conditional {
                    condition: Box::new(Expr::BackrefExistsCondition(1)),
                    true_branch: Box::new(make_literal("i")),
                    false_branch: Box::new(make_literal("x")),
                },
            ])
        );

        assert_eq!(
            p("(h)?(?(1)i)"),
            Expr::Concat(vec![
                Expr::Repeat {
                    child: Box::new(Expr::Group(Box::new(make_literal("h")))),
                    lo: 0,
                    hi: 1,
                    greedy: true
                },
                Expr::Conditional {
                    condition: Box::new(Expr::BackrefExistsCondition(1)),
                    true_branch: Box::new(make_literal("i")),
                    false_branch: Box::new(Expr::Empty),
                },
            ])
        );

        assert_eq!(
            p("(h)?(?(1)ii|xy|z)"),
            Expr::Concat(vec![
                Expr::Repeat {
                    child: Box::new(Expr::Group(Box::new(make_literal("h")))),
                    lo: 0,
                    hi: 1,
                    greedy: true
                },
                Expr::Conditional {
                    condition: Box::new(Expr::BackrefExistsCondition(1)),
                    true_branch: Box::new(make_literal("ii")),
                    false_branch: Box::new(Expr::Alt(vec![make_literal("xy"), make_literal("z"),])),
                },
            ])
        );

        assert_eq!(
            p("(?<cap>h)?(?(<cap>)ii|xy|z)"),
            Expr::Concat(vec![
                Expr::Repeat {
                    child: Box::new(Expr::Group(Box::new(make_literal("h")))),
                    lo: 0,
                    hi: 1,
                    greedy: true
                },
                Expr::Conditional {
                    condition: Box::new(Expr::BackrefExistsCondition(1)),
                    true_branch: Box::new(make_literal("ii")),
                    false_branch: Box::new(Expr::Alt(vec![make_literal("xy"), make_literal("z"),])),
                },
            ])
        );
    }

    #[test]
    fn conditional() {
        assert_eq!(
            p("((?(a)b|c))(\\1)"),
            Expr::Concat(vec![
                Expr::Group(Box::new(Expr::Conditional {
                    condition: Box::new(make_literal("a")),
                    true_branch: Box::new(make_literal("b")),
                    false_branch: Box::new(make_literal("c"))
                })),
                Expr::Group(Box::new(Expr::Backref(1)))
            ])
        );

        assert_eq!(
            p(r"^(?(\d)abc|\d!)$"),
            Expr::Concat(vec![
                Expr::Assertion(Assertion::StartText),
                Expr::Conditional {
                    condition: Box::new(Expr::Delegate {
                        inner: "\\d".into(),
                        size: 1,
                        casei: false,
                    }),
                    true_branch: Box::new(make_literal("abc")),
                    false_branch: Box::new(Expr::Concat(vec![
                        Expr::Delegate {
                            inner: "\\d".into(),
                            size: 1,
                            casei: false,
                        },
                        make_literal("!"),
                    ])),
                },
                Expr::Assertion(Assertion::EndText),
            ])
        );

        assert_eq!(
            p(r"(?((?=\d))\w|!)"),
            Expr::Conditional {
                condition: Box::new(Expr::LookAround(
                    Box::new(Expr::Delegate {
                        inner: "\\d".into(),
                        size: 1,
                        casei: false
                    }),
                    LookAhead
                )),
                true_branch: Box::new(Expr::Delegate {
                    inner: "\\w".into(),
                    size: 1,
                    casei: false,
                }),
                false_branch: Box::new(make_literal("!")),
            },
        );

        assert_eq!(
            p(r"(?((ab))c|d)"),
            Expr::Conditional {
                condition: Box::new(Expr::Group(Box::new(make_literal("ab")))),
                true_branch: Box::new(make_literal("c")),
                false_branch: Box::new(make_literal("d")),
            },
        );
    }

    // found by cargo fuzz, then minimized
    #[test]
    fn fuzz_1() {
        p(r"\ä");
    }

    #[test]
    fn fuzz_2() {
        p(r"\pä");
    }
}
