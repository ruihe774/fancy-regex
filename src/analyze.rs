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

//! Analysis of regex expressions.

use std::cmp::min;

use crate::parse::ExprTree;
use crate::CompileError;
use crate::Error;
use crate::Expr;
use crate::LookAround;
use crate::Result;

#[derive(Debug)]
pub struct Info<'a> {
    pub(crate) start_group: usize,
    pub(crate) end_group: usize,
    pub(crate) min_size: usize,
    pub(crate) const_size: bool,
    pub(crate) literal_segment: bool,
    pub(crate) must_exist: bool,
    pub(crate) offset: isize,
    pub(crate) longest_positive_lookahead: usize,
    pub(crate) longest_positive_lookbehind: usize,
    pub(crate) has_continue: bool,
    pub(crate) hard: bool,
    pub(crate) expr: &'a Expr,
    pub(crate) children: Vec<Info<'a>>,
}

struct Analyzer {
    group_ix: usize,
}

impl Analyzer {
    fn visit1<'a>(&mut self, expr: &'a Expr, must_exist: bool) -> Result<Info<'a>> {
        let start_group = self.group_ix;
        let mut children = Vec::new();
        let mut min_size = 0;
        let mut const_size = false;
        let mut literal_segment = false;
        let mut longest_positive_lookahead = 0;
        let mut longest_positive_lookbehind = 0;
        let mut has_continue = false;
        let mut hard = false;
        match *expr {
            Expr::Assertion(assertion) if assertion.is_hard() => {
                const_size = true;
                hard = true;
            }
            Expr::Empty | Expr::Assertion(_) => {
                const_size = true;
            }
            Expr::Any { .. } => {
                min_size = 1;
                const_size = true;
            }
            Expr::Literal { ref val, casei } => {
                min_size = val.chars().count();
                // our prefilter can handle ascii case-insensitive only
                literal_segment = (!casei || val.is_ascii()) && !val.is_empty();
                // very heuristic
                const_size = !casei
                    || val
                        .chars()
                        .all(|c| c.to_lowercase().count() == 1 && c.to_uppercase().count() == 1);
            }
            Expr::Concat(ref v) => {
                const_size = true;
                literal_segment = true;
                for child in v {
                    let child_info = self.visit1(child, must_exist)?;
                    min_size += child_info.min_size;
                    const_size &= child_info.const_size;
                    literal_segment &= child_info.literal_segment;
                    if child_info.longest_positive_lookahead != 0 {
                        debug_assert_eq!(longest_positive_lookahead, 0);
                        longest_positive_lookahead = child_info.longest_positive_lookahead;
                    }
                    if child_info.longest_positive_lookbehind != 0 {
                        debug_assert_eq!(longest_positive_lookbehind, 0);
                        longest_positive_lookbehind = child_info.longest_positive_lookbehind;
                    }
                    hard |= child_info.hard;
                    has_continue |= child_info.has_continue;
                    children.push(child_info);
                }
            }
            Expr::Alt(ref v) => {
                let child_info = self.visit1(&v[0], must_exist && children.len() == 1)?;
                min_size = child_info.min_size;
                const_size = child_info.const_size;
                literal_segment = child_info.literal_segment;
                has_continue = child_info.has_continue;
                hard = child_info.hard;
                children.push(child_info);
                for child in &v[1..] {
                    let child_info = self.visit1(child, false)?;
                    const_size &= child_info.const_size && min_size == child_info.min_size;
                    min_size = min(min_size, child_info.min_size);
                    literal_segment &= child_info.literal_segment;
                    longest_positive_lookahead =
                        longest_positive_lookahead.max(child_info.longest_positive_lookahead);
                    longest_positive_lookbehind =
                        longest_positive_lookbehind.max(child_info.longest_positive_lookbehind);
                    has_continue |= child_info.has_continue;
                    hard |= child_info.hard;
                    children.push(child_info);
                }
            }
            Expr::Group(ref child) => {
                self.group_ix += 1;
                let child_info = self.visit1(child, must_exist)?;
                min_size = child_info.min_size;
                const_size = child_info.const_size;
                literal_segment = child_info.literal_segment;
                longest_positive_lookahead = child_info.longest_positive_lookahead;
                longest_positive_lookbehind = child_info.longest_positive_lookbehind;
                has_continue |= child_info.has_continue;
                // If there's a backref to this group, we potentially have to backtrack within the
                // group. E.g. with `(x|xy)\1` and input `xyxy`, `x` matches but then the backref
                // doesn't, so we have to backtrack and try `xy`.
                hard = child_info.hard;
                children.push(child_info);
            }
            Expr::LookAround(ref child, look) => {
                let child_info = self.visit1(
                    child,
                    must_exist && matches!(look, LookAround::LookAhead | LookAround::LookBehind),
                )?;
                // min_size = 0
                const_size = true;
                literal_segment = child_info.literal_segment;
                let longest = match look {
                    LookAround::LookAhead => Some(&mut longest_positive_lookahead),
                    LookAround::LookBehind => Some(&mut longest_positive_lookbehind),
                    _ => None,
                };
                if let Some(longest) = longest {
                    // the same special treatment for Alt as in Compiler
                    if let Info {
                        const_size: false,
                        expr: &Expr::Alt(_),
                        ..
                    } = child_info
                    {
                        *longest = child_info
                            .children
                            .iter()
                            .map(|info| {
                                if info.const_size {
                                    info.min_size
                                } else {
                                    // XXX: a recursion is needed to find out actual length
                                    usize::MAX
                                }
                            })
                            .max()
                            .unwrap_or_default();
                    } else if child_info.const_size {
                        *longest = child_info.min_size;
                    } else {
                        // XXX
                        *longest = usize::MAX;
                    }
                }
                // for lookahead inside lookahead
                longest_positive_lookahead =
                    longest_positive_lookahead.max(child_info.longest_positive_lookahead);
                longest_positive_lookbehind =
                    longest_positive_lookbehind.max(child_info.longest_positive_lookbehind);
                has_continue |= child_info.has_continue;
                hard = true;
                children.push(child_info);
            }
            Expr::Repeat {
                ref child, lo, hi, ..
            } => {
                let child_info = self.visit1(child, must_exist && lo != 0)?;
                min_size = child_info.min_size * lo;
                const_size = child_info.const_size && lo == hi;
                literal_segment = child_info.literal_segment && lo != 0;
                longest_positive_lookahead = child_info.longest_positive_lookahead;
                longest_positive_lookbehind = child_info.longest_positive_lookbehind;
                has_continue |= child_info.has_continue;
                hard = child_info.hard;
                children.push(child_info);
            }
            Expr::Delegate { size, .. } => {
                // currently only used for empty and single-char matches
                min_size = size;
                const_size = true;
            }
            Expr::Backref(group) => {
                if group >= self.group_ix {
                    return Err(Error::CompileError(CompileError::InvalidBackref));
                }
                hard = true;
            }
            Expr::AtomicGroup(ref child) => {
                let child_info = self.visit1(child, must_exist)?;
                min_size = child_info.min_size;
                const_size = child_info.const_size;
                literal_segment = child_info.literal_segment;
                longest_positive_lookahead = child_info.longest_positive_lookahead;
                longest_positive_lookbehind = child_info.longest_positive_lookbehind;
                has_continue |= child_info.has_continue;
                hard = true; // TODO: possibly could weaken
                children.push(child_info);
            }
            Expr::KeepOut => {
                hard = true;
                const_size = true;
            }
            Expr::ContinueFromPreviousMatchEnd => {
                hard = true;
                const_size = true;
                has_continue = true;
            }
            Expr::BackrefExistsCondition(group) => {
                if group >= self.group_ix {
                    return Err(Error::CompileError(CompileError::InvalidBackref));
                }
                hard = true;
                const_size = true;
            }
            Expr::Conditional {
                ref condition,
                ref true_branch,
                ref false_branch,
            } => {
                hard = true;

                let child_info_condition = self.visit1(condition, false)?;
                let child_info_truth = self.visit1(true_branch, false)?;
                let child_info_false = self.visit1(false_branch, false)?;

                min_size = child_info_condition.min_size
                    + min(child_info_truth.min_size, child_info_false.min_size);
                const_size = child_info_condition.const_size
                    && child_info_truth.const_size
                    && child_info_false.const_size
                    // if the condition's size plus the truth branch's size is equal to the false branch's size then it's const size
                    && child_info_condition.min_size + child_info_truth.min_size == child_info_false.min_size;
                literal_segment =
                    child_info_truth.literal_segment && child_info_false.literal_segment;

                children.push(child_info_condition);
                children.push(child_info_truth);
                children.push(child_info_false);

                for child_info in &children {
                    longest_positive_lookahead =
                        longest_positive_lookahead.max(child_info.longest_positive_lookahead);
                    longest_positive_lookbehind =
                        longest_positive_lookbehind.max(child_info.longest_positive_lookbehind);
                    has_continue |= child_info.has_continue;
                }
            }
        };

        Ok(Info {
            expr,
            children,
            start_group,
            end_group: self.group_ix,
            min_size,
            const_size,
            literal_segment,
            must_exist,
            longest_positive_lookahead,
            longest_positive_lookbehind,
            has_continue,
            hard,
            offset: isize::MAX, // do it in next pass
        })
    }

    fn visit2(&self, info: &mut Info<'_>, offset: isize) -> Result<()> {
        if offset == isize::MAX {
            return Ok(());
        }
        match info.expr {
            Expr::Empty
            | Expr::Assertion(_)
            | Expr::Any { .. }
            | Expr::Literal { .. }
            | Expr::Delegate { .. }
            | Expr::Backref(_)
            | Expr::KeepOut
            | Expr::ContinueFromPreviousMatchEnd
            | Expr::BackrefExistsCondition(_) => {
                // do nothing
            }
            Expr::Concat(_) => {
                let mut offset = offset;
                for child_info in &mut info.children {
                    self.visit2(child_info, offset)?;
                    // offset == MIN || offset == MAX means we are out of range.
                    // I don't think it's posssible, though.
                    // Who will have such a long regex pattern string?
                    offset = (offset == isize::MIN).then_some(isize::MIN).unwrap_or(
                        offset.saturating_add_unsigned(
                            child_info
                                .const_size
                                .then_some(child_info.min_size)
                                .unwrap_or(usize::MAX),
                        ),
                    );
                }
            }
            Expr::Alt(_) => {
                for child_info in &mut info.children {
                    self.visit2(child_info, offset)?;
                }
            }
            Expr::Group(_) | Expr::AtomicGroup(_) | Expr::Repeat { .. } => {
                debug_assert_eq!(info.children.len(), 1);
                self.visit2(&mut info.children[0], offset)?;
            }
            Expr::LookAround(_, look) => {
                debug_assert_eq!(info.children.len(), 1);
                let child_info = &mut info.children[0];
                if child_info.const_size {
                    self.visit2(
                        child_info,
                        match look {
                            LookAround::LookAhead | LookAround::LookAheadNeg => offset,
                            LookAround::LookBehind | LookAround::LookBehindNeg => {
                                offset.saturating_sub_unsigned(child_info.min_size)
                            }
                        },
                    )?;
                } else {
                    // We don't check child != Alt and !const_size for children of children here.
                    // It will fail the compiler latter.
                    for info in &mut child_info.children {
                        self.visit2(
                            info,
                            match look {
                                LookAround::LookAhead | LookAround::LookAheadNeg => offset,
                                LookAround::LookBehind | LookAround::LookBehindNeg => {
                                    offset.saturating_sub_unsigned(info.min_size)
                                }
                            },
                        )?;
                    }
                }
            }
            Expr::Conditional { .. } => {
                debug_assert_eq!(info.children.len(), 3);
                let child_info = &mut info.children[0];
                self.visit2(child_info, offset)?;
                // XXX: merge duplicated code
                let new_offset = (offset == isize::MIN).then_some(isize::MIN).unwrap_or(
                    offset.saturating_add_unsigned(
                        child_info
                            .const_size
                            .then_some(child_info.min_size)
                            .unwrap_or(usize::MAX),
                    ),
                );
                self.visit2(&mut info.children[1], new_offset)?;
                self.visit2(&mut info.children[2], offset)?;
            }
        }
        info.offset = offset;
        Ok(())
    }

    fn visit<'a>(&mut self, expr: &'a Expr) -> Result<Info<'a>> {
        let mut info = self.visit1(expr, true)?;
        // if there is continue, no one uses offset
        if !info.has_continue {
            self.visit2(&mut info, 0)?;
        }
        Ok(info)
    }
}

/// Analyze the parsed expression to determine whether it requires fancy features.
pub fn analyze(tree: &ExprTree) -> Result<Info<'_>> {
    let mut analyzer = Analyzer { group_ix: 0 };

    analyzer.visit(&tree.expr)
}

#[cfg(test)]
mod tests {
    use regex_automata::meta::Regex;

    use super::analyze;
    use crate::Expr;

    #[test]
    fn case_folding_safe() {
        let re = Regex::new("(?i:ß)").unwrap();
        assert!(!re.is_match("SS"));

        // Another tricky example, Armenian ECH YIWN
        let re = Regex::new("(?i:\\x{0587})").unwrap();
        assert!(!re.is_match("\u{0565}\u{0582}"));
    }

    #[test]
    fn invalid_backref_1() {
        assert!(analyze(&Expr::parse_tree(".\\0").unwrap()).is_err());
    }

    #[test]
    fn invalid_backref_2() {
        assert!(analyze(&Expr::parse_tree("(.\\1)").unwrap()).is_err());
    }

    #[test]
    fn invalid_backref_3() {
        assert!(analyze(&Expr::parse_tree("\\1(.)").unwrap()).is_err());
    }
}
