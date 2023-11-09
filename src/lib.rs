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

/*!
An implementation of regexes, supporting a relatively rich set of features, including backreferences
and lookaround.

It builds on top of the excellent [regex] crate. If you are not
familiar with it, make sure you read its documentation and maybe you don't even need fancy-regex.

If your regex or parts of it does not use any special features, the matching is delegated to the
regex crate. That means it has linear runtime. But if you use "fancy" features such as
backreferences or look-around, an engine with backtracking needs to be used. In that case, the regex
can be slow and take exponential time to run because of what is called "catastrophic backtracking".
This depends on the regex and the input.

# Usage

The API should feel very similar to the regex crate, and involves compiling a regex and then using
it to find matches in text.

## Example: Matching text

An example with backreferences to check if a text consists of two identical words:

```rust
use fancy_regex::Regex;

let re = Regex::new(r"^(\w+) (\1)$").unwrap();
let result = re.is_match("foo foo");

assert!(result.is_ok());
let did_match = result.unwrap();
assert!(did_match);
```

Note that like in the regex crate, the regex needs anchors like `^` and `$` to match against the
entire input text.

## Example: Finding the position of matches

```rust
use fancy_regex::Regex;

let re = Regex::new(r"(\d)\1").unwrap();
let result = re.find("foo 22");

assert!(result.is_ok(), "execution was successful");
let match_option = result.unwrap();

assert!(match_option.is_some(), "found a match");
let m = match_option.unwrap();

assert_eq!(m.start(), 4);
assert_eq!(m.end(), 6);
assert_eq!(m.as_str(), "22");
```

## Example: Capturing groups

```rust
use fancy_regex::Regex;

let re = Regex::new(r"(?<!AU)\$(\d+)").unwrap();
let result = re.captures("AU$10, $20");

let captures = result.expect("Error running regex").expect("No match found");
let group = captures.get(1).expect("No group");
assert_eq!(group.as_str(), "20");
```

# Syntax

The regex syntax is based on the [regex] crate's, with some additional supported syntax.

Escapes:

`\h`
: hex digit (`[0-9A-Fa-f]`) \
`\H`
: not hex digit (`[^0-9A-Fa-f]`) \
`\e`
: escape control character (`\x1B`) \
`\K`
: keep text matched so far out of the overall match ([docs](https://www.regular-expressions.info/keep.html))\
`\G`
: anchor to where the previous match ended ([docs](https://www.regular-expressions.info/continue.html))

Backreferences:

`\1`
: match the exact string that the first capture group matched \
`\2`
: backref to the second capture group, etc

Named capture groups:

`(?<name>exp)`
: match *exp*, creating capture group named *name* \
`\k<name>`
: match the exact string that the capture group named *name* matched \
`(?P<name>exp)`
: same as `(?<name>exp)` for compatibility with Python, etc. \
`(?P=name)`
: same as `\k<name>` for compatibility with Python, etc.

Look-around assertions for matching without changing the current position:

`(?=exp)`
: look-ahead, succeeds if *exp* matches to the right of the current position \
`(?!exp)`
: negative look-ahead, succeeds if *exp* doesn't match to the right \
`(?<=exp)`
: look-behind, succeeds if *exp* matches to the left of the current position \
`(?<!exp)`
: negative look-behind, succeeds if *exp* doesn't match to the left

Atomic groups using `(?>exp)` to prevent backtracking within `exp`, e.g.:

```
# use fancy_regex::Regex;
let re = Regex::new(r"^a(?>bc|b)c$").unwrap();
assert!(re.is_match("abcc").unwrap());
// Doesn't match because `|b` is never tried because of the atomic group
assert!(!re.is_match("abc").unwrap());
```

Conditionals - if/then/else:

`(?(1))`
: continue only if first capture group matched \
`(?(<name>))`
: continue only if capture group named *name* matched \
`(?(1)true_branch|false_branch)`
: if the first capture group matched then execute the `true_branch` regex expression, else execute `false_branch` ([docs](https://www.regular-expressions.info/conditional.html)) \
`(?(condition)true_branch|false_branch)`
: if the condition matches then execute the `true_branch` regex expression, else execute `false_branch` from the point just before the condition was evaluated

[regex]: https://crates.io/crates/regex
*/

#![doc(html_root_url = "https://docs.rs/fancy-regex/0.11.0")]
#![deny(missing_docs)]
#![deny(missing_debug_implementations)]
#![warn(clippy::pedantic)]
#![allow(clippy::enum_glob_use)]
#![allow(clippy::if_not_else)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::redundant_else)]
#![allow(clippy::similar_names)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::wildcard_imports)]

use regex_automata::util::pool::{Pool, PoolGuard};
use std::borrow::Cow;
use std::iter::FusedIterator;
use std::num::NonZeroUsize;
use std::ops::{Index, Range};
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::slice::ChunksExact;
use std::str::FromStr;
use std::sync::Arc;
use vm::{Machine, Session, DEFAULT_BACKTRACK_LIMIT, DEFAULT_MAX_STACK};

mod analyze;
mod compile;
mod error;
mod expand;
mod parse;
mod prefilter;
mod replacer;
mod vm;

use crate::analyze::analyze;
use crate::compile::compile_with_options;
use crate::parse::NamedGroups;
pub use crate::parse::{Assertion, Expr, ExprTree, LookAround};
use crate::vm::OPTION_SKIPPED_EMPTY_MATCH;

pub use crate::error::{CompileError, Error, ParseError, Result, RuntimeError};
pub use crate::expand::Expander;
pub use crate::replacer::{NoExpand, Replacer, ReplacerRef};

const MAX_RECURSION: usize = 64;

// the public API

/// A compiled regular expression.
#[derive(Debug)]
pub struct Regex {
    pattern: Option<Arc<String>>,
    tree: Arc<ExprTree>,
    machine: Machine,
    session: Pool<Session, Box<dyn Fn() -> Session + Send + Sync + UnwindSafe + RefUnwindSafe>>,
    saves: Pool<Vec<usize>, fn() -> Vec<usize>>,
    n_groups: usize,
}

impl Regex {
    /// Parse and compile a regex with default options, see [`RegexBuilder`].
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if the pattern could not be parsed.
    #[inline]
    pub fn new(re: impl Into<String>) -> Result<Regex> {
        RegexBuilder::new().build(re)
    }

    fn new_with_source_and_options(source: RegexSource, options: RegexOptions) -> Result<Regex> {
        let (raw_tree, pattern) = match source {
            RegexSource::Pattern(pattern) => (Expr::parse_tree(pattern.as_str())?, Some(pattern)),
            RegexSource::ExprTree(tree) => (tree, None),
        };

        let tree = ExprTree {
            expr: Expr::Group(Box::new(raw_tree.expr)),
            ..raw_tree
        };

        let info = analyze(&tree)?;
        let prog = Arc::new(compile_with_options(&info, &tree.backrefs, options)?);
        let n_groups = info.end_group;
        let machine = Machine::new(prog.clone(), options.max_stack, options.backtrack_limit, 0);

        let raw_tree = ExprTree {
            expr: match tree.expr {
                Expr::Group(raw_expr) => *raw_expr,
                _ => unreachable!(),
            },
            ..tree
        };

        Ok(Regex {
            n_groups,
            pattern: pattern.map(Arc::new),
            tree: Arc::new(raw_tree),
            machine: machine.clone(),
            session: new_session_pool(machine),
            saves: new_saves_pool(),
        })
    }

    /// Returns the original pattern string used to create this regex.
    ///
    /// # Panics
    ///
    /// Panics if this regex is created by [`RegexBuilder::build_from_expr_tree()`].
    #[must_use]
    #[inline]
    pub fn as_str(&self) -> &str {
        self.pattern
            .as_ref()
            .expect("cannot get pattern as this regex is built from expr tree")
            .as_str()
    }

    /// Returns the expression tree of this regex.
    #[must_use]
    #[inline]
    pub fn as_expr_tree(&self) -> &ExprTree {
        &self.tree
    }

    /// Check if the regex matches the input text.
    ///
    /// # Example
    ///
    /// Test if some text contains the same word twice:
    ///
    /// ```rust
    /// # use fancy_regex::Regex;
    /// let re = Regex::new(r"(\w+) \1").unwrap();
    /// assert!(re.is_match("mirror mirror on the wall").unwrap());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an [`Error::RuntimeError`] for any runtime error occurred.
    #[inline]
    pub fn is_match(&self, text: &str) -> Result<bool> {
        let result = self.session.get().run(text, 0..text.len(), Some(0))?;
        Ok(result.is_some())
    }

    /// Returns an iterator for each successive non-overlapping match in `text`.
    ///
    /// If you have capturing groups in your regex that you want to extract, use the [`Regex::captures_iter()`] method.
    ///
    /// # Example
    ///
    /// Find all words followed by an exclamation point:
    ///
    /// ```rust
    /// # use fancy_regex::Regex;
    /// let re = Regex::new(r"\w+(?=!)").unwrap();
    /// let mut matches = re.find_iter("so fancy! even with! iterators!");
    /// assert_eq!(matches.next().unwrap().unwrap().as_str(), "fancy");
    /// assert_eq!(matches.next().unwrap().unwrap().as_str(), "with");
    /// assert_eq!(matches.next().unwrap().unwrap().as_str(), "iterators");
    /// assert!(matches.next().is_none());
    /// ```
    #[must_use]
    #[inline]
    pub fn find_iter<'r, 't>(&'r self, text: &'t str) -> Matches<'r, 't> {
        Matches {
            re: self,
            text,
            last_end: 0,
            last_match: None,
        }
    }

    /// Find the first match in the input text.
    ///
    /// If you have capturing groups in your regex that you want to extract, use the [`Regex::captures()`] method.
    ///
    /// # Example
    ///
    /// Find a word that is followed by an exclamation point:
    ///
    /// ```rust
    /// # use fancy_regex::Regex;
    /// let re = Regex::new(r"\w+(?=!)").unwrap();
    /// assert_eq!(re.find("so fancy!").unwrap().unwrap().as_str(), "fancy");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an [`Error::RuntimeError`] for any runtime error occurred.
    #[inline]
    pub fn find<'t>(&self, text: &'t str) -> Result<Option<Match<'t>>> {
        self.find_from_pos(text, 0)
    }

    /// Returns the first match in `text`, starting from the specified byte position `pos`.
    ///
    /// # Examples
    ///
    /// Finding match starting at a position:
    ///
    /// ```
    /// # use fancy_regex::Regex;
    /// let re = Regex::new(r"(?m:^)(\d+)").unwrap();
    /// let text = "1 test 123\n2 foo";
    /// let mat = re.find_from_pos(text, 7).unwrap().unwrap();
    ///
    /// assert_eq!(mat.start(), 11);
    /// assert_eq!(mat.end(), 12);
    /// ```
    ///
    /// Note that in some cases this is not the same as using the `find`
    /// method and passing a slice of the string, see [`Regex::captures_from_pos()`] for details.
    ///
    /// # Errors
    /// Returns an [`Error::RuntimeError`] for any runtime error occurred.
    #[inline]
    pub fn find_from_pos<'t>(&self, text: &'t str, pos: usize) -> Result<Option<Match<'t>>> {
        self.find_within_range(text, pos..text.len())
    }

    /// Returns the first match in `text` within the specified `range`.
    /// The `start` and `end` of `range` are treated as byte positions in `text`.
    ///
    /// # Examples
    ///
    /// Finding match starting at a position:
    ///
    /// ```
    /// # use fancy_regex::Regex;
    /// let re = Regex::new(r"(\d+)(?m:$)").unwrap();
    /// let text = "1 test 123\n2 foo";
    /// let mat = re.find_within_range(text, 7..10).unwrap().unwrap();
    ///
    /// assert_eq!(mat.start(), 7);
    /// assert_eq!(mat.end(), 10);
    /// ```
    ///
    /// Note that in some cases this is not the same as using the `find`
    /// method and passing a slice of the string, see [`Regex::captures_within_range()`] for details.
    ///
    /// # Errors
    /// Returns an [`Error::RuntimeError`] for any runtime error occurred.
    #[inline]
    pub fn find_within_range<'t>(
        &self,
        text: &'t str,
        range: impl Into<Range<usize>>,
    ) -> Result<Option<Match<'t>>> {
        self.find_within_range_with_option_flags(text, range, 0)
    }

    fn find_within_range_with_option_flags<'t>(
        &self,
        text: &'t str,
        range: impl Into<Range<usize>>,
        option_flags: u32,
    ) -> Result<Option<Match<'t>>> {
        let result =
            self.session
                .get()
                .run_with_options(text, range.into(), Some(1), option_flags)?;
        Ok(result.map(|saves| Match {
            text,
            start: saves[0],
            end: saves[1],
        }))
    }

    /// Returns an iterator over all the non-overlapping capture groups matched in `text`.
    ///
    /// # Examples
    ///
    /// Finding all matches and capturing parts of each:
    ///
    /// ```rust
    /// # use fancy_regex::Regex;
    /// let re = Regex::new(r"(\d{4})-(\d{2})").unwrap();
    /// let text = "It was between 2018-04 and 2020-01";
    /// let mut all_captures = re.captures_iter(text);
    ///
    /// let first = all_captures.next().unwrap().unwrap();
    /// assert_eq!(first.get(1).unwrap().as_str(), "2018");
    /// assert_eq!(first.get(2).unwrap().as_str(), "04");
    /// assert_eq!(first.get(0).unwrap().as_str(), "2018-04");
    ///
    /// let second = all_captures.next().unwrap().unwrap();
    /// assert_eq!(second.get(1).unwrap().as_str(), "2020");
    /// assert_eq!(second.get(2).unwrap().as_str(), "01");
    /// assert_eq!(second.get(0).unwrap().as_str(), "2020-01");
    ///
    /// assert!(all_captures.next().is_none());
    /// ```
    #[must_use]
    #[inline]
    pub fn captures_iter<'r, 't>(&'r self, text: &'t str) -> CaptureMatches<'r, 't> {
        CaptureMatches(self.find_iter(text))
    }

    /// Returns the capture groups for the first match in `text`.
    ///
    /// If no match is found, then `Ok(None)` is returned.
    ///
    /// # Examples
    ///
    /// Finding matches and capturing parts of the match:
    ///
    /// ```rust
    /// # use fancy_regex::Regex;
    /// let re = Regex::new(r"(\d{4})-(\d{2})-(\d{2})").unwrap();
    /// let text = "The date was 2018-04-07";
    /// let captures = re.captures(text).unwrap().unwrap();
    ///
    /// assert_eq!(captures.get(1).unwrap().as_str(), "2018");
    /// assert_eq!(captures.get(2).unwrap().as_str(), "04");
    /// assert_eq!(captures.get(3).unwrap().as_str(), "07");
    /// assert_eq!(captures.get(0).unwrap().as_str(), "2018-04-07");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an [`Error::RuntimeError`] for any runtime error occurred.
    #[inline]
    pub fn captures<'r, 't>(&'r self, text: &'t str) -> Result<Option<Captures<'r, 't>>> {
        self.captures_from_pos(text, 0)
    }

    /// Returns the capture groups for the first match in `text`, starting from
    /// the specified byte position `pos`.
    ///
    /// # Examples
    ///
    /// Finding captures starting at a position:
    ///
    /// ```
    /// # use fancy_regex::Regex;
    /// let re = Regex::new(r"(?m:^)(\d+)").unwrap();
    /// let text = "1 test 123\n2 foo";
    /// let captures = re.captures_from_pos(text, 7).unwrap().unwrap();
    ///
    /// let group = captures.get(1).unwrap();
    /// assert_eq!(group.as_str(), "2");
    /// assert_eq!(group.start(), 11);
    /// assert_eq!(group.end(), 12);
    /// ```
    ///
    /// Note that in some cases this is not the same as using the [`Regex::captures`]
    /// method and passing a slice of the string, see the capture that we get
    /// when we do this:
    ///
    /// ```
    /// # use fancy_regex::Regex;
    /// # let re = Regex::new(r"(?m:^)(\d+)").unwrap();
    /// # let text = "1 test 123\n2 foo";
    /// let captures = re.captures(&text[7..]).unwrap().unwrap();
    /// assert_eq!(captures.get(1).unwrap().as_str(), "123");
    /// ```
    ///
    /// This matched the number "123" because it's at the beginning of the text
    /// of the string slice.
    ///
    /// # Errors
    ///
    /// Returns an [`Error::RuntimeError`] for any runtime error occurred.
    #[inline]
    pub fn captures_from_pos<'r, 't>(
        &'r self,
        text: &'t str,
        pos: usize,
    ) -> Result<Option<Captures<'r, 't>>> {
        self.captures_within_range(text, pos..text.len())
    }

    /// Returns the capture groups for the first match in `text`
    /// within the specified `range`.
    /// The `start` and `end` of `range` are treated as byte positions in `text`.
    ///
    /// # Example
    ///
    /// Finding captures within a range:
    ///
    /// ```
    /// # use fancy_regex::Regex;
    /// let re = Regex::new(r"\<(\w+)\>").unwrap();
    /// let text = "The quick brown fox";
    ///
    /// let captures = re.captures_within_range(text, 1..11).unwrap().unwrap();
    /// let group = captures.get(1).unwrap();
    /// assert_eq!(group.as_str(), "quick");
    /// assert_eq!(group.start(), 4);
    /// assert_eq!(group.end(), 9);
    ///
    /// let captures = re.captures_within_range(text, 5..11).unwrap();
    /// assert!(captures.is_none());
    /// ```
    ///
    /// Note that in some cases this is not the same as using the [`Regex::captures`]
    /// method and passing a slice of the string, see the capture that we get
    /// when we do this:
    ///
    /// ```
    /// # use fancy_regex::Regex;
    /// # let re = Regex::new(r"\<(\w+)\>").unwrap();
    /// # let text = "The quick brown fox";
    /// let captures = re.captures(&text[5..11]).unwrap();
    /// assert!(captures.is_some());
    /// let group = captures.unwrap().get(1).unwrap();
    /// assert_eq!(group.as_str(), "uick");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an [`Error::RuntimeError`] for any runtime error occurred.
    #[inline]
    pub fn captures_within_range<'r, 't>(
        &'r self,
        text: &'t str,
        range: impl Into<Range<usize>>,
    ) -> Result<Option<Captures<'r, 't>>> {
        let range = range.into();
        let named_groups = &self.tree.named_groups;
        let mut saves = self.saves.get();
        saves.clear();
        let result = self
            .session
            .get()
            .run_to(&mut saves, text, range, Some(self.n_groups))?;
        Ok(result.then_some(Captures {
            text,
            saves,
            named_groups,
        }))
    }

    /// Returns the number of captures, including the implicit capture of the entire expression.
    #[must_use]
    #[inline]
    pub fn captures_len(&self) -> usize {
        self.n_groups
    }

    /// Returns an iterator over the capture names.
    ///
    /// This method allocate and create a new [`CaptureNames`] every time it is called.
    #[must_use]
    #[inline]
    pub fn capture_names(&self) -> CaptureNames {
        let mut names = Vec::new();
        names.resize(self.captures_len(), None);
        for (name, &i) in &self.tree.named_groups {
            names[i] = Some(name.as_str());
        }
        CaptureNames(names.into_iter())
    }

    // for debugging only
    #[cfg(debug_assertions)]
    #[doc(hidden)]
    pub fn debug_print(&self) {
        self.machine.debug_print();
    }

    /// Replaces the leftmost-first match with the replacement provided.
    /// The replacement can be a regular string (where `$N` and `$name` are
    /// expanded to match capture groups) or a function that takes the matches'
    /// `Captures` and returns the replaced string.
    ///
    /// If no match is found, then a copy of the string is returned unchanged.
    ///
    /// # Replacement string syntax
    ///
    /// All instances of `$name` in the replacement text is replaced with the
    /// corresponding capture group `name`.
    ///
    /// `name` may be an integer corresponding to the index of the
    /// capture group (counted by order of opening parenthesis where `0` is the
    /// entire match) or it can be a name (consisting of letters, digits or
    /// underscores) corresponding to a named capture group.
    ///
    /// If `name` isn't a valid capture group (whether the name doesn't exist
    /// or isn't a valid index), then it is replaced with the empty string.
    ///
    /// The longest possible name is used. e.g., `$1a` looks up the capture
    /// group named `1a` and not the capture group at index `1`. To exert more
    /// precise control over the name, use braces, e.g., `${1}a`.
    ///
    /// To write a literal `$` use `$$`.
    ///
    /// # Examples
    ///
    /// Note that this function is polymorphic with respect to the replacement.
    /// In typical usage, this can just be a normal string:
    ///
    /// ```rust
    /// # use fancy_regex::Regex;
    /// let re = Regex::new("[^01]+").unwrap();
    /// assert_eq!(re.replace("1078910", "").unwrap(), "1010");
    /// ```
    ///
    /// But anything satisfying the `Replacer` trait will work. For example,
    /// a closure of type `|&Captures| -> String` provides direct access to the
    /// captures corresponding to a match. This allows one to access
    /// capturing group matches easily:
    ///
    /// ```rust
    /// # use fancy_regex::{Regex, Captures};
    /// let re = Regex::new(r"([^,\s]+),\s+(\S+)").unwrap();
    /// let result = re.replace("Springsteen, Bruce", |caps: &Captures| {
    ///     format!("{} {}", &caps[2], &caps[1])
    /// });
    /// assert_eq!(result.unwrap(), "Bruce Springsteen");
    /// ```
    ///
    /// But this is a bit cumbersome to use all the time. Instead, a simple
    /// syntax is supported that expands `$name` into the corresponding capture
    /// group. Here's the last example, but using this expansion technique
    /// with named capture groups:
    ///
    /// ```rust
    /// # use fancy_regex::Regex;
    /// let re = Regex::new(r"(?P<last>[^,\s]+),\s+(?P<first>\S+)").unwrap();
    /// let result = re.replace("Springsteen, Bruce", "$first $last");
    /// assert_eq!(result.unwrap(), "Bruce Springsteen");
    /// ```
    ///
    /// Note that using `$2` instead of `$first` or `$1` instead of `$last`
    /// would produce the same result. To write a literal `$` use `$$`.
    ///
    /// Sometimes the replacement string requires use of curly braces to
    /// delineate a capture group replacement and surrounding literal text.
    /// For example, if we wanted to join two words together with an
    /// underscore:
    ///
    /// ```rust
    /// # use fancy_regex::Regex;
    /// let re = Regex::new(r"(?P<first>\w+)\s+(?P<second>\w+)").unwrap();
    /// let result = re.replace("deep fried", "${first}_$second");
    /// assert_eq!(result.unwrap(), "deep_fried");
    /// ```
    ///
    /// Without the curly braces, the capture group name `first_` would be
    /// used, and since it doesn't exist, it would be replaced with the empty
    /// string.
    ///
    /// Finally, sometimes you just want to replace a literal string with no
    /// regard for capturing group expansion. This can be done by wrapping a
    /// byte string with `NoExpand`:
    ///
    /// ```rust
    /// # use fancy_regex::Regex;
    /// use fancy_regex::NoExpand;
    ///
    /// let re = Regex::new(r"(?P<last>[^,\s]+),\s+(\S+)").unwrap();
    /// let result = re.replace("Springsteen, Bruce", NoExpand("$2 $last"));
    /// assert_eq!(result.unwrap(), "$2 $last");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an [`Error::RuntimeError`] for any runtime error occurred.
    #[allow(clippy::missing_panics_doc)] // this cannot panic
    #[inline]
    pub fn replace<'t, R: Replacer>(&self, text: &'t str, rep: R) -> Result<Cow<'t, str>> {
        self.replacen(text, Some(NonZeroUsize::new(1).unwrap()), rep)
    }

    /// Replaces all non-overlapping matches in `text` with the replacement
    /// provided. This is the same as calling `replacen` with `limit` set to
    /// `0`.
    ///
    /// See the documentation for `replace` for details on how to access
    /// capturing group matches in the replacement string.
    ///
    /// # Errors
    ///
    /// Returns an [`Error::RuntimeError`] for any runtime error occurred.
    #[inline]
    pub fn replace_all<'t, R: Replacer>(&self, text: &'t str, rep: R) -> Result<Cow<'t, str>> {
        self.replacen(text, None, rep)
    }

    /// Replaces at most `limit` non-overlapping matches in `text` with the
    /// replacement provided. If `limit` is `None`, then all non-overlapping matches
    /// are replaced.
    ///
    /// See the documentation for `replace` for details on how to access
    /// capturing group matches in the replacement string.
    ///
    /// # Errors
    ///
    /// Returns an [`Error::RuntimeError`] for any runtime error occurred.
    #[allow(clippy::missing_panics_doc)] // this cannot panic
    pub fn replacen<'t, R: Replacer>(
        &self,
        text: &'t str,
        limit: Option<NonZeroUsize>,
        mut rep: R,
    ) -> Result<Cow<'t, str>> {
        // If we know that the replacement doesn't have any capture expansions,
        // then we can fast path. The fast path can make a tremendous
        // difference:
        //
        //   1) We use `find_iter` instead of `captures_iter`. Not asking for
        //      captures generally makes the regex engines faster.
        //   2) We don't need to look up all of the capture groups and do
        //      replacements inside the replacement string. We just push it
        //      at each match and be done with it.
        if let Some(rep) = rep.no_expansion() {
            let mut it = self.find_iter(text).enumerate().peekable();
            if it.peek().is_none() {
                return Ok(Cow::Borrowed(text));
            }
            let mut new = String::with_capacity(text.len());
            let mut last_match = 0;
            for (i, m) in it {
                let m = m?;
                if limit.map_or(false, |limit| i >= limit.into()) {
                    break;
                }
                new.push_str(&text[last_match..m.start()]);
                new.push_str(&rep);
                last_match = m.end();
            }
            new.push_str(&text[last_match..]);
            return Ok(Cow::Owned(new));
        }

        // The slower path, which we use if the replacement needs access to
        // capture groups.
        let mut it = self.captures_iter(text).enumerate().peekable();
        if it.peek().is_none() {
            return Ok(Cow::Borrowed(text));
        }
        let mut new = String::with_capacity(text.len());
        let mut last_match = 0;
        for (i, cap) in it {
            let cap = cap?;
            if limit.map_or(false, |limit| i >= limit.into()) {
                break;
            }
            // unwrap on 0 is OK because captures only reports matches
            let m = cap.get(0).unwrap();
            new.push_str(&text[last_match..m.start()]);
            rep.replace_append(&cap, &mut new);
            last_match = m.end();
        }
        new.push_str(&text[last_match..]);
        Ok(Cow::Owned(new))
    }
}

fn new_saves_pool() -> Pool<Vec<usize>, fn() -> Vec<usize>> {
    Pool::new(Vec::default)
}

fn new_session_pool(
    machine: Machine,
) -> Pool<Session, Box<dyn Fn() -> Session + Send + Sync + UnwindSafe + RefUnwindSafe>> {
    Pool::new(Box::new(move || {
        let state = Machine::create_state(&machine.prog);

        machine.clone().create_session(state)
    }))
}

impl Clone for Regex {
    fn clone(&self) -> Self {
        let machine = self.machine.clone();
        Regex {
            pattern: self.pattern.clone(),
            tree: self.tree.clone(),
            machine: machine.clone(),
            session: new_session_pool(machine),
            saves: new_saves_pool(),
            n_groups: self.n_groups,
        }
    }
}

impl FromStr for Regex {
    type Err = Error;

    /// Attempts to parse a string into a regular expression
    fn from_str(s: &str) -> Result<Regex> {
        Regex::new(s)
    }
}

#[derive(Clone, Debug)]
enum RegexSource {
    Pattern(String),
    ExprTree(ExprTree),
}

#[derive(Copy, Clone, Debug)]
struct RegexOptions {
    anchored: bool,
    max_stack: usize,
    backtrack_limit: usize,
    delegate_size_limit: Option<usize>,
    delegate_dfa_size_limit: Option<usize>,
}

impl Default for RegexOptions {
    fn default() -> Self {
        RegexOptions {
            anchored: false,
            max_stack: DEFAULT_MAX_STACK,
            backtrack_limit: DEFAULT_BACKTRACK_LIMIT,
            delegate_size_limit: None,
            delegate_dfa_size_limit: None,
        }
    }
}

/// A builder for a `Regex` to allow configuring options.
#[derive(Debug, Copy, Clone, Default)]
pub struct RegexBuilder(RegexOptions);

impl RegexBuilder {
    /// Create a new regex builder with default options.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build the [`Regex`].
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if the pattern could not be parsed or compiled.
    #[inline]
    pub fn build(&self, pattern: impl Into<String>) -> Result<Regex> {
        Regex::new_with_source_and_options(RegexSource::Pattern(pattern.into()), self.0)
    }

    /// Build the [`Regex`] from an [`ExprTree`].
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if the pattern could not be compiled.
    #[inline]
    pub fn build_from_expr_tree(&self, expr_tree: impl Into<ExprTree>) -> Result<Regex> {
        Regex::new_with_source_and_options(RegexSource::ExprTree(expr_tree.into()), self.0)
    }

    /// Limit for how many times backtracking should be attempted for fancy regexes (where
    /// backtracking is used). If this limit is exceeded, execution returns an
    /// [`Error::RuntimeError`] with [`RuntimeError::BacktrackLimitExceeded`].
    /// This is for preventing a regex with catastrophic backtracking to run for too long.
    ///
    /// Default is `1_000_000` (1 million).
    #[inline]
    pub fn backtrack_limit(&mut self, limit: usize) -> &mut Self {
        self.0.backtrack_limit = limit;
        self
    }

    /// Limit the stack height of the virtual machine when running for fancy regexes (where
    /// backtracking is used). If this limit is exceeded, execution returns an
    /// [`Error::RuntimeError`] with [`RuntimeError::StackOverflow`].
    /// This is for preventing a regex with catastrophic backtracking to comsume too many memory.
    ///
    /// Default is `1_000_000` (1 million).
    #[inline]
    pub fn max_stack(&mut self, limit: usize) -> &mut Self {
        self.0.max_stack = limit;
        self
    }

    /// Set the approximate size limit of the compiled regular expression.
    ///
    /// This option is forwarded from the wrapped `regex` crate. Note that depending on the used
    /// regex features there may be multiple delegated sub-regexes fed to the `regex` crate. As
    /// such the actual limit is closer to `<number of delegated regexes> * delegate_size_limit`.
    #[inline]
    pub fn delegate_size_limit(&mut self, limit: usize) -> &mut Self {
        self.0.delegate_size_limit = Some(limit);
        self
    }

    /// Set the approximate size of the cache used by the DFA.
    ///
    /// This option is forwarded from the wrapped `regex` crate. Note that depending on the used
    /// regex features there may be multiple delegated sub-regexes fed to the `regex` crate. As
    /// such the actual limit is closer to `<number of delegated regexes> *
    /// delegate_dfa_size_limit`.
    #[inline]
    pub fn delegate_dfa_size_limit(&mut self, limit: usize) -> &mut Self {
        self.0.delegate_dfa_size_limit = Some(limit);
        self
    }
}

/// A single match of a regex or group in an input text
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Match<'t> {
    text: &'t str,
    start: usize,
    end: usize,
}

impl<'t> Match<'t> {
    /// Returns the starting byte offset of the match in the text.
    #[must_use]
    #[inline]
    pub fn start(&self) -> usize {
        self.start
    }

    /// Returns the ending byte offset of the match in the text.
    #[must_use]
    #[inline]
    pub fn end(&self) -> usize {
        self.end
    }

    /// Returns the range over the starting and ending byte offsets of the match in text.
    #[must_use]
    #[inline]
    pub fn range(&self) -> Range<usize> {
        self.start..self.end
    }

    /// Returns the matched text.
    #[must_use]
    #[inline]
    pub fn as_str(&self) -> &'t str {
        &self.text[self.start..self.end]
    }

    /// Returns the length, in bytes, of this match.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.range().len()
    }

    /// Returns true if and only if this match has a length of zero.
    ///
    /// Note that an empty match can only occur when the regex itself can match the empty string.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.range().is_empty()
    }
}

impl<'t> AsRef<str> for Match<'t> {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl<'t> From<Match<'t>> for Range<usize> {
    fn from(m: Match<'t>) -> Range<usize> {
        m.range()
    }
}

/// An iterator over all non-overlapping matches for a particular string.
///
/// The iterator yields a `Result<Match>`. The iterator stops when no more
/// matches can be found.
///
/// `'r` is the lifetime of the compiled regular expression and `'t` is the
/// lifetime of the matched string.
///
/// Note that this iterator does not implement [`FusedIterator`].
#[derive(Debug, Clone)]
pub struct Matches<'r, 't> {
    re: &'r Regex,
    text: &'t str,
    last_end: usize,
    last_match: Option<usize>,
}

impl<'r, 't> Matches<'r, 't> {
    /// Return the text being searched.
    #[must_use]
    #[inline]
    pub fn text(&self) -> &'t str {
        self.text
    }

    /// Return the underlying regex.
    #[must_use]
    #[inline]
    pub fn regex(&self) -> &'r Regex {
        self.re
    }
}

impl<'r, 't> Iterator for Matches<'r, 't> {
    type Item = Result<Match<'t>>;

    /// Adapted from the `regex` crate. Calls `find_from_pos` repeatedly.
    /// Ignores empty matches immediately after a match.
    fn next(&mut self) -> Option<Self::Item> {
        if self.last_end > self.text.len() {
            return None;
        }

        let option_flags = match self.last_match {
            Some(last_match) if self.last_end > last_match => OPTION_SKIPPED_EMPTY_MATCH,
            _ => 0,
        };

        let mat = match self.re.find_within_range_with_option_flags(
            self.text,
            self.last_end..self.text.len(),
            option_flags,
        ) {
            Err(error) => return Some(Err(error)),
            Ok(None) => return None,
            Ok(Some(mat)) => mat,
        };

        if mat.start == mat.end {
            // This is an empty match. To ensure we make progress, start
            // the next search at the smallest possible starting position
            // of the next match following this one.
            self.last_end = next_codepoint_ix(self.text, mat.end);
            // Don't accept empty matches immediately following a match.
            // Just move on to the next match.
            if Some(mat.end) == self.last_match {
                return self.next();
            }
        } else {
            self.last_end = mat.end;
        }

        self.last_match = Some(mat.end);

        Some(Ok(mat))
    }
}

/// An iterator that yields all non-overlapping capture groups matching a
/// particular regular expression.
///
/// The iterator yields a `Result<Captures>`. The iterator stops when no
/// more matches can be found.
///
/// `'r` is the lifetime of the compiled regular expression and `'t` is the
/// lifetime of the matched string.
///
/// Note that this iterator does not implement [`FusedIterator`].
#[derive(Debug, Clone)]
pub struct CaptureMatches<'r, 't>(Matches<'r, 't>);

impl<'r, 't> CaptureMatches<'r, 't> {
    /// Return the text being searched.
    #[must_use]
    #[inline]
    pub fn text(&self) -> &'t str {
        self.0.text
    }

    /// Return the underlying regex.
    #[must_use]
    #[inline]
    pub fn regex(&self) -> &'r Regex {
        self.0.re
    }
}

impl<'r, 't> Iterator for CaptureMatches<'r, 't> {
    type Item = Result<Captures<'r, 't>>;

    /// Adapted from the `regex` crate. Calls `captures_from_pos` repeatedly.
    /// Ignores empty matches immediately after a match.
    fn next(&mut self) -> Option<Self::Item> {
        if self.0.last_end > self.0.text.len() {
            return None;
        }

        let captures = match self.0.re.captures_from_pos(self.0.text, self.0.last_end) {
            Err(error) => return Some(Err(error)),
            Ok(None) => return None,
            Ok(Some(captures)) => captures,
        };

        let mat = captures
            .get(0)
            .expect("`Captures` is expected to have entire match at 0th position");
        if mat.start == mat.end {
            self.0.last_end = next_codepoint_ix(self.0.text, mat.end);
            if Some(mat.end) == self.0.last_match {
                return self.next();
            }
        } else {
            self.0.last_end = mat.end;
        }

        self.0.last_match = Some(mat.end);

        Some(Ok(captures))
    }
}

impl<'r, 't> From<Matches<'r, 't>> for CaptureMatches<'r, 't> {
    fn from(value: Matches<'r, 't>) -> Self {
        CaptureMatches(value)
    }
}

impl<'r, 't> From<CaptureMatches<'r, 't>> for Matches<'r, 't> {
    fn from(value: CaptureMatches<'r, 't>) -> Self {
        value.0
    }
}

/// A set of capture groups found for a regex.
#[derive(Debug)]
pub struct Captures<'r, 't> {
    text: &'t str,
    saves: PoolGuard<'r, Vec<usize>, fn() -> Vec<usize>>,
    named_groups: &'r NamedGroups,
}

#[allow(clippy::len_without_is_empty)] // follow regex's API
impl<'r, 't> Captures<'r, 't> {
    /// Get the capture group by its index in the regex.
    ///
    /// If there is no match for that group or the index does not correspond to a group, `None` is
    /// returned. The index 0 returns the whole match.
    #[must_use]
    #[inline]
    pub fn get(&self, i: usize) -> Option<Match<'t>> {
        let Captures {
            text, ref saves, ..
        } = self;
        let slot = i.saturating_mul(2);
        if slot >= saves.len() {
            return None;
        }
        let lo = saves[slot];
        if lo == std::usize::MAX {
            return None;
        }
        let hi = saves[slot + 1];
        Some(Match {
            text,
            start: lo,
            end: hi,
        })
    }

    /// Returns the match for a named capture group.  Returns `None` the capture
    /// group did not match or if there is no group with the given name.
    #[must_use]
    #[inline]
    pub fn name(&self, name: &str) -> Option<Match<'t>> {
        self.named_groups.get(name).and_then(|i| self.get(*i))
    }

    /// Expands all instances of `$group` in `replacement` to the corresponding
    /// capture group `name`, and writes them to the `dst` buffer given.
    ///
    /// `group` may be an integer corresponding to the index of the
    /// capture group (counted by order of opening parenthesis where `\0` is the
    /// entire match) or it can be a name (consisting of letters, digits or
    /// underscores) corresponding to a named capture group.
    ///
    /// If `group` isn't a valid capture group (whether the name doesn't exist
    /// or isn't a valid index), then it is replaced with the empty string.
    ///
    /// The longest possible name is used. e.g., `$1a` looks up the capture
    /// group named `1a` and not the capture group at index `1`. To exert more
    /// precise control over the name, use braces, e.g., `${1}a`.
    ///
    /// To write a literal `$`, use `$$`.
    ///
    /// For more control over expansion, see [`Expander`].
    #[inline]
    pub fn expand(&self, replacement: &str, dst: &mut String) {
        Expander::default().append_expansion(dst, replacement, self);
    }

    /// Iterate over the captured groups in order in which they appeared in the regex. The first
    /// capture corresponds to the whole match.
    #[must_use]
    #[inline]
    pub fn iter<'c>(&'c self) -> SubCaptureMatches<'c, 't> {
        SubCaptureMatches(self.saves.chunks_exact(2), self.text)
    }

    /// How many groups were captured. This is always at least 1 because group 0 returns the whole
    /// match.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.saves.len() / 2
    }
}

impl<'r, 't, 'c> IntoIterator for &'c Captures<'r, 't> {
    type IntoIter = SubCaptureMatches<'c, 't>;
    type Item = std::option::Option<Match<'t>>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Get a group by index.
///
/// `'t` is the lifetime of the matched text.
///
/// The text can't outlive the `Captures` object if this method is
/// used, because of how `Index` is defined (normally `a[i]` is part
/// of `a` and can't outlive it); to do that, use `get()` instead.
///
/// # Panics
///
/// If there is no group at the given index.
impl<'r, 't> Index<usize> for Captures<'r, 't> {
    type Output = str;

    fn index(&self, i: usize) -> &str {
        self.get(i)
            .map_or_else(|| panic!("no group at index '{i}'"), |m| m.as_str())
    }
}

/// Get a group by name.
///
/// `'t` is the lifetime of the matched text and `'i` is the lifetime
/// of the group name (the index).
///
/// The text can't outlive the `Captures` object if this method is
/// used, because of how `Index` is defined (normally `a[i]` is part
/// of `a` and can't outlive it); to do that, use `name` instead.
///
/// # Panics
///
/// If there is no group named by the given value.
impl<'r, 't, 'i> Index<&'i str> for Captures<'r, 't> {
    type Output = str;

    fn index<'a>(&'a self, name: &'i str) -> &'a str {
        self.name(name)
            .map_or_else(|| panic!("no group named '{name}'"), |m| m.as_str())
    }
}

/// Iterator for captured groups in order in which they appear in the regex.
#[derive(Debug)]
pub struct SubCaptureMatches<'r, 't>(ChunksExact<'r, usize>, &'t str);

impl<'c, 't> SubCaptureMatches<'c, 't> {
    fn get(&self, span: [usize; 2]) -> Option<Match<'t>> {
        if span[0] == usize::MAX {
            None
        } else {
            Some(Match {
                text: self.1,
                start: span[0],
                end: span[1],
            })
        }
    }
}

impl<'c, 't> Iterator for SubCaptureMatches<'c, 't> {
    type Item = Option<Match<'t>>;

    fn next(&mut self) -> Option<Option<Match<'t>>> {
        let span = self.0.next()?.try_into().unwrap();
        Some(self.get(span))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    fn count(self) -> usize {
        self.0.count()
    }
}

impl<'c, 't> DoubleEndedIterator for SubCaptureMatches<'c, 't> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let span = self.0.next_back()?.try_into().unwrap();
        Some(self.get(span))
    }
}

impl<'c, 't> ExactSizeIterator for SubCaptureMatches<'c, 't> {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'c, 't> FusedIterator for SubCaptureMatches<'c, 't> {}

/// An iterator over capture names in a [`Regex`].  The iterator
/// returns the name of each group, or `None` if the group has
/// no name.  Because capture group 0 cannot have a name, the
/// first item returned is always `None`.
#[derive(Debug, Clone)]
pub struct CaptureNames<'r>(std::vec::IntoIter<Option<&'r str>>);

impl<'r> Iterator for CaptureNames<'r> {
    type Item = Option<&'r str>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    fn count(self) -> usize {
        self.0.count()
    }
}

impl<'r> DoubleEndedIterator for CaptureNames<'r> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}

impl<'r> ExactSizeIterator for CaptureNames<'r> {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'r> FusedIterator for CaptureNames<'r> {}

#[inline]
fn codepoint_len(b: u8) -> usize {
    match b {
        b if b < 0x80 => 1,
        b if b < 0xe0 => 2,
        b if b < 0xf0 => 3,
        _ => 4,
    }
}

// precondition: ix > 0
#[allow(clippy::cast_possible_wrap)]
#[inline]
fn prev_codepoint_ix(s: impl AsRef<[u8]>, mut ix: usize) -> usize {
    let bytes = s.as_ref();
    loop {
        ix -= 1;
        // fancy bit magic for ranges 0..0x80 + 0xc0..
        if (bytes[ix] as i8) >= -0x40 {
            break;
        }
    }
    ix
}

#[inline]
fn next_codepoint_ix(s: impl AsRef<[u8]>, ix: usize) -> usize {
    ix + codepoint_len(s.as_ref()[ix])
}

// If this returns false, then there is no possible backref in the re

// Both potential implementations are turned off, because we currently
// always need to do a deeper analysis because of 1-character
// look-behind. If we could call a find_from_pos method of regex::Regex,
// it would make sense to bring this back.
/*
pub fn detect_possible_backref(re: &str) -> bool {
    let mut last = b'\x00';
    for b in re.as_bytes() {
        if b'0' <= *b && *b <= b'9' && last == b'\\' { return true; }
        last = *b;
    }
    false
}

pub fn detect_possible_backref(re: &str) -> bool {
    let mut bytes = re.as_bytes();
    loop {
        match memchr::memchr(b'\\', &bytes[..bytes.len() - 1]) {
            Some(i) => {
                bytes = &bytes[i + 1..];
                let c = bytes[0];
                if b'0' <= c && c <= b'9' { return true; }
            }
            None => return false
        }
    }
}
*/

/// The internal module only exists so that the toy example can access internals for debugging and
/// experimenting.
#[doc(hidden)]
pub mod internal {
    pub use crate::analyze::analyze;
    pub use crate::compile::compile;
    pub use crate::vm::{run_default_from_pos, run_trace_from_pos, Insn, Prog};
}

#[cfg(test)]
mod tests {
    use crate::parse::make_literal;
    use crate::Expr;
    use crate::Regex;
    use crate::RegexBuilder;
    //use detect_possible_backref;

    // tests for to_str

    fn to_str(e: Expr) -> String {
        e.to_str().unwrap()
    }

    #[test]
    fn to_str_concat_alt() {
        let e = Expr::Concat(vec![
            Expr::Alt(vec![make_literal("a"), make_literal("b")]),
            make_literal("c"),
        ]);
        assert_eq!(to_str(e), "(?:[ab]c)");
    }

    #[test]
    fn to_str_rep_concat() {
        let e = Expr::Repeat {
            child: Box::new(Expr::Concat(vec![make_literal("a"), make_literal("b")])),
            lo: 2,
            hi: 3,
            greedy: true,
        };
        assert_eq!(to_str(e), "(?:ab){2,3}");
    }

    #[test]
    fn to_str_group_alt() {
        let e = Expr::Group(Box::new(Expr::Alt(vec![
            make_literal("a"),
            make_literal("b"),
        ])));
        assert_eq!(to_str(e), "([ab])");
    }

    #[test]
    fn from_str() {
        let s = r"(a+)b\1";
        let regex = s.parse::<Regex>().unwrap();
        assert_eq!(regex.as_str(), s);
    }

    #[test]
    fn to_str_repeat() {
        fn repeat(lo: usize, hi: usize, greedy: bool) -> Expr {
            Expr::Repeat {
                child: Box::new(make_literal("a")),
                lo,
                hi,
                greedy,
            }
        }

        assert_eq!(to_str(repeat(2, 2, true)), "a{2}");
        assert_eq!(to_str(repeat(2, 2, false)), "a{2}");
        assert_eq!(to_str(repeat(2, 3, true)), "a{2,3}");
        assert_eq!(to_str(repeat(2, 3, false)), "a{2,3}?");
        assert_eq!(to_str(repeat(2, usize::MAX, true)), "a{2,}");
        assert_eq!(to_str(repeat(2, usize::MAX, false)), "a{2,}?");
        assert_eq!(to_str(repeat(0, 1, true)), "a?");
        assert_eq!(to_str(repeat(0, 1, false)), "a??");
        assert_eq!(to_str(repeat(0, usize::MAX, true)), "a*");
        assert_eq!(to_str(repeat(0, usize::MAX, false)), "a*?");
        assert_eq!(to_str(repeat(1, usize::MAX, true)), "a+");
        assert_eq!(to_str(repeat(1, usize::MAX, false)), "a+?");
    }

    #[test]
    fn to_str_multiline() {
        let tree = Expr::parse_tree("(?m)^yes$").unwrap();
        let expr = tree.expr;
        assert_eq!(to_str(expr), "(?:(?m:^)(?:yes)(?m:$))");
    }

    #[test]
    fn expr_roundtrip() {
        let tree = Expr::parse_tree("(?m)^yes$").unwrap();
        let regex = RegexBuilder::new()
            .build_from_expr_tree(tree.clone())
            .unwrap();
        assert_eq!(regex.as_expr_tree(), &tree);
    }

    /*
    #[test]
    fn detect_backref() {
        assert_eq!(detect_possible_backref("a0a1a2"), false);
        assert_eq!(detect_possible_backref("a0a1\\a2"), false);
        assert_eq!(detect_possible_backref("a0a\\1a2"), true);
        assert_eq!(detect_possible_backref("a0a1a2\\"), false);
    }
    */
}
