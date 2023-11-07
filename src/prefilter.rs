use std::ops::Range;

#[allow(missing_docs)] // I havn't write it yet
use aho_corasick::{AhoCorasick, AhoCorasickBuilder, AhoCorasickKind, MatchKind, StartKind};

use crate::{analyze::Info, Expr, LookAround};

#[derive(Debug, Clone)]
struct Prefixer {
    aho: AhoCorasick,
    offsets: Vec<isize>,
    safe_offset: isize,
}

#[derive(Debug, Clone)]
struct Asserter {
    aho: AhoCorasick,
    count: usize,
}

#[derive(Debug, Clone)]
pub struct Prefilter {
    prefixer: Option<Prefixer>,
    asserter: Option<Asserter>,
    longest_positive_lookahead: usize,
    longest_positive_lookbehind: usize,
}

impl Prefilter {
    pub fn new(info: &Info<'_>) -> Option<Self> {
        if info.has_continue {
            // XXX: we cannot handle it
            return None;
        }

        let mut seq = Vec::new();
        extract_literals(info, &mut seq, false);

        // it will be insane to build a DFA with too many patterns
        const MAX_FRAG: usize = std::mem::size_of::<usize>() * 8;
        let mut builder = AhoCorasickBuilder::new();
        builder.kind(Some(AhoCorasickKind::DFA));
        builder.start_kind(StartKind::Unanchored);
        builder.match_kind(MatchKind::Standard);
        builder.ascii_case_insensitive(seq.iter().any(|frag| frag.casei));

        if seq.len() < MAX_FRAG {
            let mut prefixes = seq.clone();
            // put fragments with large offset at the back
            prefixes.sort_by_key(|frag| {
                (
                    (frag.offset == isize::MIN)
                        .then_some(isize::MAX)
                        .unwrap_or(frag.offset),
                    frag.val,
                )
            });
            // eliminate unbounded
            while prefixes.last().map_or(false, |frag| {
                frag.offset == isize::MAX || frag.offset == isize::MIN
            }) {
                prefixes.pop().unwrap();
            }
            // dedup prefixes
            prefixes.dedup_by_key(|frag| (frag.offset, frag.val));

            // XXX: what if we are not speaking English?
            let too_cheap = prefixes
                .iter()
                .filter(|frag| {
                    frag.size == 1 && {
                        let b = frag.val.as_bytes()[0];
                        b.is_ascii_alphanumeric() || b.is_ascii_whitespace()
                    }
                })
                .count()
                > 8;

            let prefixer = (!prefixes.is_empty() && !too_cheap)
                .then(|| {
                    let prefixer = builder.build(
                        prefixes.iter().map(|frag| frag.val.as_bytes()).rev(), // large offset comes first
                    );

                    let prefixer = if cfg!(debug_assertions) {
                        prefixer.expect("failed to build AhoCorasick")
                    } else {
                        prefixer.ok()?
                    };

                    let offsets: Vec<_> = prefixes.iter().map(|frag| frag.offset).rev().collect(); // remember to rev() at both places

                    let safe_offset = offsets
                        .iter()
                        .copied()
                        .max()
                        .unwrap()
                        .saturating_add_unsigned(
                            prefixes
                                .into_iter()
                                .map(|frag| frag.size - 1)
                                .max()
                                .unwrap(),
                        );

                    Some(Prefixer {
                        aho: prefixer,
                        offsets,
                        safe_offset,
                    })
                })
                .flatten();

            let mut asserts = seq;
            // dedup asserts
            asserts.dedup_by_key(|frag| frag.val);

            let asserter = (!asserts.is_empty())
                .then(|| {
                    let count = asserts.iter().filter(|frag| frag.must).count();

                    let asserter = builder.build(
                        asserts
                            .iter()
                            .filter(|frag| count == 0 || frag.must)
                            .map(|frag| frag.val.as_bytes()),
                    );

                    let asserter = if cfg!(debug_assertions) {
                        asserter.expect("failed to build AhoCorasick")
                    } else {
                        asserter.ok()?
                    };

                    Some(Asserter {
                        aho: asserter,
                        count,
                    })
                })
                .flatten();

            Some(Prefilter {
                prefixer,
                asserter,
                longest_positive_lookahead: info.longest_positive_lookahead,
                longest_positive_lookbehind: info.longest_positive_lookbehind,
            })
        } else {
            None
        }
    }

    pub fn search<'a>(
        &'a self,
        haystack: &'a [u8],
        range: &Range<usize>,
    ) -> Option<impl Iterator<Item = Match> + 'a> {
        self.prefixer.as_ref().map(|prefixer| {
            let original_range = range.clone();
            let mut range = self.enlarge_range(haystack, range);
            range.end = range.end.min(
                original_range
                    .end
                    .saturating_add(1usize.saturating_add_signed(prefixer.safe_offset)),
            );
            // Why there is no impl<T: Copy> Copy for Range<T>?
            let haystack = &haystack[range.clone()];
            prefixer
                .aho
                .find_overlapping_iter(haystack)
                .map(move |m| Match {
                    position: m.start() + range.start,
                    offset: prefixer.offsets[m.pattern().as_usize()],
                })
                .filter(move |m| original_range.contains(&m.position))
        })
    }

    pub fn assert(&self, haystack: &[u8], range: &Range<usize>) -> bool {
        let Some(asserter) = self.asserter.as_ref() else {
            return true;
        };
        let range = self.enlarge_range(haystack, range);
        let haystack = &haystack[range];
        if asserter.count == 0 {
            asserter.aho.is_match(haystack)
        } else {
            let mut asserted: usize = 0;
            for m in asserter.aho.find_overlapping_iter(haystack) {
                let pattern = m.pattern().as_usize();
                asserted |= 1 << pattern;
                if asserted == (1 << asserter.count) - 1 {
                    return true;
                }
            }
            false
        }
    }

    fn enlarge_range(&self, haystack: &[u8], range: &Range<usize>) -> Range<usize> {
        range.start.saturating_sub(self.longest_positive_lookbehind)
            ..range
                .end
                .saturating_add(self.longest_positive_lookahead)
                .min(haystack.len())
    }

    pub fn safe_offset(&self) -> Option<isize> {
        self.prefixer.as_ref().map(|prefixer| prefixer.safe_offset)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Match {
    /// Position in haystack, in bytes
    pub position: usize,
    /// Offset in perfix, in chars
    pub offset: isize,
}

#[derive(Debug, Copy, Clone)]
struct Fragment<'a> {
    val: &'a str,
    size: usize,
    offset: isize,
    casei: bool,
    must: bool,
}

fn extract_literals<'a>(info: &Info<'a>, seq: &mut Vec<Fragment<'a>>, all: bool) {
    if !all && !info.must_exist {
        return;
    }
    match info.expr {
        Expr::Literal { val, casei } => {
            if (all || info.must_exist) && info.literal_segment {
                debug_assert_ne!(info.min_size, 0);
                seq.push(Fragment {
                    val: &val,
                    size: info.min_size, // it's size in chars
                    offset: info.offset,
                    casei: *casei,
                    must: info.must_exist,
                });
            }
        }
        Expr::Concat(_) | Expr::Alt(_) => {
            // we do not merge literals here
            // it is done by Parser::optimize
            for child_info in &info.children {
                extract_literals(
                    child_info,
                    seq,
                    all || info.must_exist && info.literal_segment,
                );
            }
        }
        Expr::Group(_) | Expr::AtomicGroup(_) | Expr::Repeat { .. } => {
            debug_assert_eq!(info.children.len(), 1);
            extract_literals(&info.children[0], seq, all);
        }
        Expr::LookAround(_, look)
            if matches!(look, LookAround::LookAhead | LookAround::LookBehind) =>
        {
            debug_assert_eq!(info.children.len(), 1);
            extract_literals(
                &info.children[0],
                seq,
                all || info.must_exist && info.literal_segment,
            );
        }
        Expr::Conditional { .. } => {
            debug_assert_eq!(info.children.len(), 3);
            extract_literals(
                &info.children[1],
                seq,
                all || info.must_exist && info.literal_segment,
            );
            extract_literals(
                &info.children[2],
                seq,
                all || info.must_exist && info.literal_segment,
            );
        }
        Expr::Empty
        | Expr::Assertion(_)
        | Expr::Any { .. }
        | Expr::Delegate { .. }
        | Expr::Backref(_)
        | Expr::KeepOut
        | Expr::BackrefExistsCondition(_)
        | Expr::LookAround(_, _) => {
            // do nothing
        }
        Expr::ContinueFromPreviousMatchEnd => unreachable!(),
    }
}

#[cfg(test)]
mod test {
    use crate::analyze;
    use crate::Expr;

    use super::extract_literals;
    use super::Match;
    use super::Prefilter;

    const SAMPLE: &str = r"(?<=The )(fast|slow) fox (jumps|runs) over the (lazy|\w+) dog(?=\.)";

    #[test]
    fn test_extract_literals() {
        let tree = Expr::parse_tree(SAMPLE).unwrap();
        let info = analyze(&tree).unwrap();
        let mut seq = Vec::new();
        extract_literals(&info, &mut seq, false);

        assert_eq!(
            format!("{:?}", seq),
            format!(
                r#"[Fragment {{ val: "The ", size: 4, offset: -4, casei: false, must: true }}, Fragment {{ val: "fast", size: 4, offset: 0, casei: false, must: false }}, Fragment {{ val: "slow", size: 4, offset: 0, casei: false, must: false }}, Fragment {{ val: " fox ", size: 5, offset: 4, casei: false, must: true }}, Fragment {{ val: "jumps", size: 5, offset: 9, casei: false, must: false }}, Fragment {{ val: "runs", size: 4, offset: 9, casei: false, must: false }}, Fragment {{ val: " over the ", size: 10, offset: {max}, casei: false, must: true }}, Fragment {{ val: " dog", size: 4, offset: {max}, casei: false, must: true }}, Fragment {{ val: ".", size: 1, offset: {max}, casei: false, must: true }}]"#,
                max = isize::MAX
            )
        );
    }

    #[test]
    fn test_asserter() {
        let tree = Expr::parse_tree(SAMPLE).unwrap();
        let info = analyze(&tree).unwrap();
        let prefilter = Prefilter::new(&info).unwrap();

        assert!(!prefilter.assert("The".as_bytes(), &(0..3)));
        assert!(prefilter.assert("The fast fox  over the  dog.".as_bytes(), &(4..27)));
        assert!(!prefilter.assert("The fast fox  over the  dog.".as_bytes(), &(4..26)));
        assert!(!prefilter.assert("The fast fox  over the  dog.".as_bytes(), &(5..27)));
        assert!(!prefilter.assert("fast fox over the dog".as_bytes(), &(0..21)));
    }

    #[test]
    fn test_searcher() {
        let tree = Expr::parse_tree(SAMPLE).unwrap();
        let info = analyze(&tree).unwrap();
        let prefilter = Prefilter::new(&info).unwrap();

        assert_eq!(format!("{:?}", prefilter.search("The fast fox  over the  dog.".as_bytes(), &(0..28)).unwrap().collect::<Vec<Match>>()), "[Match { position: 0, offset: -4 }, Match { position: 4, offset: 0 }, Match { position: 8, offset: 4 }]");
    }

    #[test]
    fn eliminate_stack_overflow() {
        let tree = Expr::parse_tree("(?i)(a|b|ab)*(?=c)").unwrap();
        let info = analyze(&tree).unwrap();
        let prefilter = Prefilter::new(&info).unwrap();
        let haystack = "abababababababababababababababababababababababababababab";
        assert!(!prefilter.assert(haystack.as_bytes(), &(0..haystack.len())));
    }
}
