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

//! Backtracking VM for implementing fancy regexes.
//!
//! Read <https://swtch.com/~rsc/regexp/regexp2.html> for a good introduction for how this works.
//!
//! The VM executes a sequence of instructions (a program) against an input string. It keeps track
//! of a program counter (PC) and an index into the string (IX). Execution can have one or more
//! threads.
//!
//! One of the basic instructions is `Lit`, which matches a string against the input. If it matches,
//! the PC advances to the next instruction and the IX to the position after the matched string.
//! If not, the current thread is stopped because it failed.
//!
//! If execution reaches an `End` instruction, the program is successful because a match was found.
//! If there are no more threads to execute, the program has failed to match.
//!
//! A very simple program for the regex `a`:
//!
//! ```text
//! 0: Lit("a")
//! 1: End
//! ```
//!
//! The `Split` instruction causes execution to split into two threads. The first thread is executed
//! with the current string index. If it fails, we reset the string index and resume execution with
//! the second thread. That is what "backtracking" refers to. In order to do that, we keep a stack
//! of threads (PC and IX) to try.
//!
//! Example program for the regex `ab|ac`:
//!
//! ```text
//! 0: Split(1, 4)
//! 1: Lit("a")
//! 2: Lit("b")
//! 3: Jmp(6)
//! 4: Lit("a")
//! 5: Lit("c")
//! 6: End
//! ```
//!
//! The `Jmp` instruction causes execution to jump to the specified instruction. In the example it
//! is needed to separate the two threads.
//!
//! Let's step through execution with that program for the input `ac`:
//!
//! 1. We're at PC 0 and IX 0
//! 2. `Split(1, 4)` means we save a thread with PC 4 and IX 0 for trying later
//! 3. Continue at `Lit("a")` which matches, so we advance IX to 1
//! 4. `Lit("b")` doesn't match at IX 1 (`"b" != "c"`), so the thread fails
//! 5. We continue with the previously saved thread at PC 4 and IX 0 (backtracking)
//! 6. Both `Lit("a")` and `Lit("c")` match and we reach `End` -> successful match (index 0 to 2)

use bit_set::BitSet;
use compact_str::CompactString;
use regex_automata::meta::Regex;
use regex_automata::util::look::LookMatcher;
use regex_automata::util::primitives::NonMaxUsize;
use regex_automata::Anchored;
use regex_automata::Input;
use std::fmt;
use std::mem;
use std::ops::Range;
use std::sync::Arc;

use crate::codepoint_len;
use crate::error::RuntimeError;
use crate::prefilter::Prefilter;
use crate::prev_codepoint_ix;
use crate::Assertion;
use crate::Error;
use crate::Result;

/// Enable tracing of VM execution. Only for debugging/investigating.
const OPTION_TRACE: u32 = 1 << 0;
/// When iterating over all matches within a text (e.g. with `find_iter`), empty matches need to be
/// handled specially. If we kept matching at the same position, we'd never stop. So what we do
/// after we've had an empty match, is to advance the position where matching is attempted.
/// If `\G` is used in the pattern, that means it no longer matches. If we didn't tell the VM about
/// the fact that we skipped because of an empty match, it would still treat `\G` as matching. So
/// this option is for communicating that to the VM. Phew.
pub(crate) const OPTION_SKIPPED_EMPTY_MATCH: u32 = 1 << 1;

pub(crate) const DEFAULT_MAX_STACK: usize = 1_000_000;
pub(crate) const DEFAULT_BACKTRACK_LIMIT: usize = 1_000_000;

/// Instruction of the VM.
#[derive(Debug)]
pub enum Insn {
    /// Successful end of program
    End,
    /// Match any character (including newline)
    Any {
        /// Whether to match newline (\n)
        newline: bool,
        /// CRLF mode
        crlf: bool,
    },
    /// Assertions
    Assertion(Assertion),
    /// Match the literal string at the current index
    Lit {
        /// The Literal string
        val: CompactString,
        /// Case insensitive
        casei: bool,
    },
    /// Split execution into two threads. The two fields are positions of instructions. Execution
    /// first tries the first thread. If that fails, the second position is tried.
    Split(usize, usize),
    /// Jump to instruction at position
    Jmp(usize),
    /// Save the current string index into the specified slot
    Save(usize),
    /// Save `0` into the specified slot
    Save0(usize),
    /// Set the string index to the value that was saved in the specified slot
    Restore(usize),
    /// Repeat
    Repeat {
        /// Minimum number of matches
        lo: usize,
        /// Maximum number of matches
        hi: usize,
        /// The instruction after the repeat
        next: usize,
        /// The slot for keeping track of the number of repetitions
        repeat: usize,
        /// Greedy (match as much as possible)
        greedy: bool,
    },
    /// Repeat and prevent infinite loops from empty matches
    RepeatEpsilon {
        /// Minimum number of matches
        lo: usize,
        /// The instruction after the repeat
        next: usize,
        /// The slot for keeping track of the number of repetitions
        repeat: usize,
        /// The slot for saving the previous IX to check if we had an empty match
        check: usize,
        /// Greedy (match as much as possible)
        greedy: bool,
    },
    /// Negative look-around failed
    FailNegativeLookAround,
    /// Set IX back by the specified number of characters
    GoBack(usize),
    /// Back reference to a group number to check
    Backref(usize),
    /// Begin of atomic group
    BeginAtomic,
    /// End of atomic group
    EndAtomic,
    /// Delegate matching to the regex crate
    Delegate {
        /// The regex
        inner: Regex,
        /// The first group number that this regex captures (if it contains groups)
        start_group: usize,
        /// The last group number
        end_group: usize,
        /// Whether to perform anchored search
        anchored: bool,
        /// Whether to drop 0th group
        drop_first: bool,
    },
    /// Anchor to match at the position where the previous match ended
    ContinueFromPreviousMatchEnd,
    /// Continue only if the specified capture group has already been populated as part of the match
    BackrefExistsCondition(usize),
    /// Prefilter to perform unanchored search
    Prefilter(Arc<Prefilter>),
}

/// Sequence of instructions for the VM to execute.
#[derive(Debug)]
pub struct Prog {
    /// Instructions of the program
    pub body: Vec<Insn>,
    n_saves: usize,
}

impl Prog {
    pub(crate) fn new(body: Vec<Insn>, n_saves: usize) -> Prog {
        Prog { body, n_saves }
    }

    #[cfg(debug_assertions)]
    #[doc(hidden)]
    pub(crate) fn debug_print(&self) {
        for (i, insn) in self.body.iter().enumerate() {
            println!("{i:3}: {insn:?}");
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct Branch {
    pc: usize,
    ix: usize,
    nsave: usize,
}

#[derive(Debug, Copy, Clone)]
struct Save {
    slot: usize,
    value: usize,
}

#[derive(Debug)]
pub(crate) struct State {
    /// Saved values indexed by slot. Mostly indices to s, but can be repeat values etc.
    /// Always contains the saves of the current state.
    saves: Vec<usize>,
    /// Stack of backtrack branches.
    stack: Vec<Branch>,
    /// Old saves (slot, value)
    oldsave: Vec<Save>,
    /// Number of saves at the end of `oldsave` that need to be restored to `saves` on pop
    nsave: usize,
    /// Search slots for regex-automata
    inner_slots: Vec<Option<NonMaxUsize>>,
    /// Slot mask
    slot_mask: BitSet,
    /// Visited mask
    visited: BitSet,
}

#[derive(Debug, Clone)]
pub(crate) struct Machine {
    pub prog: Arc<Prog>,
    pub options: u32,
    pub max_stack: usize,
    pub backtrack_limit: usize,
}

#[derive(Debug)]
pub(crate) struct Session {
    pub prog: Arc<Prog>,
    pub options: u32,
    pub max_stack: usize,
    pub backtrack_limit: usize,
    state: State,
}

// Each element in the stack conceptually represents the entire state
// of the machine: the pc (index into prog), the index into the
// string, and the entire vector of saves. However, copying the save
// vector on every push/pop would be inefficient, so instead we use a
// copy-on-write approach for each slot within the save vector. The
// top `nsave` elements in `oldsave` represent the delta from the
// current machine state to the top of stack.

impl State {
    fn new(n_saves: usize) -> State {
        State {
            saves: vec![usize::MAX; n_saves],
            stack: Vec::new(),
            oldsave: Vec::new(),
            nsave: 0,
            inner_slots: Vec::new(),
            slot_mask: BitSet::new(),
            visited: BitSet::new(),
        }
    }

    fn reset(&mut self, n_saves: usize) {
        self.saves.clear();
        self.saves.resize(n_saves, usize::MAX);
        self.stack.clear();
        self.oldsave.clear();
        self.nsave = 0;
        // inner_slots no need to clear, as it is cleared every time before use
        // slot_mask no need to clear, as it is cleared every time before use
        // visited no need to clear, as it is cleared every time before use
    }
}

impl Machine {
    pub(crate) fn new(
        prog: Arc<Prog>,
        max_stack: usize,
        backtrack_limit: usize,
        options: u32,
    ) -> Machine {
        Machine {
            prog,
            options,
            max_stack,
            backtrack_limit,
        }
    }

    pub(crate) fn create_state(prog: &Prog) -> State {
        State::new(prog.n_saves)
    }

    pub(crate) fn create_session(self, state: State) -> Session {
        Session {
            prog: self.prog,
            options: self.options,
            max_stack: self.max_stack,
            backtrack_limit: self.backtrack_limit,
            state,
        }
    }

    #[cfg(debug_assertions)]
    #[doc(hidden)]
    pub(crate) fn debug_print(&self) {
        self.prog.debug_print();
    }
}

impl Session {
    fn explicit_sp(&self) -> usize {
        self.prog.n_saves
    }

    // push a backtrack branch
    fn push(&mut self, pc: usize, ix: usize) -> Result<()> {
        if self.state.stack.len() < self.max_stack {
            let nsave = self.state.nsave;
            self.state.stack.push(Branch { pc, ix, nsave });
            self.state.nsave = 0;
            self.trace_stack("push");
            Ok(())
        } else {
            Err(Error::RuntimeError(RuntimeError::StackOverflow))
        }
    }

    // pop a backtrack branch
    fn pop(&mut self) -> (usize, usize) {
        for _ in 0..self.state.nsave {
            let Save { slot, value } = self.state.oldsave.pop().unwrap();
            self.state.saves[slot] = value;
        }
        let Branch { pc, ix, nsave } = self.state.stack.pop().unwrap();
        self.state.nsave = nsave;
        self.trace_stack("pop");
        (pc, ix)
    }

    fn save(&mut self, slot: usize, val: usize) {
        for i in 0..self.state.nsave {
            // could avoid this iteration with some overhead; worth it?
            if self.state.oldsave[self.state.oldsave.len() - i - 1].slot == slot {
                // already saved, just update
                self.state.saves[slot] = val;
                return;
            }
        }
        self.state.oldsave.push(Save {
            slot,
            value: self.state.saves[slot],
        });
        self.state.nsave += 1;
        self.state.saves[slot] = val;

        self.trace(format_args!("saves: {:?}", self.state.saves));
    }

    fn get(&self, slot: usize) -> usize {
        self.state.saves[slot]
    }

    // push a value onto the explicit stack; note: the entire contents of
    // the explicit stack is saved and restored on backtrack.
    fn stack_push(&mut self, val: usize) {
        if self.state.saves.len() == self.explicit_sp() {
            self.state.saves.push(self.explicit_sp() + 1);
        }
        let explicit_sp = self.explicit_sp();
        let sp = self.get(explicit_sp);
        if self.state.saves.len() == sp {
            self.state.saves.push(val);
        } else {
            self.save(sp, val);
        }
        self.save(explicit_sp, sp + 1);
    }

    // pop a value from the explicit stack
    fn stack_pop(&mut self) -> usize {
        let explicit_sp = self.explicit_sp();
        let sp = self.get(explicit_sp) - 1;
        let result = self.get(sp);
        self.save(explicit_sp, sp);
        result
    }

    /// Get the current number of backtrack branches
    fn backtrack_count(&self) -> usize {
        self.state.stack.len()
    }

    /// Discard backtrack branches that were pushed since the call to `backtrack_count`.
    ///
    /// What we want:
    /// * Keep the current `saves` as they are
    /// * Only keep `count` backtrack branches on `stack`, discard the rest
    /// * Keep the first `oldsave` for each slot, discard the rest (multiple pushes might have
    ///   happened with saves to the same slot)
    fn backtrack_cut(&mut self, count: usize) {
        if self.state.stack.len() == count {
            // no backtrack branches to discard, all good
            return;
        }
        // start and end indexes of old saves for the branch we're cutting to
        let (oldsave_start, oldsave_end) = {
            let mut end = self.state.oldsave.len() - self.state.nsave;
            for &Branch { nsave, .. } in &self.state.stack[count + 1..] {
                end -= nsave;
            }
            let start = end - self.state.stack[count].nsave;
            (start, end)
        };
        reset_bitset(&mut self.state.slot_mask);
        // keep all the old saves of our branch (they're all for different slots)
        for &Save { slot, .. } in &self.state.oldsave[oldsave_start..oldsave_end] {
            self.state.slot_mask.insert(slot);
        }
        let mut oldsave_ix = oldsave_end;
        // for other old saves, keep them only if they're for a slot that we haven't saved yet
        for ix in oldsave_end..self.state.oldsave.len() {
            let Save { slot, .. } = self.state.oldsave[ix];
            let new_slot = self.state.slot_mask.insert(slot);
            if new_slot {
                // put the save we want to keep (ix) after the ones we already have (oldsave_ix)
                // note that it's fine if the indexes are the same (then swapping is a no-op)
                self.state.oldsave.swap(oldsave_ix, ix);
                oldsave_ix += 1;
            }
        }
        self.state.stack.truncate(count);
        self.state.oldsave.truncate(oldsave_ix);
        self.state.nsave = oldsave_ix - oldsave_start;
    }

    fn trace_stack(&self, operation: &str) {
        self.trace(format_args!(
            "stack after {}: {:?}",
            operation, self.state.stack
        ));
    }

    fn trace(&self, args: fmt::Arguments) {
        #[cfg(debug_assertions)]
        if self.options & OPTION_TRACE != 0 {
            Self::do_trace(args);
        }
        let _ = args;
    }

    #[cold]
    #[cfg(debug_assertions)]
    fn do_trace(args: fmt::Arguments) {
        eprintln!("{args}");
    }

    pub(crate) fn run(
        &mut self,
        s: &str,
        range: Range<usize>,
        n_groups: Option<usize>,
    ) -> Result<Option<Vec<usize>>> {
        let mut locations = Vec::new();
        Ok(self
            .run_to(&mut locations, s, range, n_groups)?
            .then_some(locations))
    }

    pub(crate) fn run_with_options(
        &mut self,
        s: &str,
        range: Range<usize>,
        n_groups: Option<usize>,
        options: u32,
    ) -> Result<Option<Vec<usize>>> {
        let old_options = self.options;
        self.options = options;
        let mut locations = Vec::new();
        let r = self.run_to(&mut locations, s, range, n_groups);
        self.options = old_options;
        Ok(r?.then_some(locations))
    }

    pub(crate) fn run_to(
        &mut self,
        locations: &mut Vec<usize>,
        s: &str,
        range: Range<usize>,
        n_groups: Option<usize>,
    ) -> Result<bool> {
        let prefilter = match &self.prog.body[0] {
            Insn::Prefilter(prefilter) => prefilter.clone(),
            _ => return self.run_inner(0, locations, s, range, n_groups),
        };

        let sb = s.as_bytes();

        if !prefilter.assert(sb, &range) {
            return Ok(false);
        }

        let Some(iter) = prefilter.search(sb, &range) else {
            for pos in range.start..=range.end {
                if self.run_inner(1, locations, s, pos..range.end, n_groups)? {
                    return Ok(true);
                }
            }
            return Ok(false);
        };
        let safe_offset = prefilter.safe_offset().unwrap();

        let mut last_success: Option<usize> = None;
        let old_len = locations.len();
        reset_bitset(&mut self.state.visited);
        'outer: for m in iter {
            let mut pos = m.position;

            if let Some(last_success) = last_success {
                let mut cursor = last_success;
                for _ in safe_offset..0 {
                    cursor = prev_codepoint_ix(s, cursor);
                }
                for _ in 0..safe_offset {
                    if cursor >= range.end {
                        break;
                    }
                    cursor += codepoint_len_at(s, cursor);
                }
                if cursor <= pos {
                    break;
                }
            }

            for _ in m.offset..0 {
                if pos >= range.end {
                    continue 'outer;
                }
                pos += codepoint_len_at(s, pos);
            }
            for _ in 0..m.offset {
                if pos <= range.start {
                    continue 'outer;
                }
                pos = prev_codepoint_ix(s, pos);
            }

            if range.start <= pos
                && pos <= range.end
                && last_success.map_or(true, |last_success| pos < last_success)
                && self.state.visited.insert(pos)
            {
                let new_range = pos..range.end;
                let cur_len = locations.len();
                if self.run_inner(1, locations, s, new_range, n_groups)? {
                    if let Some(last_success) = last_success.as_mut() {
                        debug_assert!(pos < *last_success);
                        *last_success = pos;
                        locations.drain(old_len..cur_len);
                    } else {
                        last_success = Some(pos);
                        debug_assert_eq!(old_len, cur_len);
                    }
                }
            }
        }

        Ok(last_success.is_some())
    }

    #[allow(clippy::match_on_vec_items)]
    #[allow(clippy::too_many_lines)]
    fn run_inner(
        &mut self,
        mut pc: usize,
        locations: &mut Vec<usize>,
        s: &str,
        range: Range<usize>,
        n_groups: Option<usize>,
    ) -> Result<bool> {
        self.trace(format_args!("pos\tinstruction"));
        self.state.reset(self.explicit_sp());
        let look_matcher = LookMatcher::new();
        let mut backtrack_count = 0;
        let mut ix = range.start;
        let sb = s.as_bytes();
        assert!(range.end <= sb.len(), "range out of bound");
        loop {
            // break from this loop to fail, causes stack to pop
            'fail: loop {
                self.trace(format_args!("{}\t{} {:?}", ix, pc, self.prog.body[pc]));
                match self.prog.body[pc] {
                    Insn::End => {
                        // save of end position into slot 1 is now done
                        // with an explicit group; we might want to
                        // optimize that.
                        //state.saves[1] = ix;
                        self.trace(format_args!("saves: {:?}", self.state.saves));
                        if let Some(&slot1) = self.state.saves.get(1) {
                            // With some features like keep out (\K), the match start can be after
                            // the match end. Cap the start to <= end.
                            if self.get(0) > slot1 {
                                self.save(0, slot1);
                            }
                        }
                        if let Some(n_groups) = n_groups {
                            locations.extend(self.state.saves.iter().take(n_groups * 2));
                        } else {
                            locations.extend(self.state.saves.iter());
                        };
                        return Ok(true);
                    }
                    Insn::Any { newline, crlf } => {
                        if ix < range.end
                            && (newline || sb[ix] != b'\n' && (!crlf || sb[ix] != b'\r'))
                        {
                            ix += codepoint_len_at(sb, ix);
                        } else {
                            break 'fail;
                        }
                    }
                    Insn::Assertion(assertion) => {
                        if !match assertion {
                            Assertion::StartText => look_matcher.is_start(sb, ix),
                            Assertion::EndText => look_matcher.is_end(sb, ix),
                            Assertion::StartLine { crlf: false } => {
                                look_matcher.is_start_lf(sb, ix)
                            }
                            Assertion::StartLine { crlf: true } => {
                                look_matcher.is_start_crlf(sb, ix)
                            }
                            Assertion::EndLine { crlf: false } => look_matcher.is_end_lf(sb, ix),
                            Assertion::EndLine { crlf: true } => look_matcher.is_end_crlf(sb, ix),
                            Assertion::LeftWordBoundary => {
                                look_matcher.is_word_start_unicode(sb, ix).unwrap()
                            }
                            Assertion::RightWordBoundary => {
                                look_matcher.is_word_end_unicode(sb, ix).unwrap()
                            }
                            Assertion::WordBoundary => {
                                look_matcher.is_word_unicode(sb, ix).unwrap()
                            }
                            Assertion::NotWordBoundary => {
                                look_matcher.is_word_unicode_negate(sb, ix).unwrap()
                            }
                        } {
                            break 'fail;
                        }
                    }
                    Insn::Lit { ref val, casei } => {
                        let end = ix.saturating_add(val.len()).min(range.end);
                        let sb = &sb[ix..end];
                        if !casei {
                            if sb != val.as_bytes() {
                                break 'fail;
                            }
                        } else if !eq_test_exact_size(sb.iter().copied(), val.bytes(), |a, b| {
                            // Do we need a == b shortcut? Is to_ascii_lowercase() fast?
                            a == b || a.to_ascii_lowercase() == b.to_ascii_lowercase()
                        }) && !s.get(ix..end).map_or(false, |s| {
                            eq_test(
                                s.chars().flat_map(char::to_lowercase),
                                val.chars().flat_map(char::to_lowercase),
                                |a, b| a == b,
                            )
                        }) {
                            break 'fail;
                        }
                        ix = end;
                    }
                    Insn::Split(x, y) => {
                        self.push(y, ix)?;
                        pc = x;
                        continue;
                    }
                    Insn::Jmp(target) => {
                        pc = target;
                        continue;
                    }
                    Insn::Save(slot) => self.save(slot, ix),
                    Insn::Save0(slot) => self.save(slot, 0),
                    Insn::Restore(slot) => ix = self.get(slot),
                    Insn::Repeat {
                        lo,
                        hi,
                        next,
                        repeat,
                        greedy,
                    } => {
                        let repcount = self.get(repeat);
                        if repcount == hi {
                            pc = next;
                            continue;
                        }
                        self.save(repeat, repcount + 1);
                        if repcount >= lo {
                            self.push(next, ix)?;
                            if !greedy {
                                pc = next;
                                continue;
                            }
                        }
                    }
                    Insn::RepeatEpsilon {
                        lo,
                        next,
                        repeat,
                        check,
                        greedy,
                    } => {
                        let repcount = self.get(repeat);
                        if repcount > lo && self.get(check) == ix {
                            // prevent zero-length match on repeat
                            break 'fail;
                        }
                        self.save(repeat, repcount + 1);
                        if repcount >= lo {
                            self.save(check, ix);
                            self.push(next, ix)?;
                            if !greedy {
                                pc = next;
                                continue;
                            }
                        }
                    }
                    Insn::GoBack(count) => {
                        for _ in 0..count {
                            if ix == 0 {
                                break 'fail;
                            }
                            ix = prev_codepoint_ix(sb, ix);
                        }
                    }
                    Insn::FailNegativeLookAround => {
                        // Reaching this instruction means that the body of the
                        // look-around matched. Because it's a *negative* look-around,
                        // that means the look-around itself should fail (not match).
                        // But before, we need to discard all the states that have
                        // been pushed with the look-around, because we don't want to
                        // explore them.
                        loop {
                            let (popped_pc, _) = self.pop();
                            if popped_pc == pc + 1 {
                                // We've reached the state that would jump us to
                                // after the look-around (in case the look-around
                                // succeeded). That means we popped enough states.
                                break;
                            }
                        }
                        break 'fail;
                    }
                    Insn::Backref(slot) => {
                        let lo = self.get(slot);
                        let hi = self.get(slot + 1);
                        if lo == usize::MAX || hi == usize::MAX {
                            // Referenced group hasn't matched, so the backref doesn't match either
                            break 'fail;
                        }
                        let ref_text = &sb[lo..hi];
                        let end = ix.saturating_add(ref_text.len()).min(range.end);
                        let sb = &sb[ix..end];
                        if sb != ref_text {
                            break 'fail;
                        }
                        ix = end;
                    }
                    Insn::BackrefExistsCondition(group) => {
                        let lo = self.get(group * 2);
                        if lo == usize::MAX {
                            // Referenced group hasn't matched, so the backref doesn't match either
                            break 'fail;
                        }
                    }
                    Insn::BeginAtomic => {
                        let count = self.backtrack_count();
                        self.stack_push(count);
                    }
                    Insn::EndAtomic => {
                        let count = self.stack_pop();
                        self.backtrack_cut(count);
                    }
                    Insn::Delegate {
                        ref inner,
                        start_group,
                        end_group,
                        anchored,
                        drop_first,
                    } => {
                        debug_assert!(start_group <= end_group);
                        let input = Input::new(s).span(ix..range.end).anchored(if anchored {
                            Anchored::Yes
                        } else {
                            Anchored::No
                        });
                        if start_group == end_group {
                            // No groups, so we can use faster methods
                            match inner.search_half(&input) {
                                Some(m) => ix = m.offset(),
                                _ => break 'fail,
                            }
                        } else {
                            // Do we need to check start_group + 1 == end_group && !drop_first for non-capturing search?
                            // No, because regex-automata will check this for us.
                            let offset = usize::from(drop_first);
                            self.state
                                .inner_slots
                                .resize((end_group - start_group + offset) * 2, None);
                            if inner
                                .search_slots(&input, &mut self.state.inner_slots)
                                .is_some()
                            {
                                for i in 0..(end_group - start_group) {
                                    let slot = (start_group + i) * 2;
                                    if let Some(start) = self.state.inner_slots[(i + offset) * 2] {
                                        let end =
                                            self.state.inner_slots[(i + offset) * 2 + 1].unwrap();
                                        self.save(slot, start.get());
                                        self.save(slot + 1, end.get());
                                    } else {
                                        self.save(slot, usize::MAX);
                                        self.save(slot + 1, usize::MAX);
                                    }
                                }
                                ix = self.state.inner_slots[1].unwrap().get();
                            } else {
                                break 'fail;
                            }
                        }
                    }
                    Insn::ContinueFromPreviousMatchEnd => {
                        if ix > range.start || self.options & OPTION_SKIPPED_EMPTY_MATCH != 0 {
                            break 'fail;
                        }
                    }
                    Insn::Prefilter(_) => {
                        unreachable!("Insn::Prefile can only appear at the beginning of Prog")
                    }
                }
                pc += 1;
            }
            self.trace(format_args!("fail"));
            // "break 'fail" goes here
            if self.state.stack.is_empty() {
                return Ok(false);
            }

            backtrack_count += 1;
            if backtrack_count > self.backtrack_limit {
                return Err(Error::RuntimeError(RuntimeError::BacktrackLimitExceeded));
            }

            let (newpc, newix) = self.pop();
            pc = newpc;
            ix = newix;
        }
    }
}

fn create_session(prog: Arc<Prog>) -> Session {
    let state = Machine::create_state(&prog);
    let machine = Machine::new(
        prog,
        DEFAULT_MAX_STACK,
        DEFAULT_BACKTRACK_LIMIT,
        OPTION_TRACE,
    );

    machine.create_session(state)
}

/// Run the program with trace printing for debugging.
#[doc(hidden)]
pub fn run_trace(prog: Arc<Prog>, s: &str, range: Range<usize>) -> Result<Option<Vec<usize>>> {
    let mut session = create_session(prog);
    session.run_with_options(s, range, None, OPTION_TRACE)
}

/// Run the program with default options.
#[doc(hidden)]
pub fn run_default(prog: Arc<Prog>, s: &str, range: Range<usize>) -> Result<Option<Vec<usize>>> {
    let mut session = create_session(prog);
    session.run(s, range, None)
}

/// Run the program with trace printing for debugging.
#[doc(hidden)]
pub fn run_trace_from_pos(prog: Arc<Prog>, s: &str, pos: usize) -> Result<Option<Vec<usize>>> {
    run_trace(prog, s, pos..s.len())
}

/// Run the program with default options.
#[doc(hidden)]
pub fn run_default_from_pos(prog: Arc<Prog>, s: &str, pos: usize) -> Result<Option<Vec<usize>>> {
    run_default(prog, s, pos..s.len())
}

#[inline]
fn codepoint_len_at(s: impl AsRef<[u8]>, ix: usize) -> usize {
    codepoint_len(s.as_ref()[ix])
}

#[inline]
fn eq_test<T: Copy, F: Fn(T, T) -> bool>(
    mut a: impl Iterator<Item = T>,
    mut b: impl Iterator<Item = T>,
    pred: F,
) -> bool {
    loop {
        match (a.next(), b.next()) {
            (Some(a), Some(b)) if pred(a, b) => (),
            (None, None) => break true,
            _ => break false,
        }
    }
}

#[inline]
fn eq_test_exact_size<T: Copy, F: Fn(T, T) -> bool>(
    a: impl ExactSizeIterator<Item = T>,
    b: impl ExactSizeIterator<Item = T>,
    pred: F,
) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.zip(b).all(|(a, b)| pred(a, b))
}

fn reset_bitset(set: &mut BitSet) {
    let mut bitvec = mem::take(set).into_bit_vec();
    bitvec.truncate(0);
    *set = BitSet::from_bit_vec(bitvec);
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::{quickcheck, Arbitrary, Gen};

    #[test]
    fn state_push_pop() {
        let mut vm = create_session(Arc::new(Prog {
            body: Vec::new(),
            n_saves: 1,
        }));

        vm.push(0, 0).unwrap();
        vm.push(1, 1).unwrap();
        assert_eq!(vm.pop(), (1, 1));
        assert_eq!(vm.pop(), (0, 0));
        assert!(vm.state.stack.is_empty());

        vm.push(2, 2).unwrap();
        assert_eq!(vm.pop(), (2, 2));
        assert!(vm.state.stack.is_empty());
    }

    #[test]
    fn state_save_override() {
        let mut vm = create_session(Arc::new(Prog {
            body: Vec::new(),
            n_saves: 1,
        }));
        vm.save(0, 10);
        vm.push(0, 0).unwrap();
        vm.save(0, 20);
        assert_eq!(vm.pop(), (0, 0));
        assert_eq!(vm.get(0), 10);
    }

    #[test]
    fn state_save_override_twice() {
        let mut vm = create_session(Arc::new(Prog {
            body: Vec::new(),
            n_saves: 1,
        }));
        vm.save(0, 10);
        vm.push(0, 0).unwrap();
        vm.save(0, 20);
        vm.push(1, 1).unwrap();
        vm.save(0, 30);

        assert_eq!(vm.get(0), 30);
        assert_eq!(vm.pop(), (1, 1));
        assert_eq!(vm.get(0), 20);
        assert_eq!(vm.pop(), (0, 0));
        assert_eq!(vm.get(0), 10);
    }

    #[test]
    fn state_explicit_stack() {
        let mut vm = create_session(Arc::new(Prog {
            body: Vec::new(),
            n_saves: 1,
        }));
        vm.stack_push(11);
        vm.stack_push(12);

        vm.push(100, 101).unwrap();
        vm.stack_push(13);
        assert_eq!(vm.stack_pop(), 13);
        vm.stack_push(14);
        assert_eq!(vm.pop(), (100, 101));

        // Note: 14 is not there because it was pushed as part of the backtrack branch
        assert_eq!(vm.stack_pop(), 12);
        assert_eq!(vm.stack_pop(), 11);
    }

    #[test]
    fn state_backtrack_cut_simple() {
        let mut vm = create_session(Arc::new(Prog {
            body: Vec::new(),
            n_saves: 2,
        }));
        vm.save(0, 1);
        vm.save(1, 2);

        let count = vm.backtrack_count();

        vm.push(0, 0).unwrap();
        vm.save(0, 3);
        assert_eq!(vm.backtrack_count(), 1);

        vm.backtrack_cut(count);
        assert_eq!(vm.backtrack_count(), 0);
        assert_eq!(vm.get(0), 3);
        assert_eq!(vm.get(1), 2);
    }

    #[test]
    fn state_backtrack_cut_complex() {
        let mut vm = create_session(Arc::new(Prog {
            body: Vec::new(),
            n_saves: 2,
        }));
        vm.save(0, 1);
        vm.save(1, 2);

        vm.push(0, 0).unwrap();
        vm.save(0, 3);

        let count = vm.backtrack_count();

        vm.push(1, 1).unwrap();
        vm.save(0, 4);
        vm.push(2, 2).unwrap();
        vm.save(1, 5);
        assert_eq!(vm.backtrack_count(), 3);

        vm.backtrack_cut(count);
        assert_eq!(vm.backtrack_count(), 1);
        assert_eq!(vm.get(0), 4);
        assert_eq!(vm.get(1), 5);

        vm.pop();
        assert_eq!(vm.backtrack_count(), 0);
        // Check that oldsave were set correctly
        assert_eq!(vm.get(0), 1);
        assert_eq!(vm.get(1), 2);
    }

    #[derive(Clone, Debug)]
    enum Operation {
        Push,
        Pop,
        Save(usize, usize),
    }

    impl Arbitrary for Operation {
        fn arbitrary(g: &mut Gen) -> Self {
            match g.choose(&[0, 1, 2]) {
                Some(0) => Operation::Push,
                Some(1) => Operation::Pop,
                _ => Operation::Save(
                    *g.choose(&[0usize, 1, 2, 3, 4]).unwrap(),
                    usize::arbitrary(g),
                ),
            }
        }
    }

    fn check_saves_for_operations(operations: Vec<Operation>) -> bool {
        let slots = operations
            .iter()
            .map(|o| match o {
                &Operation::Save(slot, _) => slot + 1,
                _ => 0,
            })
            .max()
            .unwrap_or(0);
        if slots == 0 {
            // No point checking if there's no save instructions
            return true;
        }

        // Stack with the complete VM state (including saves)
        let mut stack = Vec::new();
        let mut saves = vec![usize::MAX; slots];

        let mut vm = create_session(Arc::new(Prog {
            body: Vec::new(),
            n_saves: slots,
        }));

        let mut expected = Vec::new();
        let mut actual = Vec::new();

        for operation in operations {
            match operation {
                Operation::Push => {
                    // We're not checking pc and ix later, so don't bother
                    // putting in random values.
                    stack.push((0, 0, saves.clone()));
                    vm.push(0, 0).unwrap();
                }
                Operation::Pop => {
                    // Note that because we generate the operations randomly
                    // there might be more pops than pushes. So ignore a pop
                    // if the stack was empty.
                    if let Some((_, _, previous_saves)) = stack.pop() {
                        saves = previous_saves;
                        vm.pop();
                    }
                }
                Operation::Save(slot, value) => {
                    saves[slot] = value;
                    vm.save(slot, value);
                }
            }

            // Remember state of saves for checking later
            expected.push(saves.clone());
            let mut actual_saves = vec![usize::MAX; slots];
            for i in 0..slots {
                actual_saves[i] = vm.get(i);
            }
            actual.push(actual_saves);
        }

        expected == actual
    }

    quickcheck! {
        fn state_save_quickcheck(operations: Vec<Operation>) -> bool {
            check_saves_for_operations(operations)
        }
    }
}
