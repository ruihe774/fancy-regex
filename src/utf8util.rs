#[inline]
pub fn codepoint_len(b: u8) -> usize {
    match b {
        b if b < 0x80 => 1,
        b if b < 0xe0 => 2,
        b if b < 0xf0 => 3,
        _ => 4,
    }
}

#[allow(clippy::cast_possible_wrap)]
#[inline]
fn is_utf8_first_byte(b: u8) -> bool {
    (b as i8) >= -0x40
}

// The size of usize is the width of general purpose registers
// unless we are using x32 ABI. (Who actually use it? Sorry, Dr. Kruth.)
const CHUNK_SIZE: usize = std::mem::size_of::<usize>();

// See comparison of assembly at
// https://play.rust-lang.org/?version=stable&mode=release&edition=2021&gist=c300a5e88531207b146cc56e1f5eb985
#[inline]
fn prev_nth_codepoint_ix_inner(bytes: &[u8], mut ix: usize, mut n: usize) -> Option<usize> {
    debug_assert_ne!(n, 0, "n cannot be zero");

    loop {
        if let Some(start) = ix.checked_sub(CHUNK_SIZE) {
            // I may be performing unaligned load here,
            // but I don't want to make things any more complicated.
            let chunk: [u8; CHUNK_SIZE] = bytes[start..ix].try_into().unwrap();
            for b in chunk.into_iter().rev() {
                ix -= 1;
                n -= usize::from(is_utf8_first_byte(b));
                if n == 0 {
                    return Some(ix);
                }
            }
            debug_assert_eq!(ix, start);
            // The compiler is stupid.
            ix = start;
        } else {
            // Plain old code path.
            loop {
                ix = ix.checked_sub(1)?;
                n -= usize::from(is_utf8_first_byte(bytes[ix]));
                if n == 0 {
                    return Some(ix);
                }
            }
        }
    }
}

#[inline]
pub fn prev_nth_codepoint_ix(s: impl AsRef<[u8]>, ix: usize, n: usize) -> Option<usize> {
    if n == 0 {
        Some(ix)
    } else {
        prev_nth_codepoint_ix_inner(s.as_ref(), ix, n)
    }
}

#[inline]
pub fn prev_codepoint_ix(s: impl AsRef<[u8]>, ix: usize) -> usize {
    prev_nth_codepoint_ix_inner(s.as_ref(), ix, 1).expect("out of bound")
}

#[inline]
fn next_nth_codepoint_ix_inner(bytes: &[u8], mut ix: usize, mut n: usize) -> Option<usize> {
    debug_assert_ne!(n, 0, "n cannot be zero");

    loop {
        if let Some(chunk) = bytes[ix..].chunks_exact(CHUNK_SIZE).next() {
            let chunk: [u8; CHUNK_SIZE] = chunk.try_into().unwrap();
            for b in chunk {
                let first_byte = is_utf8_first_byte(b);
                ix += codepoint_len(b) & if first_byte { usize::MAX } else { 0 };
                n -= usize::from(first_byte);
                if n == 0 {
                    return Some(ix);
                }
            }
        } else {
            for _ in 0..n {
                ix += codepoint_len(*bytes.get(ix)?);
            }
            return Some(ix);
        }
    }
}

#[inline]
pub fn next_nth_codepoint_ix(s: impl AsRef<[u8]>, ix: usize, n: usize) -> Option<usize> {
    if n == 0 {
        Some(ix)
    } else {
        next_nth_codepoint_ix_inner(s.as_ref(), ix, n)
    }
}

#[inline]
pub fn next_codepoint_ix(s: impl AsRef<[u8]>, ix: usize) -> usize {
    ix + codepoint_len(s.as_ref()[ix])
}
