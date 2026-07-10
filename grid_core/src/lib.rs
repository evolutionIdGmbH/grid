//! GRID Rust kernels (DESIGN.md SS2; kernel version 4).
//!
//! `RustWalker` is the incremental-state trie walk; `RustVerdicts` is the
//! per-step CD-group verdict batch + the LALR virtual-stack machinery behind it
//! (simulate, allowed-set, `advance_frames` = the SS2 `lalr_advance` symbol).
//! Both are transcriptions of the Python executable specification
//! (`grid/trie/walk.py`, `grid/mask/producer.py`, `grid/lalr/stack.py`), bound
//! bit-identical by tests/trie/test_rust_parity.py and
//! tests/mask/test_kernel_parity.py.
//!
//! Kernel v4: `RustVerdicts` owns a persistent, structurally-interned stack
//! arena (`(parent, state)` dedup) with cross-call memos for allowed sets,
//! EOS legality, and shifts. Parser configurations recur heavily across token
//! positions, so warm-path verdicts become memo lookups instead of fresh
//! simulations; `hit_pass` assembles the full allowed-id buffer
//! (ci ++ cd-pass ++ eos) in one FFI call. Python addresses nodes by `kidx`
//! (intern index); `reset_interning` invalidates all outstanding kidx and is
//! guarded by a generation counter on the Python side.
//!
//! Terminal sets are fixed-width bitmask arrays `[u64; W]`, monomorphized for
//! W in {1, 2, 4, 8} (up to 512 terminals; W=1 compiles to the original scalar
//! ops). The width is chosen from `n_terminals` at construction and exposed as
//! `.width`; masks cross the FFI as little-endian word lists.
//!
//! Kernel v7: cold-miss materialization moves in-kernel. `RustWalker::
//! walk_payload` returns the walk as (ci i32-le bytes, opaque blob v1);
//! `RustVerdicts::register_blob` parses the blob, builds the VGroups,
//! adaptive-encodes the ci payload, hashes the entry id (BLAKE2b-128, byte-
//! identical to the Python hashlib construction) and registers the entry —
//! all inside ONE GIL-released call, deduplicated kernel-side by entry id.
//! The blob doubles as the cross-producer export payload: a foreign kernel's
//! register_blob recomputes VEvents/tails under ITS OWN lexicons, exactly
//! like today's register_bytes-from-kernel_groups semantics. To make the
//! detached build safe against concurrent verdict/session calls, every
//! v7-reachable RustVerdicts method is `&self`: the entry store and the v6
//! session tables live behind RwLocks (lock order: sessions -> mem ->
//! entries; the session-tables lock nests inside sessions/mem only).

use blake2::digest::consts::U16;
use blake2::{Blake2b, Digest};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::sync::{Arc, Mutex, RwLock};
use std::collections::{HashMap, HashSet};

type Blake2b128 = Blake2b<U16>;

const DEAD: i32 = -1;

// ------------------------------------------------------------------ mask ops

#[inline]
fn m_from_words<const W: usize>(v: &[u64]) -> [u64; W] {
    let mut m = [0u64; W];
    for (i, w) in v.iter().take(W).enumerate() {
        m[i] = *w;
    }
    m
}

#[inline]
fn m_bit<const W: usize>(m: &[u64; W], t: u32) -> bool {
    let w = (t / 64) as usize;
    w < W && (m[w] >> (t % 64)) & 1 == 1
}

#[inline]
fn m_set<const W: usize>(m: &mut [u64; W], t: u32) {
    m[(t / 64) as usize] |= 1u64 << (t % 64);
}

#[inline]
fn m_and<const W: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; W] {
    let mut out = [0u64; W];
    for i in 0..W {
        out[i] = a[i] & b[i];
    }
    out
}

#[inline]
fn m_and_not<const W: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; W] {
    let mut out = [0u64; W];
    for i in 0..W {
        out[i] = a[i] & !b[i];
    }
    out
}

#[inline]
fn m_any<const W: usize>(a: &[u64; W]) -> bool {
    a.iter().any(|w| *w != 0)
}

#[inline]
fn m_and_any<const W: usize>(a: &[u64; W], b: &[u64; W]) -> bool {
    (0..W).any(|i| a[i] & b[i] != 0)
}

/// Lowest set bit (ascending terminal id), or None.
#[inline]
fn m_first<const W: usize>(a: &[u64; W]) -> Option<u32> {
    for i in 0..W {
        if a[i] != 0 {
            return Some(i as u32 * 64 + a[i].trailing_zeros());
        }
    }
    None
}

/// Iterate set bits in ascending order; `f` returns true to stop (found).
#[inline]
fn m_find<const W: usize>(a: &[u64; W], mut f: impl FnMut(u32) -> bool) -> Option<u32> {
    for i in 0..W {
        let mut w = a[i];
        while w != 0 {
            let t = i as u32 * 64 + w.trailing_zeros();
            if f(t) {
                return Some(t);
            }
            w &= w - 1;
        }
    }
    None
}

#[inline]
fn width_for(n_terminals: usize) -> Option<usize> {
    match n_terminals.div_ceil(64) {
        0 | 1 => Some(1),
        2 => Some(2),
        3 | 4 => Some(4),
        5..=8 => Some(8),
        _ => None,
    }
}

// ------------------------------------------------- walk threads (W8, rayon)
//
// GRID_WALK_THREADS (read per walk call; unset/0/1/invalid = 0) is the kill
// switch for rayon intra-walk parallelism: at 0 the walk is the sequential
// `walk_raw`, byte-for-byte the pre-W8 code path. >= 2 dispatches to
// `walk_raw_par` on a dedicated pool of min(GRID_WALK_THREADS, ncpu) threads,
// lazily (re)built per PID so a forked vLLM worker never touches pool threads
// that died with its parent. GRID_WALK_PAR_MIN (default 4096 trie nodes;
// test-tunable) keeps small tries inline where task overhead would exceed the
// walk itself. Parallel output is bit-identical to sequential
// (tests/trie/test_walk_threads.py differential corpus).

const WALK_PAR_MIN_DEFAULT: usize = 4096;

fn env_usize(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(s) => s.trim().parse::<usize>().unwrap_or(default),
        Err(_) => default,
    }
}

#[allow(clippy::type_complexity)]
static WALK_POOL: Mutex<Option<(u32, usize, Arc<rayon::ThreadPool>)>> = Mutex::new(None);

/// Dedicated walk pool: `threads` capped at ncpu; None when the cap leaves
/// < 2 threads or the pool cannot be built (callers fall back to sequential).
fn walk_pool(threads: usize) -> Option<Arc<rayon::ThreadPool>> {
    let ncpu = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let want = threads.min(ncpu);
    if want < 2 {
        return None;
    }
    let pid = std::process::id();
    let mut slot = WALK_POOL.lock().unwrap();
    if let Some((p, n, pool)) = slot.as_ref() {
        if *p == pid && *n == want {
            return Some(Arc::clone(pool));
        }
        if *p != pid {
            // forked child: the parent's pool threads do not exist here — never
            // run Drop on the inherited registry (leak is bounded: once per
            // fork); rebuild lazily below.
            std::mem::forget(slot.take());
        }
    }
    // GRID_WALK_NICE (default 10): walk threads yield to the engine under
    // host CPU/membw contention — the fresh-schema window's co-batch
    // degradation is walk compute sharing the host with the engine loop
    // (measured: MORE walk threads = LESS degradation because the window
    // shrinks; niceness attacks the residual). Linux-only; 0 disables.
    let nice = env_usize("GRID_WALK_NICE", 10).min(19) as i32;
    let mut builder = rayon::ThreadPoolBuilder::new().num_threads(want);
    if nice > 0 {
        builder = builder.start_handler(move |_| {
            #[cfg(target_os = "linux")]
            unsafe {
                let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
                libc::setpriority(libc::PRIO_PROCESS, tid, nice);
            }
            #[cfg(not(target_os = "linux"))]
            let _ = nice;
        });
    }
    let pool = Arc::new(builder.build().ok()?);
    *slot = Some((pid, want, Arc::clone(&pool)));
    Some(pool)
}

// ------------------------------------------------------------------ walker

struct RustWalkerImpl<const W: usize> {
    nodes: Vec<u64>,
    trans: Vec<i32>, // [state * 256 + byte]
    accept: Vec<i32>,
    accepts_all: Vec<[u64; W]>,
    live: Vec<[u64; W]>,
    ignored: [u64; W],
    literal: [u64; W],
    dfa_start: i32,
    lex_allowed: Option<HashMap<u32, HashSet<Vec<u8>>>>,
    lex_prefixes: Option<HashMap<u32, HashSet<Vec<u8>>>>,
    aliases: HashMap<u32, Vec<u32>>,
}

struct Frame {
    end: usize,
    dfa_state: i32,
    // The frame's segment is path[seg_start..seg_start + seg_len] in the ONE
    // shared DFS path buffer (walk_raw), truncated on frame pop exactly like
    // the shared `events` vector — no per-frame owned Vec, no per-node clone.
    seg_start: usize,
    seg_len: usize,
    last_len: usize, // relative to seg_start
    last_state: i32,
    events_len: usize,
    n_real: u8,
    cd_flag: bool,
}

/// One forced-emission event on the walk, with the verdict-equivalence
/// components precomputed at event creation (lexicon walks only):
/// `pass` = candidates surviving lexeme_ok(t, lexeme) — exactly the
/// `VEvent::cand_pass` RustVerdicts::register derives from (cands, lexeme) —
/// and `ign_pick` = the min-priority ignored candidate (the register-time
/// `VEvent::ign_pick`, a pure function of `cands`). The per-step CD verdict
/// consumes an event ONLY through (pass, ign_pick), so they key CD groups.
struct Ev<const W: usize> {
    cands: [u64; W],
    lexeme: Vec<u8>,
    pass: [u64; W],
    ign_pick: i64,
}

type WalkGroups = Vec<(Vec<(Vec<u64>, u32)>, Vec<Py<PyBytes>>, Py<PyBytes>, Vec<u32>)>;
type WalkGroupRaw = (Vec<(Vec<u64>, u32)>, Vec<Vec<u8>>, Vec<u8>, Vec<u32>);
type WalkGroupsRaw = Vec<WalkGroupRaw>;

impl<const W: usize> RustWalkerImpl<W> {
    #[inline]
    fn tr(&self, state: i32, byte: u8) -> i32 {
        self.trans[(state as usize) * 256 + byte as usize]
    }

    fn lexeme_ok(&self, t: u32, lexeme: &[u8]) -> bool {
        match &self.lex_allowed {
            None => true,
            Some(m) => match m.get(&t) {
                None => true,
                Some(set) => set.contains(lexeme),
            },
        }
    }

    fn prefix_ok(&self, t: u32, partial: &[u8]) -> bool {
        match &self.lex_prefixes {
            None => true,
            Some(m) => match m.get(&t) {
                None => true,
                Some(set) => set.contains(partial),
            },
        }
    }

    /// pick_viable (walk.py): priority-ordered viable real candidate passing its
    /// lexicon, else an ignored candidate, else None.
    fn pick_viable(&self, cands: &[u64; W], lexeme: &[u8], viable: &[u64; W]) -> Option<u32> {
        let real = m_and(cands, viable);
        for pass in 0..2u8 {
            let pool = if pass == 0 { m_and(&real, &self.literal) } else { m_and_not(&real, &self.literal) };
            if let Some(t) = m_find(&pool, |t| self.lexeme_ok(t, lexeme)) {
                return Some(t);
            }
        }
        let ign = m_and(cands, &self.ignored);
        for pass in 0..2u8 {
            let pool = if pass == 0 { m_and(&ign, &self.literal) } else { m_and_not(&ign, &self.literal) };
            if let Some(t) = m_first(&pool) {
                return Some(t);
            }
        }
        None
    }

    fn partial_viable(&self, seg: &[u8], dfa_state: i32, cand_mask: &[u64; W]) -> bool {
        let pool = m_and(&self.live[dfa_state as usize], cand_mask);
        m_find(&pool, |t| m_bit(&self.ignored, t) || self.prefix_ok(t, seg)).is_some()
    }

    /// First of `pool` by producer priority (literals ascending, then named) —
    /// identical to RustVerdictsImpl::pick_first (register's ign_pick).
    #[inline]
    fn pick_first(&self, pool: &[u64; W]) -> i64 {
        if let Some(t) = m_first(&m_and(pool, &self.literal)) {
            return t as i64;
        }
        if let Some(t) = m_first(&m_and_not(pool, &self.literal)) {
            return t as i64;
        }
        -1
    }

    fn seed(&self, remainder: &[u8]) -> (i32, usize, i32) {
        let mut cur = self.dfa_start;
        let mut last_len = 0usize;
        let mut last_state = -1i32;
        for (i, &b) in remainder.iter().enumerate() {
            cur = self.tr(cur, b);
            debug_assert!(cur != DEAD, "remainder must be scannable");
            if self.accept[cur as usize] != -1 {
                last_len = i + 1;
                last_state = cur;
            }
        }
        (cur, last_len, last_state)
    }

    /// -> (ci, groups): identical semantics to kernel v1/v2; masks are [u64; W].
    /// GIL-free (kernel v4): callers detach() around this so ms-scale cold
    /// walks overlap Python work (SS6 batch scheduling contract); PyBytes
    /// wrapping happens after reattach.
    ///
    /// W8: `walk_range` below is a lockstep transcript of this per-node body
    /// (the rayon path). Any change here MUST land there too —
    /// tests/trie/test_walk_threads.py binds the two bit-for-bit.
    fn walk_raw(&self, remainder: &[u8], a_mask: &[u64; W]) -> (Vec<u32>, WalkGroupsRaw) {
        let n = self.nodes.len();
        let mut ci: Vec<u32> = Vec::new();
        let lex_sensitive = self.lex_allowed.is_some();
        let mut group_ix: HashMap<Vec<u8>, usize> = HashMap::new();
        let mut groups: Vec<(Vec<(Vec<u64>, u32)>, Vec<Vec<u8>>, Vec<u8>, Vec<u32>)> = Vec::new();
        let mut a_or_ign = *a_mask;
        for i in 0..W {
            a_or_ign[i] |= self.ignored[i];
        }

        let mut events: Vec<Ev<W>> = Vec::new();
        // ONE shared DFS path buffer: each frame's segment is a (start, len)
        // window into it; truncated on frame pop like `events`. Child nodes
        // extend the parent's window in place (parent's window is always the
        // buffer tail while its subtree is walked); segment bytes only
        // materialize at emission (event lexeme / pending requeue) and into
        // group representatives.
        let mut path: Vec<u8> = Vec::with_capacity(remainder.len() + 64);
        path.extend_from_slice(remainder);
        let (s_state, s_len, s_last) = self.seed(remainder);
        let mut stack: Vec<Frame> = vec![Frame {
            end: n + 1,
            dfa_state: s_state,
            seg_start: 0,
            seg_len: remainder.len(),
            last_len: s_len,
            last_state: s_last,
            events_len: 0,
            n_real: 0,
            cd_flag: false,
        }];

        let mut i = 0usize;
        while i < n {
            while stack.len() > 1 && i >= stack.last().unwrap().end {
                stack.pop();
            }
            {
                let f = stack.last().unwrap();
                events.truncate(f.events_len);
                path.truncate(f.seg_start + f.seg_len);
            }
            let word = self.nodes[i];
            let byte = (word & 0xFF) as u8;
            let tid_raw = ((word >> 8) & 0xFF_FFFF) as i64 - 1;
            let size = (word >> 32) as usize;

            let parent = stack.last().unwrap();
            let mut cur = parent.dfa_state;
            let mut seg_start = parent.seg_start;
            let mut seg_len = parent.seg_len;
            let base = seg_start + seg_len; // parent window end == path.len()
            debug_assert_eq!(base, path.len());
            let mut last_len = parent.last_len;
            let mut last_state = parent.last_state;
            let mut n_real = parent.n_real;
            let mut cd_flag = parent.cd_flag;
            let mut reject = false;
            let events_base = events.len();

            let mut pending: Vec<u8> = vec![byte];
            let mut idx = 0usize;
            while idx < pending.len() {
                let b = pending[idx];
                idx += 1;
                let nx = self.tr(cur, b);
                if nx != DEAD {
                    path.push(b);
                    seg_len += 1;
                    cur = nx;
                    if self.accept[nx as usize] != -1 {
                        last_len = seg_len;
                        last_state = nx;
                    }
                    continue;
                }
                if last_state == -1 {
                    reject = true;
                    break;
                }
                let cands = self.accepts_all[last_state as usize];
                let lexeme: Vec<u8> = path[seg_start..seg_start + last_len].to_vec();
                if n_real == 0 {
                    match self.pick_viable(&cands, &lexeme, a_mask) {
                        None => {
                            reject = true;
                            break;
                        }
                        Some(t) => {
                            if !m_bit(&self.ignored, t) {
                                n_real = 1;
                            }
                        }
                    }
                } else {
                    let pure_ignored =
                        m_and_any(&cands, &self.ignored) && !m_any(&m_and_not(&cands, &self.ignored));
                    if !pure_ignored {
                        cd_flag = true;
                    }
                }
                // verdict-equivalence components, computed once per event (the
                // event is shared by every node in the subtree via the frame
                // stack); zeros when no lexicon — the non-lex key ignores them
                let (pass, ign_pick) = if lex_sensitive {
                    let mut p = [0u64; W];
                    m_find(&cands, |t| {
                        if self.lexeme_ok(t, &lexeme) {
                            m_set(&mut p, t);
                        }
                        false
                    });
                    (p, self.pick_first(&m_and(&cands, &self.ignored)))
                } else {
                    ([0u64; W], -1)
                };
                events.push(Ev { cands, lexeme, pass, ign_pick });
                // requeue rest then the dead byte, exactly like the Python cascade
                let rest_len = seg_len - last_len;
                let mut tail: Vec<u8> = Vec::with_capacity(rest_len + 1 + pending.len() - idx);
                tail.extend_from_slice(&path[seg_start + last_len..seg_start + seg_len]);
                tail.push(b);
                tail.extend_from_slice(&pending[idx..]);
                pending = tail;
                idx = 0;
                cur = self.dfa_start;
                // seg.clear(): drop this node's window but keep the parent's
                // window (path[..base]) intact for siblings after pop
                path.truncate(base);
                seg_start = base;
                seg_len = 0;
                last_len = 0;
                last_state = -1;
            }

            if reject {
                events.truncate(events_base);
                i += size;
                continue;
            }

            // node verdict (identical to walk.py)
            let seg = &path[seg_start..seg_start + seg_len];
            let verdict_ci: Option<bool> = if n_real == 0 {
                if !seg.is_empty() && !self.partial_viable(seg, cur, &a_or_ign) {
                    events.truncate(events_base);
                    i += size; // monotone subtree prune
                    continue;
                }
                Some(true)
            } else if cd_flag || n_real >= 2 {
                Some(false)
            } else if seg.is_empty() {
                Some(true)
            } else if self.partial_viable(seg, cur, &self.ignored) {
                Some(true)
            } else {
                Some(false)
            };

            if tid_raw >= 0 {
                let tid = tid_raw as u32;
                match verdict_ci {
                    Some(true) => match self.aliases.get(&tid) {
                        Some(all) => ci.extend_from_slice(all),
                        None => ci.push(tid),
                    },
                    Some(false) => {
                        // group key (mirrors cache.make_entry): VERDICT-EQUIVALENCE
                        // components, not raw bytes. The per-step CD verdict
                        // (cd_groups_compute) consumes an entry only through the
                        // per-event (cand_pass, ign_pick) finite predicates and the
                        // tail (live, prefix_ok-filtered allow, ign_ok), so entries
                        // sharing those are verdict-indistinguishable at EVERY
                        // parser configuration (tests/mask/test_verdict_equivalence
                        // .py). Without lexicons cand_pass == cands, allow == live
                        // and ign_pick/ign_ok are functions of them — the original
                        // (cands, live) key already partitions by equivalence, so
                        // it is kept byte-for-byte. The group REPRESENTATIVE still
                        // carries real segments/remainder bytes (register/audit
                        // payloads recompute the same predicates from them).
                        let mut key: Vec<u8> = Vec::with_capacity(
                            events.len() * (8 * W + 8) + 2 + 16 * W,
                        );
                        if lex_sensitive {
                            for e in events.iter() {
                                for w in e.pass.iter() {
                                    key.extend_from_slice(&w.to_le_bytes());
                                }
                                key.extend_from_slice(&e.ign_pick.to_le_bytes());
                            }
                            if seg.is_empty() {
                                key.push(0); // empty tail: verdict is always true
                            } else {
                                key.push(1);
                                let lv = self.live[cur as usize];
                                let mut allow = [0u64; W];
                                m_find(&lv, |t| {
                                    if self.prefix_ok(t, seg) {
                                        m_set(&mut allow, t);
                                    }
                                    false
                                });
                                for w in lv.iter() {
                                    key.extend_from_slice(&w.to_le_bytes());
                                }
                                for w in allow.iter() {
                                    key.extend_from_slice(&w.to_le_bytes());
                                }
                                key.push(m_and_any(&lv, &self.ignored) as u8);
                            }
                        } else {
                            for e in events.iter() {
                                for w in e.cands.iter() {
                                    key.extend_from_slice(&w.to_le_bytes());
                                }
                            }
                            for w in self.live[cur as usize].iter() {
                                key.extend_from_slice(&w.to_le_bytes());
                            }
                        }
                        let gi = match group_ix.get(&key) {
                            Some(&gi) => gi,
                            None => {
                                let evs: Vec<(Vec<u64>, u32)> = events
                                    .iter()
                                    .map(|e| (e.cands.to_vec(), e.lexeme.len() as u32))
                                    .collect();
                                let segs: Vec<Vec<u8>> =
                                    events.iter().map(|e| e.lexeme.clone()).collect();
                                group_ix.insert(key, groups.len());
                                groups.push((evs, segs, seg.to_vec(), Vec::new()));
                                groups.len() - 1
                            }
                        };
                        match self.aliases.get(&tid) {
                            Some(all) => groups[gi].3.extend_from_slice(all),
                            None => groups[gi].3.push(tid),
                        }
                    }
                    None => {}
                }
            }

            stack.push(Frame {
                end: i + size,
                dfa_state: cur,
                seg_start,
                seg_len,
                last_len,
                last_state,
                events_len: events.len(),
                n_real,
                cd_flag,
            });
            i += 1;
        }
        ci.sort_unstable();
        (ci, groups)
    }

    /// Env-gated dispatch (W8): the sequential `walk_raw` (GRID_WALK_THREADS
    /// unset/0/1 — the kill switch — or a sub-threshold trie or pool failure)
    /// or the rayon `walk_raw_par`. Output is bit-identical either way.
    fn walk_auto(&self, remainder: &[u8], a_mask: &[u64; W]) -> (Vec<u32>, WalkGroupsRaw) {
        let threads = env_usize("GRID_WALK_THREADS", 0);
        if threads >= 2 {
            let par_min = env_usize("GRID_WALK_PAR_MIN", WALK_PAR_MIN_DEFAULT).max(1);
            if self.nodes.len() >= par_min {
                if let Some(pool) = walk_pool(threads) {
                    return self.walk_raw_par(remainder, a_mask, &pool, par_min);
                }
            }
        }
        self.walk_raw(remainder, a_mask)
    }

    /// W8 rayon walk: the DFS splits at TOP-LEVEL trie children. Every
    /// top-level subtree is walked with exactly the state the sequential DFS
    /// reaches it with (the root frame: `seed(remainder)`, empty events,
    /// n_real = 0 — walk_raw pops/truncates back to it at every top-level
    /// boundary, so subtrees are independent given the root frame). Contiguous
    /// subtrees are chunked to bound task overhead, and chunk outputs merge IN
    /// TRIE ORDER: ci concatenated then globally sort_unstable exactly like
    /// walk_raw's final sort; groups re-interned first-encounter-first, so the
    /// group order, each group's representative (globally first encounter) and
    /// each group's tid append order are byte-identical to sequential.
    fn walk_raw_par(
        &self,
        remainder: &[u8],
        a_mask: &[u64; W],
        pool: &rayon::ThreadPool,
        par_min: usize,
    ) -> (Vec<u32>, WalkGroupsRaw) {
        use rayon::prelude::*;
        let n = self.nodes.len();
        let target = (n / (pool.current_num_threads() * 8).max(1))
            .max(par_min / 8)
            .max(1);
        let mut chunks: Vec<(usize, usize)> = Vec::new();
        {
            let mut start = 0usize;
            let mut i = 0usize;
            while i < n {
                i += (self.nodes[i] >> 32) as usize; // top-level sibling hop
                if i - start >= target || i >= n {
                    chunks.push((start, i.min(n)));
                    start = i;
                }
            }
        }
        if chunks.len() <= 1 {
            return self.walk_raw(remainder, a_mask);
        }
        let seed = self.seed(remainder);
        let mut a_or_ign = *a_mask;
        for i in 0..W {
            a_or_ign[i] |= self.ignored[i];
        }
        let results: Vec<(Vec<u32>, Vec<(Vec<u8>, WalkGroupRaw)>)> = pool.install(|| {
            chunks
                .par_iter()
                .map(|&(lo, hi)| self.walk_range(remainder, a_mask, &a_or_ign, seed, lo, hi))
                .collect()
        });
        let mut ci: Vec<u32> = Vec::new();
        let mut group_ix: HashMap<Vec<u8>, usize> = HashMap::new();
        let mut groups: WalkGroupsRaw = Vec::new();
        for (ci_part, keyed) in results {
            ci.extend_from_slice(&ci_part);
            for (key, g) in keyed {
                match group_ix.get(&key) {
                    Some(&gi) => groups[gi].3.extend_from_slice(&g.3),
                    None => {
                        group_ix.insert(key, groups.len());
                        groups.push(g);
                    }
                }
            }
        }
        ci.sort_unstable();
        (ci, groups)
    }

    /// One walk_raw_par task: the walk_raw DFS restricted to the top-level
    /// subtree window [lo, hi) (both are top-level sibling boundaries), seeded
    /// from the shared root frame. The per-node body is a LOCKSTEP TRANSCRIPT
    /// of walk_raw (any change there must land here too — the differential
    /// tests in tests/trie/test_walk_threads.py bind them); the only deltas:
    /// ci is unsorted (caller sorts globally) and groups carry their interning
    /// KEY out so the caller can merge chunks into walk_raw's global
    /// first-encounter group order.
    #[allow(clippy::type_complexity)]
    fn walk_range(
        &self,
        remainder: &[u8],
        a_mask: &[u64; W],
        a_or_ign: &[u64; W],
        seed: (i32, usize, i32),
        lo: usize,
        hi: usize,
    ) -> (Vec<u32>, Vec<(Vec<u8>, WalkGroupRaw)>) {
        let mut ci: Vec<u32> = Vec::new();
        let lex_sensitive = self.lex_allowed.is_some();
        let mut group_ix: HashMap<Vec<u8>, usize> = HashMap::new();
        let mut groups: Vec<(Vec<u8>, WalkGroupRaw)> = Vec::new();

        let mut events: Vec<Ev<W>> = Vec::new();
        let mut path: Vec<u8> = Vec::with_capacity(remainder.len() + 64);
        path.extend_from_slice(remainder);
        let (s_state, s_len, s_last) = seed;
        let mut stack: Vec<Frame> = vec![Frame {
            end: hi + 1,
            dfa_state: s_state,
            seg_start: 0,
            seg_len: remainder.len(),
            last_len: s_len,
            last_state: s_last,
            events_len: 0,
            n_real: 0,
            cd_flag: false,
        }];

        let mut i = lo;
        while i < hi {
            while stack.len() > 1 && i >= stack.last().unwrap().end {
                stack.pop();
            }
            {
                let f = stack.last().unwrap();
                events.truncate(f.events_len);
                path.truncate(f.seg_start + f.seg_len);
            }
            let word = self.nodes[i];
            let byte = (word & 0xFF) as u8;
            let tid_raw = ((word >> 8) & 0xFF_FFFF) as i64 - 1;
            let size = (word >> 32) as usize;

            let parent = stack.last().unwrap();
            let mut cur = parent.dfa_state;
            let mut seg_start = parent.seg_start;
            let mut seg_len = parent.seg_len;
            let base = seg_start + seg_len;
            debug_assert_eq!(base, path.len());
            let mut last_len = parent.last_len;
            let mut last_state = parent.last_state;
            let mut n_real = parent.n_real;
            let mut cd_flag = parent.cd_flag;
            let mut reject = false;
            let events_base = events.len();

            let mut pending: Vec<u8> = vec![byte];
            let mut idx = 0usize;
            while idx < pending.len() {
                let b = pending[idx];
                idx += 1;
                let nx = self.tr(cur, b);
                if nx != DEAD {
                    path.push(b);
                    seg_len += 1;
                    cur = nx;
                    if self.accept[nx as usize] != -1 {
                        last_len = seg_len;
                        last_state = nx;
                    }
                    continue;
                }
                if last_state == -1 {
                    reject = true;
                    break;
                }
                let cands = self.accepts_all[last_state as usize];
                let lexeme: Vec<u8> = path[seg_start..seg_start + last_len].to_vec();
                if n_real == 0 {
                    match self.pick_viable(&cands, &lexeme, a_mask) {
                        None => {
                            reject = true;
                            break;
                        }
                        Some(t) => {
                            if !m_bit(&self.ignored, t) {
                                n_real = 1;
                            }
                        }
                    }
                } else {
                    let pure_ignored =
                        m_and_any(&cands, &self.ignored) && !m_any(&m_and_not(&cands, &self.ignored));
                    if !pure_ignored {
                        cd_flag = true;
                    }
                }
                let (pass, ign_pick) = if lex_sensitive {
                    let mut p = [0u64; W];
                    m_find(&cands, |t| {
                        if self.lexeme_ok(t, &lexeme) {
                            m_set(&mut p, t);
                        }
                        false
                    });
                    (p, self.pick_first(&m_and(&cands, &self.ignored)))
                } else {
                    ([0u64; W], -1)
                };
                events.push(Ev { cands, lexeme, pass, ign_pick });
                let rest_len = seg_len - last_len;
                let mut tail: Vec<u8> = Vec::with_capacity(rest_len + 1 + pending.len() - idx);
                tail.extend_from_slice(&path[seg_start + last_len..seg_start + seg_len]);
                tail.push(b);
                tail.extend_from_slice(&pending[idx..]);
                pending = tail;
                idx = 0;
                cur = self.dfa_start;
                path.truncate(base);
                seg_start = base;
                seg_len = 0;
                last_len = 0;
                last_state = -1;
            }

            if reject {
                events.truncate(events_base);
                i += size;
                continue;
            }

            let seg = &path[seg_start..seg_start + seg_len];
            let verdict_ci: Option<bool> = if n_real == 0 {
                if !seg.is_empty() && !self.partial_viable(seg, cur, a_or_ign) {
                    events.truncate(events_base);
                    i += size;
                    continue;
                }
                Some(true)
            } else if cd_flag || n_real >= 2 {
                Some(false)
            } else if seg.is_empty() {
                Some(true)
            } else if self.partial_viable(seg, cur, &self.ignored) {
                Some(true)
            } else {
                Some(false)
            };

            if tid_raw >= 0 {
                let tid = tid_raw as u32;
                match verdict_ci {
                    Some(true) => match self.aliases.get(&tid) {
                        Some(all) => ci.extend_from_slice(all),
                        None => ci.push(tid),
                    },
                    Some(false) => {
                        let mut key: Vec<u8> = Vec::with_capacity(
                            events.len() * (8 * W + 8) + 2 + 16 * W,
                        );
                        if lex_sensitive {
                            for e in events.iter() {
                                for w in e.pass.iter() {
                                    key.extend_from_slice(&w.to_le_bytes());
                                }
                                key.extend_from_slice(&e.ign_pick.to_le_bytes());
                            }
                            if seg.is_empty() {
                                key.push(0);
                            } else {
                                key.push(1);
                                let lv = self.live[cur as usize];
                                let mut allow = [0u64; W];
                                m_find(&lv, |t| {
                                    if self.prefix_ok(t, seg) {
                                        m_set(&mut allow, t);
                                    }
                                    false
                                });
                                for w in lv.iter() {
                                    key.extend_from_slice(&w.to_le_bytes());
                                }
                                for w in allow.iter() {
                                    key.extend_from_slice(&w.to_le_bytes());
                                }
                                key.push(m_and_any(&lv, &self.ignored) as u8);
                            }
                        } else {
                            for e in events.iter() {
                                for w in e.cands.iter() {
                                    key.extend_from_slice(&w.to_le_bytes());
                                }
                            }
                            for w in self.live[cur as usize].iter() {
                                key.extend_from_slice(&w.to_le_bytes());
                            }
                        }
                        let gi = match group_ix.get(&key) {
                            Some(&gi) => gi,
                            None => {
                                let evs: Vec<(Vec<u64>, u32)> = events
                                    .iter()
                                    .map(|e| (e.cands.to_vec(), e.lexeme.len() as u32))
                                    .collect();
                                let segs: Vec<Vec<u8>> =
                                    events.iter().map(|e| e.lexeme.clone()).collect();
                                group_ix.insert(key.clone(), groups.len());
                                groups.push((key, (evs, segs, seg.to_vec(), Vec::new())));
                                groups.len() - 1
                            }
                        };
                        let ids = &mut groups[gi].1 .3;
                        match self.aliases.get(&tid) {
                            Some(all) => ids.extend_from_slice(all),
                            None => ids.push(tid),
                        }
                    }
                    None => {}
                }
            }

            stack.push(Frame {
                end: i + size,
                dfa_state: cur,
                seg_start,
                seg_len,
                last_len,
                last_state,
                events_len: events.len(),
                n_real,
                cd_flag,
            });
            i += 1;
        }
        (ci, groups)
    }
}

fn wrap_groups(py: Python<'_>, raw: WalkGroupsRaw) -> WalkGroups {
    raw.into_iter()
        .map(|(evs, segs, rem, ids)| {
            let py_segs: Vec<Py<PyBytes>> =
                segs.iter().map(|s| PyBytes::new(py, s).unbind()).collect();
            (evs, py_segs, PyBytes::new(py, &rem).unbind(), ids)
        })
        .collect()
}

// ------------------------------------------------- v7 encode + blob format
//
// Blob v1 (LE): [u8 ver=1][u32 W][u32 n_groups], per group [u32 n_events]
// {[u64 x W cands][u32 lex_len][lexeme]} [u32 rem_len][rem][u32 n_ids]
// [i32-le x n ids] — exactly WalkGroupsRaw, order-preserving (group order is
// part of the order-exact parity contract). Unknown version = hard error.
// The per-event u32 length in WalkGroupsRaw equals the lexeme byte length,
// so it is not stored: decode reconstructs it as lexeme.len().

const BLOB_V1: u8 = 1;
const TAG_ACCEPT: u8 = 0;
const TAG_REJECT: u8 = 1;
const TAG_BITSET: u8 = 2;

fn blob_encode(w: usize, groups: &WalkGroupsRaw) -> Vec<u8> {
    let mut b: Vec<u8> = Vec::with_capacity(64 + groups.len() * (32 + 16 * w));
    b.push(BLOB_V1);
    b.extend_from_slice(&(w as u32).to_le_bytes());
    b.extend_from_slice(&(groups.len() as u32).to_le_bytes());
    for (evs, segs, rem, ids) in groups {
        b.extend_from_slice(&(evs.len() as u32).to_le_bytes());
        for (i, (cands, _len)) in evs.iter().enumerate() {
            for word in cands.iter().take(w) {
                b.extend_from_slice(&word.to_le_bytes());
            }
            let lx = &segs[i];
            b.extend_from_slice(&(lx.len() as u32).to_le_bytes());
            b.extend_from_slice(lx);
        }
        b.extend_from_slice(&(rem.len() as u32).to_le_bytes());
        b.extend_from_slice(rem);
        b.extend_from_slice(&(ids.len() as u32).to_le_bytes());
        for t in ids {
            b.extend_from_slice(&(*t as i32).to_le_bytes());
        }
    }
    b
}

struct BlobReader<'a> {
    b: &'a [u8],
    off: usize,
}

impl<'a> BlobReader<'a> {
    fn take(&mut self, n: usize) -> PyResult<&'a [u8]> {
        if self.off + n > self.b.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("truncated v7 blob"));
        }
        let s = &self.b[self.off..self.off + n];
        self.off += n;
        Ok(s)
    }

    fn u32(&mut self) -> PyResult<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

fn blob_decode(blob: &[u8], expect_w: usize) -> PyResult<WalkGroupsRaw> {
    if blob.first() != Some(&BLOB_V1) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown v7 blob version: {:?}", blob.first()
        )));
    }
    let mut r = BlobReader { b: blob, off: 1 };
    let w = r.u32()? as usize;
    if w != expect_w {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "v7 blob width {w} != kernel width {expect_w}"
        )));
    }
    let n_groups = r.u32()? as usize;
    let mut groups: WalkGroupsRaw = Vec::with_capacity(n_groups);
    for _ in 0..n_groups {
        let n_events = r.u32()? as usize;
        let mut evs: Vec<(Vec<u64>, u32)> = Vec::with_capacity(n_events);
        let mut segs: Vec<Vec<u8>> = Vec::with_capacity(n_events);
        for _ in 0..n_events {
            let mut cands: Vec<u64> = Vec::with_capacity(w);
            for _ in 0..w {
                cands.push(u64::from_le_bytes(r.take(8)?.try_into().unwrap()));
            }
            let lx_len = r.u32()? as usize;
            let lx = r.take(lx_len)?.to_vec();
            evs.push((cands, lx_len as u32));
            segs.push(lx);
        }
        let rem_len = r.u32()? as usize;
        let rem = r.take(rem_len)?.to_vec();
        let n_ids = r.u32()? as usize;
        let mut ids: Vec<u32> = Vec::with_capacity(n_ids);
        for _ in 0..n_ids {
            ids.push(i32::from_le_bytes(r.take(4)?.try_into().unwrap()) as u32);
        }
        groups.push((evs, segs, rem, ids));
    }
    if r.off != blob.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("v7 blob trailing bytes"));
    }
    Ok(groups)
}

/// ci ids from ONE i32-le buffer (the walk/register wire format).
fn ids_from_le(ci: &[u8]) -> PyResult<Vec<i32>> {
    if ci.len() % 4 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "ci byte buffer length must be a multiple of 4 (i32-le ids)",
        ));
    }
    Ok(ci.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// grid/mask/cache.py adaptive_encode, transcribed byte-for-byte: payload
/// sizes compared as signed (4n, ACCEPT) / (4(V-n), REJECT) / (ceil(V/8),
/// BITSET) tuples with the tag as the tie-break (ACCEPT < REJECT < BITSET);
/// n counts DUPLICATES; ACCEPT ids ascending as u32-le preserving duplicates
/// (np.sort semantics); REJECT the ascending complement; BITSET bit t ->
/// byte t>>3, bit t&7, ceil(V/8) zero-padded bytes. Ids must lie in
/// [0, vocab) — the walk guarantees it; out-of-range is a hard error (the
/// Python reference would raise from numpy fancy indexing on the REJECT and
/// BITSET paths).
fn adaptive_encode_ids(ids: &[i32], vocab: u64) -> PyResult<(u8, Vec<u8>)> {
    let v = vocab as i64;
    for t in ids {
        if (*t as i64) < 0 || (*t as i64) >= v {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "ci id {t} outside vocab {vocab}"
            )));
        }
    }
    let n = ids.len() as i64;
    let size_accept = 4 * n;
    let size_reject = 4 * (v - n);
    let size_bitset = (v + 7) / 8;
    let mut tag = TAG_ACCEPT;
    let mut best = size_accept;
    if size_reject < best {
        best = size_reject;
        tag = TAG_REJECT;
    }
    if size_bitset < best {
        tag = TAG_BITSET;
    }
    let payload: Vec<u8> = match tag {
        TAG_ACCEPT => {
            let mut sorted: Vec<i32> = ids.to_vec();
            if sorted.windows(2).any(|p| p[0] > p[1]) {
                sorted.sort_unstable(); // defensive: kernel ci arrives sorted
            }
            let mut p = Vec::with_capacity(sorted.len() * 4);
            for t in &sorted {
                p.extend_from_slice(&(*t as u32).to_le_bytes());
            }
            p
        }
        TAG_REJECT => {
            let mut keep = vec![false; v as usize];
            for t in ids {
                keep[*t as usize] = true;
            }
            let mut p = Vec::with_capacity(((v - n).max(0) as usize) * 4);
            for (t, k) in keep.iter().enumerate() {
                if !k {
                    p.extend_from_slice(&(t as u32).to_le_bytes());
                }
            }
            p
        }
        _ => {
            let mut bits = vec![0u8; size_bitset as usize];
            for t in ids {
                bits[(*t as usize) >> 3] |= 1 << (*t & 7);
            }
            bits
        }
    };
    Ok((tag, payload))
}

/// entry_id = BLAKE2b(digest_size=16)(key_repr || [tag] || payload) — the
/// hashlib.blake2b construction (digest_length in the parameter block).
fn entry_id_128(key_repr: &[u8], tag: u8, payload: &[u8]) -> [u8; 16] {
    let mut h = Blake2b128::new();
    h.update(key_repr);
    h.update([tag]);
    h.update(payload);
    h.finalize().into()
}

fn hex32(id: &[u8; 16]) -> String {
    let mut s = String::with_capacity(32);
    for b in id {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// ci ids -> packed u32 bit words (the fill_bits prefix); sized to the
/// highest id, exactly like register_impl always did.
fn ci_bits_of(ids: &[i32]) -> Vec<u32> {
    let words = ids.iter().map(|t| (*t as usize >> 5) + 1).max().unwrap_or(0);
    let mut bits = vec![0u32; words];
    for t in ids {
        bits[*t as usize >> 5] |= 1 << (t & 31);
    }
    bits
}

/// Module-level test surface: (tag, payload) for ci ids (ONE i32-le buffer)
/// at vocab_size — cross-implementation vectors against the Python
/// adaptive_encode (tests/mask/test_v7_encode.py).
#[pyfunction]
fn encode_mask(py: Python<'_>, ci: &Bound<'_, PyBytes>, vocab_size: u64) -> PyResult<(u8, Py<PyBytes>)> {
    let ids = ids_from_le(ci.as_bytes())?;
    let (tag, payload) = adaptive_encode_ids(&ids, vocab_size)?;
    Ok((tag, PyBytes::new(py, &payload).unbind()))
}

/// Module-level test surface: (entry_id hex, tag) for (key_repr, ci, vocab)
/// — exactly the register_blob hash, for cross-impl blake2b vectors.
#[pyfunction]
fn entry_id_hex(key_repr: &Bound<'_, PyBytes>, ci: &Bound<'_, PyBytes>, vocab_size: u64) -> PyResult<(String, u8)> {
    let ids = ids_from_le(ci.as_bytes())?;
    let (tag, payload) = adaptive_encode_ids(&ids, vocab_size)?;
    Ok((hex32(&entry_id_128(key_repr.as_bytes(), tag, &payload)), tag))
}

fn parse_lexicon(
    lexicon: Option<&Bound<'_, PyDict>>,
) -> PyResult<(Option<HashMap<u32, HashSet<Vec<u8>>>>, Option<HashMap<u32, HashSet<Vec<u8>>>>)> {
    match lexicon {
        None => Ok((None, None)),
        Some(d) => {
            let mut allowed: HashMap<u32, HashSet<Vec<u8>>> = HashMap::new();
            let mut prefixes: HashMap<u32, HashSet<Vec<u8>>> = HashMap::new();
            for (k, v) in d.iter() {
                let tid: u32 = k.extract()?;
                let words: Vec<Vec<u8>> = v.extract()?;
                let mut aset = HashSet::new();
                let mut pset = HashSet::new();
                for w in words {
                    for j in 0..=w.len() {
                        pset.insert(w[..j].to_vec());
                    }
                    aset.insert(w);
                }
                allowed.insert(tid, aset);
                prefixes.insert(tid, pset);
            }
            Ok((Some(allowed), Some(prefixes)))
        }
    }
}

enum WalkerAny {
    W1(RustWalkerImpl<1>),
    W2(RustWalkerImpl<2>),
    W4(RustWalkerImpl<4>),
    W8(RustWalkerImpl<8>),
}

macro_rules! walker_dispatch {
    ($self:expr, $w:ident, $body:expr) => {
        match &$self.inner {
            WalkerAny::W1($w) => $body,
            WalkerAny::W2($w) => $body,
            WalkerAny::W4($w) => $body,
            WalkerAny::W8($w) => $body,
        }
    };
}

#[pyclass]
struct RustWalker {
    inner: WalkerAny,
    width: usize,
}

fn build_walker<const W: usize>(
    node_words: Vec<u64>,
    trans_v: Vec<i32>,
    accept_v: Vec<i32>,
    accepts_all: Vec<Vec<u64>>,
    live: Vec<Vec<u64>>,
    dfa_start: i32,
    ignored: Vec<u64>,
    literal: Vec<u64>,
    lex: (Option<HashMap<u32, HashSet<Vec<u8>>>>, Option<HashMap<u32, HashSet<Vec<u8>>>>),
    aliases: HashMap<u32, Vec<u32>>,
) -> RustWalkerImpl<W> {
    RustWalkerImpl {
        nodes: node_words,
        trans: trans_v,
        accept: accept_v,
        accepts_all: accepts_all.iter().map(|v| m_from_words::<W>(v)).collect(),
        live: live.iter().map(|v| m_from_words::<W>(v)).collect(),
        ignored: m_from_words::<W>(&ignored),
        literal: m_from_words::<W>(&literal),
        dfa_start,
        lex_allowed: lex.0,
        lex_prefixes: lex.1,
        aliases,
    }
}

#[pymethods]
impl RustWalker {
    #[new]
    #[pyo3(signature = (nodes, trans, accept, n_terminals, accepts_all, live, dfa_start,
                        ignored_mask, literal_mask, lexicon=None, aliases=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        nodes: &Bound<'_, PyBytes>,
        trans: &Bound<'_, PyBytes>,
        accept: &Bound<'_, PyBytes>,
        n_terminals: usize,
        accepts_all: Vec<Vec<u64>>,
        live: Vec<Vec<u64>>,
        dfa_start: i32,
        ignored_mask: Vec<u64>,
        literal_mask: Vec<u64>,
        lexicon: Option<&Bound<'_, PyDict>>,
        aliases: Option<HashMap<u32, Vec<u32>>>,
    ) -> PyResult<Self> {
        let nb = nodes.as_bytes();
        let mut node_words = Vec::with_capacity(nb.len() / 8);
        for c in nb.chunks_exact(8) {
            node_words.push(u64::from_le_bytes(c.try_into().unwrap()));
        }
        let tb = trans.as_bytes();
        let mut trans_v = Vec::with_capacity(tb.len() / 4);
        for c in tb.chunks_exact(4) {
            trans_v.push(i32::from_le_bytes(c.try_into().unwrap()));
        }
        let ab = accept.as_bytes();
        let mut accept_v = Vec::with_capacity(ab.len() / 4);
        for c in ab.chunks_exact(4) {
            accept_v.push(i32::from_le_bytes(c.try_into().unwrap()));
        }
        let lex = parse_lexicon(lexicon)?;
        let aliases = aliases.unwrap_or_default();
        let width = width_for(n_terminals).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "{n_terminals} terminals exceeds the 512-terminal kernel bound"
            ))
        })?;
        let inner = match width {
            1 => WalkerAny::W1(build_walker::<1>(node_words, trans_v, accept_v, accepts_all, live, dfa_start, ignored_mask, literal_mask, lex, aliases)),
            2 => WalkerAny::W2(build_walker::<2>(node_words, trans_v, accept_v, accepts_all, live, dfa_start, ignored_mask, literal_mask, lex, aliases)),
            4 => WalkerAny::W4(build_walker::<4>(node_words, trans_v, accept_v, accepts_all, live, dfa_start, ignored_mask, literal_mask, lex, aliases)),
            _ => WalkerAny::W8(build_walker::<8>(node_words, trans_v, accept_v, accepts_all, live, dfa_start, ignored_mask, literal_mask, lex, aliases)),
        };
        Ok(RustWalker { inner, width })
    }

    #[getter]
    fn width(&self) -> usize {
        self.width
    }

    /// -> (ci, groups): group event masks are little-endian u64 word lists.
    /// ci token ids cross the FFI as ONE i32-le PyBytes buffer (sorted,
    /// alias-expanded) — the Python side consumes it via np.frombuffer with
    /// zero int-object materialization; per-int extraction of the 150k-id
    /// open-literal giants was the last O(V) Python cost on the cold path.
    /// The walk itself runs with the GIL released (SS6 overlap contract):
    /// ms-scale cold walks scheduled on a worker thread no longer stall the
    /// scheduler thread's Python work. GRID_WALK_THREADS >= 2 additionally
    /// parallelizes the walk itself on the dedicated rayon pool (walk_auto;
    /// still inside detach), bit-identical to the sequential walk.
    #[pyo3(signature = (remainder, a_mask))]
    #[allow(clippy::type_complexity)]
    fn walk(
        &self,
        py: Python<'_>,
        remainder: Vec<u8>,
        a_mask: Vec<u64>,
    ) -> PyResult<(Py<PyBytes>, WalkGroups)> {
        Ok(walker_dispatch!(self, w, {
            fn go<const W: usize>(
                w: &RustWalkerImpl<W>,
                py: Python<'_>,
                remainder: Vec<u8>,
                a_mask: &[u64],
            ) -> (Py<PyBytes>, WalkGroups) {
                let mask = m_from_words::<W>(a_mask);
                let (ci, raw) = py.detach(move || {
                    let (ci, raw) = w.walk_auto(&remainder, &mask);
                    // i32-le serialization inside detach: ids are < 2^24
                    // (24-bit trie tid field), so u32 -> i32 is lossless
                    let mut ci_bytes: Vec<u8> = Vec::with_capacity(ci.len() * 4);
                    for t in &ci {
                        ci_bytes.extend_from_slice(&(*t as i32).to_le_bytes());
                    }
                    (ci_bytes, raw)
                });
                (PyBytes::new(py, &ci).unbind(), wrap_groups(py, raw))
            }
            go(w, py, remainder, &a_mask)
        }))
    }

    /// Kernel v7 walk: -> (ci as ONE i32-le PyBytes, blob v1 PyBytes). Same
    /// walk_auto as walk() (rayon-capable, GIL released), but the group
    /// output serializes to the opaque blob INSIDE the detach — no Python
    /// tuple/bytes-per-group materialization. The blob feeds
    /// RustVerdicts::register_blob (registration and cross-producer import).
    #[pyo3(signature = (remainder, a_mask))]
    fn walk_payload(
        &self,
        py: Python<'_>,
        remainder: Vec<u8>,
        a_mask: Vec<u64>,
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>)> {
        Ok(walker_dispatch!(self, w, {
            fn go<const W: usize>(
                w: &RustWalkerImpl<W>,
                py: Python<'_>,
                remainder: Vec<u8>,
                a_mask: &[u64],
            ) -> (Py<PyBytes>, Py<PyBytes>) {
                let mask = m_from_words::<W>(a_mask);
                let (ci_bytes, blob) = py.detach(move || {
                    let (ci, raw) = w.walk_auto(&remainder, &mask);
                    let mut ci_bytes: Vec<u8> = Vec::with_capacity(ci.len() * 4);
                    for t in &ci {
                        ci_bytes.extend_from_slice(&(*t as i32).to_le_bytes());
                    }
                    (ci_bytes, blob_encode(W, &raw))
                });
                (PyBytes::new(py, &ci_bytes).unbind(), PyBytes::new(py, &blob).unbind())
            }
            go(w, py, remainder, &a_mask)
        }))
    }
}

// ---------------------------------------------------------------------------
// RustVerdicts: SS2 kernel #2 (check_context_dependent) + the LALR virtual-stack
// simulate behind it. Exact transcription of grid/mask/producer.py and
// grid/lalr/stack.py; see kernel v2 notes. Masks are [u64; W] like the walker.
// ---------------------------------------------------------------------------

const ACT_NONE: u8 = 0; // "t not in action row"
const ACT_SHIFT: u8 = 1; // grid SHIFT(0) + 1
const ACT_REDUCE: u8 = 2; // grid REDUCE(1) + 1
const ACT_ACCEPT: u8 = 3; // grid ACCEPT(2) + 1

// ---------------------------------------------------------------------------
// Kernel v6 sessions: per-request accept/fill state living on RustVerdicts
// (shared arena/memos/entries). Status encoding mirrors grid/guide.py.
// ---------------------------------------------------------------------------

const ST_ACTIVE: u8 = 0;
const ST_ACCEPTING: u8 = 1;
const ST_GRAMMAR_END: u8 = 2;
const ST_COMPLETE: u8 = 3;

const FLAG_OK: u32 = 1; // bit0: token consumed (0 == REJECTED, no state mutation)
const FLAG_COMPLETE: u32 = 2; // bit1: session is COMPLETE after this token
const FLAG_UNBOUND: u32 = 4; // bit2: successor (kidx, remainder) has no fill binding

/// Rollback delta for ONE accepted token (including eos-accepts, which did not
/// bump n_generated — the log restores the exact pre-accept state).
struct SessLog {
    popped: Vec<u32>,   // original chain suffix removed by this accept (bottom->top)
    pushed: u32,        // frames this accept pushed above the common prefix
    remainder: Vec<u8>, // pre-accept remainder
    status: u8,
    prev_token: i64,
    n_generated: u32,
}

/// Per-session state. The session OWNS its authoritative LALR state chain
/// (root->top) with a gen-tagged kidx cache — the StackNode.kidx/kgen design
/// in Rust — so `reset_interning` is safe with live sessions (risk (c)):
/// no durable kidx exists anywhere, re-interning is lazy via intern_chain.
struct Session {
    chain: Vec<u32>,
    kidx: i64, // cached intern index; valid iff kgen == Memos::gen
    kgen: u64,
    remainder: Vec<u8>,
    status: u8,
    n_generated: u32,
    prev_token: i64, // -1: none
    eos_id: u32,
    log: Vec<SessLog>,
}

/// Scratch accept state: `step_core` mutates it only on success (atomicity —
/// candidate built fully, committed only if viable). Shared by accept
/// (session copy -> commit) and validate (throwaway scratch, no commit).
struct SessCore {
    chain: Vec<u32>,
    kidx: u32,
    remainder: Vec<u8>,
    status: u8,
    eos_id: u32,
}

#[derive(Default)]
struct SessCounters {
    fills_hit: u64,
    fills_miss: u64,
    accepts: u64,
    rejects: u64,
    binds: u64,
}

struct SessTable {
    next: u64,
    map: HashMap<u64, Session>,
    c: SessCounters,
}

impl Default for SessTable {
    fn default() -> Self {
        SessTable { next: 1, map: HashMap::new(), c: SessCounters::default() }
    }
}

enum Tail<const W: usize> {
    Empty,
    Dead,
    Live { ign_ok: bool, allow: [u64; W] },
}

struct VEvent<const W: usize> {
    cand_pass: [u64; W], // candidates passing lexeme_ok(t, segment)
    ign_pick: i64,       // min-priority candidate in ignored (-1: none)
}

struct VGroup<const W: usize> {
    events: Vec<VEvent<W>>,
    tail: Tail<W>,
    token_bytes: Vec<u8>, // token ids as i32-le, ready to memcpy into the output
}

/// Persistent interned-stack state (kernel v4). Nodes are deduplicated by
/// `(parent kidx, LALR state)` — parse behavior is a function of the state
/// chain alone (action/goto tables key on states), so structurally-equal
/// chains share one kidx and all memos hit across token positions.
struct Memos<const W: usize> {
    arena: Vec<(i64, u32)>,             // kidx -> (parent kidx | -1, state)
    intern: HashMap<(i64, u32), u32>,   // (parent, state) -> kidx
    allowed: Vec<Option<[u64; W]>>,     // kidx -> allowed-terminal mask
    eos: Vec<i8>,                       // kidx -> -1 unknown / 0 / 1
    shift: HashMap<(u32, u32), i64>,    // (kidx, terminal) -> kidx | -1 not viable
    cd: HashMap<(u32, u32), Vec<u8>>,   // (handle, kidx) -> passing ids (i32-le)
    row: HashMap<(u32, u32), Vec<u32>>, // (handle, kidx) -> packed fill_bits row, NO eos bit
    gen: u64,                           // arena generation; bumped by reset_interning
    bind: HashMap<(u32, Vec<u8>), u32>, // (kidx, remainder) -> entry handle (v6 warm fill)
    stat: HashMap<(u32, Vec<u8>), u8>,  // (kidx, remainder) -> derived status (never COMPLETE)
}

impl<const W: usize> Default for Memos<W> {
    fn default() -> Self {
        Memos {
            arena: Vec::new(),
            intern: HashMap::new(),
            allowed: Vec::new(),
            eos: Vec::new(),
            shift: HashMap::new(),
            cd: HashMap::new(),
            row: HashMap::new(),
            gen: 0,
            bind: HashMap::new(),
            stat: HashMap::new(),
        }
    }
}

/// The registered-entry store (kernel v7: behind an RwLock so registration —
/// including register_blob's detached build — is `&self` and never conflicts
/// with concurrent verdict/session borrows). `by_id` deduplicates
/// register_blob under pool races (registration is content-addressed by
/// entry id; register/register_bytes have no id and keep the Python-side
/// single-flight, as today).
#[derive(Default)]
struct EntryStore<const W: usize> {
    entries: Vec<Vec<VGroup<W>>>,
    entry_ci: Vec<Vec<u8>>, // handle -> ci token ids as i32-le (hit_pass prefix)
    entry_ci_bits: Vec<Vec<u32>>, // handle -> ci ids packed as bit words (fill_bits prefix)
    by_id: HashMap<[u8; 16], u32>, // entry_id -> handle (register_blob dedup)
}

/// v6 session tables (uploaded via set_dfa_accept / set_token_bytes — `&self`
/// with interior mutability since v7: an upload racing a detached
/// register_blob must not need a mutable pyclass borrow).
struct SessTabs<const W: usize> {
    dfa_accept: Vec<i32>,           // per DFA state: winning terminal or -1
    dfa_accepts_all: Vec<[u64; W]>, // per DFA state: full candidate set
    tok_blob: Vec<u8>,              // adapter token_bytes, concatenated
    tok_off: Vec<i32>,              // len = n_tokens + 1 (blob offsets)
}

impl<const W: usize> Default for SessTabs<W> {
    fn default() -> Self {
        SessTabs {
            dfa_accept: Vec::new(),
            dfa_accepts_all: Vec::new(),
            tok_blob: Vec::new(),
            tok_off: Vec::new(),
        }
    }
}

struct RustVerdictsImpl<const W: usize> {
    action_kind: Vec<u8>,
    action_arg: Vec<u32>,
    n_cols: usize, // n_terminals incl. END; also the nonterminal id base
    goto_tbl: Vec<i32>,
    n_nts: usize,
    prods: Vec<(u32, u32)>,
    end_id: u32,
    ignored: [u64; W],
    literal: [u64; W],
    trans: Vec<i32>,
    live: Vec<[u64; W]>,
    dfa_start: i32,
    lex_allowed: Option<HashMap<u32, HashSet<Vec<u8>>>>,
    lex_prefixes: Option<HashMap<u32, HashSet<Vec<u8>>>>,
    // LOCK ORDER (deadlock discipline): sessions -> mem -> store, with
    // store.write() taken with NOTHING else held (registration paths) and
    // store.read guards never held across a call that re-acquires store.
    // tabs is ordering-independent: its writers (set_token_bytes /
    // set_dfa_accept) hold no other lock, so no cycle can involve it.
    store: RwLock<EntryStore<W>>,
    mem: Mutex<Memos<W>>, // interior mutability: verdict methods take &self
    tabs: RwLock<SessTabs<W>>,
    sessions: Mutex<SessTable>, // lock order: sessions THEN mem, never reversed
}

impl<const W: usize> RustVerdictsImpl<W> {
    #[inline]
    fn act(&self, state: u32, t: u32) -> (u8, u32) {
        let ix = state as usize * self.n_cols + t as usize;
        (self.action_kind[ix], self.action_arg[ix])
    }

    #[inline]
    fn goto_of(&self, state: u32, lhs: u32) -> i32 {
        self.goto_tbl[state as usize * self.n_nts + (lhs as usize - self.n_cols)]
    }

    #[inline]
    fn tr(&self, state: i32, byte: u8) -> i32 {
        self.trans[(state as usize) * 256 + byte as usize]
    }

    fn scan_state(&self, bytes: &[u8]) -> i32 {
        let mut st = self.dfa_start;
        for &b in bytes {
            st = self.tr(st, b);
            if st == DEAD {
                return DEAD;
            }
        }
        st
    }

    fn lexeme_ok(&self, t: u32, lexeme: &[u8]) -> bool {
        match &self.lex_allowed {
            None => true,
            Some(m) => match m.get(&t) {
                None => true,
                Some(set) => set.contains(lexeme),
            },
        }
    }

    fn prefix_ok(&self, t: u32, partial: &[u8]) -> bool {
        match &self.lex_prefixes {
            None => true,
            Some(m) => match m.get(&t) {
                None => true,
                Some(set) => set.contains(partial),
            },
        }
    }

    /// First of `pool` by producer priority (literals ascending, then named).
    #[inline]
    fn pick_first(&self, pool: &[u64; W]) -> i64 {
        if let Some(t) = m_first(&m_and(pool, &self.literal)) {
            return t as i64;
        }
        if let Some(t) = m_first(&m_and_not(pool, &self.literal)) {
            return t as i64;
        }
        -1
    }

    /// stack.py::simulate on an arena chain.
    fn simulate_arena(&self, arena: &[(i64, u32)], base_ix0: usize, t: u32) -> bool {
        let mut base_ix = base_ix0;
        let mut overlay: Vec<u32> = Vec::new();
        for _ in 0..10_000 {
            let top = *overlay.last().unwrap_or(&arena[base_ix].1);
            let (kind, arg) = self.act(top, t);
            match kind {
                ACT_SHIFT | ACT_ACCEPT => return true,
                ACT_REDUCE => {
                    let (lhs, rhs_len) = self.prods[arg as usize];
                    let mut k = rhs_len;
                    while k > 0 && !overlay.is_empty() {
                        overlay.pop();
                        k -= 1;
                    }
                    while k > 0 {
                        let parent = arena[base_ix].0;
                        if parent < 0 {
                            return false; // pop past root
                        }
                        base_ix = parent as usize;
                        k -= 1;
                    }
                    let below = *overlay.last().unwrap_or(&arena[base_ix].1);
                    let nxt = self.goto_of(below, lhs);
                    if nxt < 0 {
                        return false;
                    }
                    overlay.push(nxt as u32);
                }
                _ => return false, // ACT_NONE
            }
        }
        debug_assert!(false, "reduce chain did not terminate");
        false
    }

    // ---------------------------------------------------- interned arena (v4)

    fn intern_node(&self, mem: &mut Memos<W>, parent: i64, state: u32) -> u32 {
        if let Some(&ix) = mem.intern.get(&(parent, state)) {
            return ix;
        }
        let ix = mem.arena.len() as u32;
        mem.arena.push((parent, state));
        mem.allowed.push(None);
        mem.eos.push(-1);
        mem.intern.insert((parent, state), ix);
        ix
    }

    fn intern_chain(&self, mem: &mut Memos<W>, chain: &[u32]) -> u32 {
        let mut cur: i64 = -1;
        for &s in chain {
            cur = self.intern_node(mem, cur, s) as i64;
        }
        cur as u32 // caller guarantees non-empty
    }

    /// stack.py::allowed_terminals as a mask, memoized per interned node.
    fn allowed_at(&self, mem: &mut Memos<W>, kidx: u32) -> [u64; W] {
        if let Some(a) = mem.allowed[kidx as usize] {
            return a;
        }
        let top = mem.arena[kidx as usize].1;
        let mut mask = [0u64; W];
        for t in 0..self.n_cols as u32 {
            if t == self.end_id || self.act(top, t).0 == ACT_NONE {
                continue;
            }
            if self.simulate_arena(&mem.arena, kidx as usize, t) {
                m_set(&mut mask, t);
            }
        }
        mem.allowed[kidx as usize] = Some(mask);
        mask
    }

    /// stack.py::eos_ok_stack, memoized per interned node.
    fn eos_at(&self, mem: &mut Memos<W>, kidx: u32) -> bool {
        let c = mem.eos[kidx as usize];
        if c >= 0 {
            return c == 1;
        }
        let ok = self.simulate_arena(&mem.arena, kidx as usize, self.end_id);
        mem.eos[kidx as usize] = ok as i8;
        ok
    }

    /// stack.py::shift_terminal (SS2 `lalr_advance`): reduces then shift on the
    /// interned arena. Returns (new kidx, pops into the original chain, pushed
    /// (state, sym) frames) so the Python mirror can rebuild its StackNodes —
    /// the final chain is always ancestor(pops) ++ frames because popped
    /// original frames never return and pushes are fresh interned nodes.
    fn advance_core(
        &self,
        mem: &mut Memos<W>,
        kidx: u32,
        t: u32,
    ) -> Option<(u32, u32, Vec<(u32, u32)>)> {
        let mut cur: i64 = kidx as i64;
        let mut frames: Vec<(u32, u32)> = Vec::new(); // pushes above the anchor
        let mut pops: u32 = 0; // pops consuming ORIGINAL chain frames
        for _ in 0..10_000 {
            let (kind, arg) = self.act(mem.arena[cur as usize].1, t);
            match kind {
                ACT_SHIFT => {
                    let nk = self.intern_node(mem, cur, arg);
                    frames.push((arg, t));
                    return Some((nk, pops, frames));
                }
                ACT_ACCEPT => return Some((cur as u32, pops, frames)),
                ACT_REDUCE => {
                    let (lhs, rhs_len) = self.prods[arg as usize];
                    for _ in 0..rhs_len {
                        if frames.pop().is_none() {
                            pops += 1;
                        }
                        let parent = mem.arena[cur as usize].0;
                        if parent < 0 {
                            return None; // reduce popped past root (caller bug)
                        }
                        cur = parent;
                    }
                    let nxt = self.goto_of(mem.arena[cur as usize].1, lhs);
                    if nxt < 0 {
                        return None;
                    }
                    cur = self.intern_node(mem, cur, nxt as u32) as i64;
                    frames.push((nxt as u32, lhs));
                }
                _ => return None,
            }
        }
        debug_assert!(false, "reduce chain did not terminate");
        None
    }

    /// Memoized shift: (kidx, t) -> kidx | -1.
    fn shift_at(&self, mem: &mut Memos<W>, kidx: u32, t: u32) -> i64 {
        if let Some(&v) = mem.shift.get(&(kidx, t)) {
            return v;
        }
        let v = match self.advance_core(mem, kidx, t) {
            Some((nk, _, _)) => nk as i64,
            None => -1,
        };
        mem.shift.insert((kidx, t), v);
        v
    }

    fn register(
        &self,
        groups: Vec<(Vec<(Vec<u64>, u32)>, Vec<Vec<u8>>, Vec<u8>, Vec<u32>)>,
        ci_ids: Vec<i32>,
    ) -> PyResult<usize> {
        let bytes: Vec<u8> = ci_ids.iter().flat_map(|t| t.to_le_bytes()).collect();
        self.register_impl(groups, bytes)
    }

    /// The registration body's group half: raw walk groups -> VGroups with
    /// THIS kernel's lexeme_ok/prefix_ok filtering (VEvents and tails are
    /// recomputed from (cands, lexeme, remainder) — the cross-producer import
    /// contract: a shared entry adopted by schema B gets B's lexicon
    /// semantics, exactly like register_bytes-from-kernel_groups always did).
    fn build_vgroups(
        &self,
        groups: Vec<(Vec<(Vec<u64>, u32)>, Vec<Vec<u8>>, Vec<u8>, Vec<u32>)>,
    ) -> PyResult<Vec<VGroup<W>>> {
        let mut vgroups = Vec::with_capacity(groups.len());
        for (events, segments, remainder, token_ids) in groups {
            if events.len() != segments.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "events/segments length mismatch",
                ));
            }
            let mut evs = Vec::with_capacity(events.len());
            for (i, (cand_words, _len)) in events.iter().enumerate() {
                let cands: [u64; W] = m_from_words(cand_words);
                let lexeme = &segments[i];
                let mut pass = [0u64; W];
                m_find(&cands, |t| {
                    if self.lexeme_ok(t, lexeme) {
                        m_set(&mut pass, t);
                    }
                    false
                });
                evs.push(VEvent {
                    cand_pass: pass,
                    ign_pick: self.pick_first(&m_and(&cands, &self.ignored)),
                });
            }
            let tail = if remainder.is_empty() {
                Tail::Empty
            } else {
                let st = self.scan_state(&remainder);
                if st == DEAD {
                    Tail::Dead
                } else {
                    let lv = self.live[st as usize];
                    let mut allow = [0u64; W];
                    m_find(&lv, |t| {
                        if self.prefix_ok(t, &remainder) {
                            m_set(&mut allow, t);
                        }
                        false
                    });
                    Tail::Live { ign_ok: m_and_any(&lv, &self.ignored), allow }
                }
            };
            let token_bytes: Vec<u8> =
                token_ids.iter().flat_map(|t| (*t as i32).to_le_bytes()).collect();
            vgroups.push(VGroup { events: evs, tail, token_bytes });
        }
        Ok(vgroups)
    }

    /// Shared registration body; `ci_bytes` is the ci ids as i32-le (stored
    /// verbatim as the hit_pass prefix, bit-packed for fill_bits). The bytes
    /// form exists so Python can hand over a 10k+-id buffer as one memcpy
    /// instead of per-int extraction (20-100 ms per giant entry).
    fn register_impl(
        &self,
        groups: Vec<(Vec<(Vec<u64>, u32)>, Vec<Vec<u8>>, Vec<u8>, Vec<u32>)>,
        ci_bytes: Vec<u8>,
    ) -> PyResult<usize> {
        let ids = ids_from_le(&ci_bytes)?;
        let vgroups = self.build_vgroups(groups)?;
        let ci_bits = ci_bits_of(&ids);
        let st = &mut *self.store.write().unwrap();
        st.entries.push(vgroups);
        st.entry_ci_bits.push(ci_bits);
        st.entry_ci.push(ci_bytes);
        Ok(st.entries.len() - 1)
    }

    /// Kernel v7 registration: parse the blob, build the VGroups (this
    /// kernel's lexicons), pack ci bits, adaptive-encode the payload and hash
    /// the entry id — ALL inside py.detach — then register under store.write
    /// with by_id dedup (idempotent under pool races; the walk-twice race
    /// yields one handle). Returns (handle, entry_id hex, tag, n_groups).
    fn register_blob(
        &self,
        py: Python<'_>,
        blob: Vec<u8>,
        ci_bytes: Vec<u8>,
        key_repr: Vec<u8>,
        vocab: u64,
    ) -> PyResult<(u64, String, u8, u32)> {
        type Built<const W: usize> = (Vec<VGroup<W>>, Vec<u32>, Vec<u8>, [u8; 16], u8, u32);
        let (vgroups, ci_bits, ci_bytes, id16, tag, n_groups) =
            py.detach(move || -> PyResult<Built<W>> {
                let raw = blob_decode(&blob, W)?;
                let n_groups = raw.len() as u32;
                let vgroups = self.build_vgroups(raw)?;
                let ids = ids_from_le(&ci_bytes)?;
                let ci_bits = ci_bits_of(&ids);
                let (tag, payload) = adaptive_encode_ids(&ids, vocab)?;
                let id16 = entry_id_128(&key_repr, tag, &payload);
                Ok((vgroups, ci_bits, ci_bytes, id16, tag, n_groups))
            })?;
        let hex = hex32(&id16);
        let st = &mut *self.store.write().unwrap();
        if let Some(&h) = st.by_id.get(&id16) {
            return Ok((h as u64, hex, tag, n_groups));
        }
        st.entries.push(vgroups);
        st.entry_ci_bits.push(ci_bits);
        st.entry_ci.push(ci_bytes);
        let h = (st.entries.len() - 1) as u32;
        st.by_id.insert(id16, h);
        Ok((h as u64, hex, tag, n_groups))
    }

    /// The per-step CD verdict batch at an interned node; passing groups'
    /// token ids append to `out` as i32-le. The whole batch result is a pure
    /// function of (handle, kidx), so it is memoized outright; on a memo miss
    /// all stack work hits the persistent allowed/shift memos.
    fn cd_groups_pass(&self, mem: &mut Memos<W>, handle: usize, kidx: u32, out: &mut Vec<u8>) {
        if let Some(bytes) = mem.cd.get(&(handle as u32, kidx)) {
            out.extend_from_slice(bytes);
            return;
        }
        let mut fresh: Vec<u8> = Vec::new();
        self.cd_groups_compute(mem, handle, kidx, &mut fresh);
        out.extend_from_slice(&fresh);
        if mem.cd.len() >= 1_000_000 {
            mem.cd.clear(); // unbounded-config backstop; entries recompute on demand
        }
        mem.cd.insert((handle as u32, kidx), fresh);
    }

    fn cd_groups_compute(&self, mem: &mut Memos<W>, handle: usize, kidx: u32, out: &mut Vec<u8>) {
        // lock order: mem (held by caller) -> store.read; store.write is
        // never taken with mem held, so this cannot deadlock
        let st = self.store.read().unwrap();
        for g in &st.entries[handle] {
            let mut cur = kidx;
            let mut ok = true;
            for e in &g.events {
                let allowed = self.allowed_at(mem, cur);
                let pick = {
                    let p = self.pick_first(&m_and(&e.cand_pass, &allowed));
                    if p >= 0 { p } else { e.ign_pick }
                };
                if pick < 0 {
                    ok = false;
                    break;
                }
                let p = pick as u32;
                if !m_bit(&self.ignored, p) {
                    let nxt = self.shift_at(mem, cur, p);
                    if nxt < 0 {
                        ok = false;
                        break;
                    }
                    cur = nxt as u32;
                }
            }
            if ok {
                ok = match &g.tail {
                    Tail::Empty => true,
                    Tail::Dead => false,
                    Tail::Live { ign_ok, allow } => {
                        *ign_ok || m_and_any(allow, &self.allowed_at(mem, cur))
                    }
                };
            }
            if ok {
                out.extend_from_slice(&g.token_bytes);
            }
        }
    }

    fn check_handle(&self, handle: usize) -> PyResult<()> {
        if handle >= self.store.read().unwrap().entries.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("unknown entry handle"));
        }
        Ok(())
    }

    fn check_kidx(&self, mem: &Memos<W>, kidx: u32) -> PyResult<()> {
        if kidx as usize >= mem.arena.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("unknown kidx"));
        }
        Ok(())
    }

    fn cd_pass_at(&self, py: Python<'_>, handle: usize, kidx: u32) -> PyResult<Py<PyBytes>> {
        self.check_handle(handle)?;
        let mem = &mut *self.mem.lock().unwrap();
        self.check_kidx(mem, kidx)?;
        let mut out: Vec<u8> = Vec::new();
        self.cd_groups_pass(mem, handle, kidx, &mut out);
        Ok(PyBytes::new(py, &out).unbind())
    }

    /// SS6 warm hit assembled in one call: ci ids ++ cd-passing ids ++ eos
    /// (when `eos_id >= 0`), i32-le — bit-identical to the Python
    /// np.concatenate([ci, cd_pass, eos]) path.
    fn hit_pass(
        &self,
        py: Python<'_>,
        handle: usize,
        kidx: u32,
        eos_id: i64,
    ) -> PyResult<Py<PyBytes>> {
        self.check_handle(handle)?;
        let mem = &mut *self.mem.lock().unwrap();
        self.check_kidx(mem, kidx)?;
        let mut out: Vec<u8> = {
            // store guard dropped before cd_groups_pass re-reads it (an
            // RwLock read is not reentrancy-safe against a queued writer)
            let st = self.store.read().unwrap();
            let ci = &st.entry_ci[handle];
            let mut out = Vec::with_capacity(ci.len() + 64);
            out.extend_from_slice(ci);
            out
        };
        self.cd_groups_pass(mem, handle, kidx, &mut out);
        if eos_id >= 0 {
            out.extend_from_slice(&(eos_id as i32).to_le_bytes());
        }
        Ok(PyBytes::new(py, &out).unbind())
    }

    /// SS6 warm hit as a full packed bitmask row (`n_words` u32 words): the
    /// entry's pre-packed ci bit words ++ cd-passing bits ++ eos bit. Bit-set
    /// identical to hit_pass's id buffer (tests/mask/test_kernel_parity.py);
    /// ids beyond `n_words * 32` are dropped, matching the numpy row clamp.
    fn fill_row(
        &self,
        mem: &mut Memos<W>,
        handle: usize,
        kidx: u32,
        eos_id: i64,
        n_words: usize,
    ) -> Vec<u32> {
        let mut row = vec![0u32; n_words];
        {
            let st = self.store.read().unwrap();
            let ci = &st.entry_ci_bits[handle];
            let n = ci.len().min(n_words);
            row[..n].copy_from_slice(&ci[..n]);
        }
        let mut cd: Vec<u8> = Vec::new();
        self.cd_groups_pass(mem, handle, kidx, &mut cd);
        for ch in cd.chunks_exact(4) {
            let t = i32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]) as usize;
            let w = t >> 5;
            if w < n_words {
                row[w] |= 1 << (t & 31);
            }
        }
        if eos_id >= 0 {
            let w = (eos_id as usize) >> 5;
            if w < n_words {
                row[w] |= 1 << (eos_id & 31);
            }
        }
        row
    }

    /// fill_bits body with a packed-row memo: the row WITHOUT the eos bit is a
    /// pure function of (handle, kidx) at a fixed word count — same
    /// content-addressed reasoning as `mem.cd` — so it is memoized whole and
    /// the eos bit (which varies per call) is OR'd into the caller's buffer at
    /// copy time. Rows are `n_words * 4` bytes (~19 KB at a 151k vocab), so
    /// the memo is capped: cleared at 4096 entries (~78 MB worst case),
    /// entries recompute on demand. Lives in `Memos` so `reset_interning`
    /// drops it with the arena (kidx invalidation).
    fn fill_bits_memo(
        &self,
        py: Python<'_>,
        mem: &mut Memos<W>,
        handle: usize,
        kidx: u32,
        eos_id: i64,
        buf: &pyo3::buffer::PyBuffer<u32>,
    ) -> PyResult<()> {
        let n_words = buf.item_count();
        let key = (handle as u32, kidx);
        if let Some(row) = mem.row.get(&key) {
            if row.len() == n_words {
                return Self::copy_row_with_eos(py, buf, row, eos_id);
            }
        }
        let row = self.fill_row(mem, handle, kidx, -1, n_words); // eos excluded
        Self::copy_row_with_eos(py, buf, &row, eos_id)?;
        if mem.row.len() >= 4096 {
            mem.row.clear(); // size cap (rows are big); recompute on demand
        }
        mem.row.insert(key, row);
        Ok(())
    }

    /// Overwrite every word of `buf` with `row`, OR-ing in the eos bit (if
    /// `eos_id >= 0` and in range) without mutating the memoized row.
    fn copy_row_with_eos(
        py: Python<'_>,
        buf: &pyo3::buffer::PyBuffer<u32>,
        row: &[u32],
        eos_id: i64,
    ) -> PyResult<()> {
        let w = (eos_id as usize) >> 5;
        if eos_id >= 0 && w < row.len() {
            if let Some(dst) = buf.as_mut_slice(py) {
                if dst.len() == row.len() {
                    for (d, s) in dst.iter().zip(row.iter()) {
                        d.set(*s);
                    }
                    dst[w].set(dst[w].get() | (1u32 << (eos_id & 31)));
                    return Ok(());
                }
            }
            // non-contiguous / mismatched buffer: owned copy with the bit set
            let mut tmp = row.to_vec();
            tmp[w] |= 1 << (eos_id & 31);
            return buf.copy_from_slice(py, &tmp);
        }
        buf.copy_from_slice(py, row)
    }

    fn cd_pass(&self, py: Python<'_>, handle: usize, stack: &[u32]) -> PyResult<Py<PyBytes>> {
        self.check_handle(handle)?;
        let mem = &mut *self.mem.lock().unwrap();
        let kidx = self.intern_chain(mem, stack);
        let mut out: Vec<u8> = Vec::new();
        self.cd_groups_pass(mem, handle, kidx, &mut out);
        Ok(PyBytes::new(py, &out).unbind())
    }

    // ------------------------------------------------------- v6 sessions

    /// E6 token table slice; out-of-range / vocab-hole ids map to empty
    /// (pinned improvement over the Python KeyError; empty always REJECTS).
    #[inline]
    fn token_of<'a>(&self, tabs: &'a SessTabs<W>, t: i64) -> &'a [u8] {
        if t < 0 || (t as usize) + 1 >= tabs.tok_off.len() {
            return &[];
        }
        let a = tabs.tok_off[t as usize] as usize;
        let b = tabs.tok_off[t as usize + 1] as usize;
        &tabs.tok_blob[a..b]
    }

    /// lexer/run.py::scan — maximal-munch with forced emission over ONE buffer.
    /// Returns (events as (start, end, accepting state), remainder start), or
    /// None for ScanReject. Invariant: events partition buf[..rem_start].
    fn lex_scan(&self, tabs: &SessTabs<W>, buf: &[u8]) -> Option<(Vec<(usize, usize, i32)>, usize)> {
        let mut events: Vec<(usize, usize, i32)> = Vec::new();
        let mut i = 0usize;
        loop {
            let mut st = self.dfa_start;
            let mut last: Option<(usize, i32)> = None;
            let mut j = i;
            let mut dead = false;
            while j < buf.len() {
                let nx = self.tr(st, buf[j]);
                if nx == DEAD {
                    dead = true;
                    break;
                }
                st = nx;
                j += 1;
                if tabs.dfa_accept[st as usize] != -1 {
                    last = Some((j, st));
                }
            }
            if !dead {
                // end-of-buffer at a live state: partial stays in the remainder
                // even when currently accepting (maximal munch, "sel"|"select")
                return Some((events, i));
            }
            match last {
                None => return None, // ScanReject: no accept anywhere in this lexeme
                Some((end, acc)) => {
                    events.push((i, end, acc));
                    i = end; // restart from dfa.start after the emitted lexeme
                }
            }
        }
    }

    /// lexer/run.py::finalize — end-of-input greedy re-segmentation of the
    /// remainder. None if ANY position lacks an accepting prefix (incl. a
    /// trailing partial); empty remainder yields Some(vec![]).
    fn lex_finalize(&self, tabs: &SessTabs<W>, rem: &[u8]) -> Option<Vec<(usize, usize, i32)>> {
        let mut out: Vec<(usize, usize, i32)> = Vec::new();
        let mut i = 0usize;
        while i < rem.len() {
            let mut st = self.dfa_start;
            let mut last: Option<(usize, i32)> = None;
            let mut j = i;
            while j < rem.len() {
                st = self.tr(st, rem[j]);
                if st == DEAD {
                    break;
                }
                j += 1;
                if tabs.dfa_accept[st as usize] != -1 {
                    last = Some((j, st));
                }
            }
            let (end, acc) = last?;
            out.push((i, end, acc));
            i = end;
        }
        Some(out)
    }

    /// guide.py pick_viable: priority-ordered (literals ascending, then named)
    /// candidate in `cands & allowed` passing lexeme_ok, else the min-priority
    /// ignored candidate, else None. Identical to RustWalkerImpl::pick_viable.
    fn pick_viable_sess(&self, cands: &[u64; W], lexeme: &[u8], allowed: &[u64; W]) -> Option<u32> {
        let real = m_and(cands, allowed);
        for pass in 0..2u8 {
            let pool = if pass == 0 {
                m_and(&real, &self.literal)
            } else {
                m_and_not(&real, &self.literal)
            };
            if let Some(t) = m_find(&pool, |t| self.lexeme_ok(t, lexeme)) {
                return Some(t);
            }
        }
        let ign = m_and(cands, &self.ignored);
        for pass in 0..2u8 {
            let pool = if pass == 0 {
                m_and(&ign, &self.literal)
            } else {
                m_and_not(&ign, &self.literal)
            };
            if let Some(t) = m_first(&pool) {
                return Some(t);
            }
        }
        None
    }

    /// guide.py partial-lexeme viability tail: some live terminal is ignored,
    /// or allowed at the post-shift node with prefix_ok on the FULL remainder.
    fn tail_ok(&self, mem: &mut Memos<W>, kidx: u32, rem: &[u8]) -> bool {
        let st = self.scan_state(rem);
        if st == DEAD {
            return false; // dead code by the scannable-remainder invariant
        }
        let lv = self.live[st as usize];
        if m_and_any(&lv, &self.ignored) {
            return true;
        }
        let pool = m_and(&lv, &self.allowed_at(mem, kidx));
        m_find(&pool, |t| self.prefix_ok(t, rem)).is_some()
    }

    /// guide.py::_eos_ok — finalize the remainder (winning segmentation),
    /// virtually shift the picks, then eos_ok at the finalized node.
    fn finalize_eos_ok(&self, mem: &mut Memos<W>, tabs: &SessTabs<W>, kidx: u32, rem: &[u8]) -> bool {
        let events = match self.lex_finalize(tabs, rem) {
            None => return false,
            Some(evs) => evs,
        };
        let mut cur = kidx;
        for (s0, e0, acc) in events {
            let cands = tabs.dfa_accepts_all[acc as usize];
            let allowed = self.allowed_at(mem, cur);
            match self.pick_viable_sess(&cands, &rem[s0..e0], &allowed) {
                None => return false,
                Some(t) => {
                    if !m_bit(&self.ignored, t) {
                        let nxt = self.shift_at(mem, cur, t);
                        if nxt < 0 {
                            return false;
                        }
                        cur = nxt as u32;
                    }
                }
            }
        }
        self.eos_at(mem, cur)
    }

    /// guide.py::_derive_status for eos_consumed=False, memoized on
    /// (kidx, remainder) — grammar-pure, so epoch rollover never invalidates
    /// it; reset_interning drops it with the arena (it lives in Memos).
    fn derive_status(&self, mem: &mut Memos<W>, tabs: &SessTabs<W>, kidx: u32, rem: &[u8]) -> u8 {
        let key = (kidx, rem.to_vec());
        if let Some(&s) = mem.stat.get(&key) {
            return s;
        }
        let s = if !self.finalize_eos_ok(mem, tabs, kidx, rem) {
            ST_ACTIVE
        } else if rem.is_empty() && !m_any(&self.allowed_at(mem, kidx)) {
            ST_GRAMMAR_END
        } else {
            ST_ACCEPTING
        };
        if mem.stat.len() >= 1_000_000 {
            mem.stat.clear(); // cap like mem.cd; recompute on demand
        }
        mem.stat.insert(key, s);
        s
    }

    /// The session's kidx, re-interned lazily after reset_interning (the
    /// chain is authoritative; kidx is only a gen-tagged cache).
    fn sess_kidx(&self, mem: &mut Memos<W>, s: &mut Session) -> u32 {
        if s.kgen == mem.gen && s.kidx >= 0 {
            return s.kidx as u32;
        }
        let k = self.intern_chain(mem, &s.chain);
        s.kidx = k as i64;
        s.kgen = mem.gen;
        k
    }

    /// Apply ONE token to `core` (guide.py::_advance semantics, SS6 11-16).
    /// Returns false (core untouched) when the token is not viable. Pinned
    /// COMPLETE semantics (red-team §0): non-eos REJECTED, eos consumed and
    /// stays COMPLETE. Eos legality: status in {ACCEPTING, GRAMMAR_END,
    /// COMPLETE}; the eos check precedes the token-bytes lookup.
    fn step_core(&self, mem: &mut Memos<W>, tabs: &SessTabs<W>, core: &mut SessCore, token: i64) -> bool {
        if core.status == ST_COMPLETE {
            return token == core.eos_id as i64; // repeat-eos consumed, state kept
        }
        if token == core.eos_id as i64 {
            if core.status == ST_ACCEPTING || core.status == ST_GRAMMAR_END {
                core.status = ST_COMPLETE; // stack/remainder kept (COMPLETE fill)
                return true;
            }
            return false;
        }
        let data = self.token_of(tabs, token);
        if data.is_empty() {
            return false; // specials / vocab holes / out-of-range
        }
        let mut buf = Vec::with_capacity(core.remainder.len() + data.len());
        buf.extend_from_slice(&core.remainder);
        buf.extend_from_slice(data);
        let (events, rem_start) = match self.lex_scan(tabs, &buf) {
            None => return false,
            Some(x) => x,
        };
        // shifts: allowed() re-evaluated after each previous shift; the chain
        // delta composes as (cut into the original, pushed tail)
        let mut kidx = core.kidx;
        let mut cut = 0usize;
        let mut tail: Vec<u32> = Vec::new();
        for (s0, e0, acc) in events {
            let cands = tabs.dfa_accepts_all[acc as usize];
            let allowed = self.allowed_at(mem, kidx);
            let pick = match self.pick_viable_sess(&cands, &buf[s0..e0], &allowed) {
                None => return false,
                Some(t) => t,
            };
            if !m_bit(&self.ignored, pick) {
                let (nk, pops, frames) = match self.advance_core(mem, kidx, pick) {
                    None => return false,
                    Some(x) => x,
                };
                for _ in 0..pops {
                    if tail.pop().is_none() {
                        cut += 1;
                    }
                }
                for (st, _sym) in frames {
                    tail.push(st);
                }
                kidx = nk;
            }
        }
        let rem_new = &buf[rem_start..];
        if !rem_new.is_empty() && !self.tail_ok(mem, kidx, rem_new) {
            return false;
        }
        let status = self.derive_status(mem, tabs, kidx, rem_new);
        // commit (candidate fully viable)
        let keep = core.chain.len() - cut;
        core.chain.truncate(keep);
        core.chain.extend_from_slice(&tail);
        core.kidx = kidx;
        core.remainder.clear();
        core.remainder.extend_from_slice(rem_new);
        core.status = status;
        true
    }
}

enum VerdictsAny {
    W1(RustVerdictsImpl<1>),
    W2(RustVerdictsImpl<2>),
    W4(RustVerdictsImpl<4>),
    W8(RustVerdictsImpl<8>),
}

macro_rules! verdicts_dispatch {
    ($self:expr, $v:ident, $body:expr) => {
        match &$self.inner {
            VerdictsAny::W1($v) => $body,
            VerdictsAny::W2($v) => $body,
            VerdictsAny::W4($v) => $body,
            VerdictsAny::W8($v) => $body,
        }
    };
}

#[pyclass]
struct RustVerdicts {
    inner: VerdictsAny,
    width: usize,
}

#[allow(clippy::too_many_arguments)]
fn build_verdicts<const W: usize>(
    kind: Vec<u8>,
    arg: Vec<u32>,
    n_cols: usize,
    goto_tbl: Vec<i32>,
    n_nts: usize,
    prods: Vec<(u32, u32)>,
    end_id: u32,
    ignored: Vec<u64>,
    literal: Vec<u64>,
    trans_v: Vec<i32>,
    live: Vec<Vec<u64>>,
    dfa_start: i32,
    lex: (Option<HashMap<u32, HashSet<Vec<u8>>>>, Option<HashMap<u32, HashSet<Vec<u8>>>>),
) -> RustVerdictsImpl<W> {
    RustVerdictsImpl {
        action_kind: kind,
        action_arg: arg,
        n_cols,
        goto_tbl,
        n_nts,
        prods,
        end_id,
        ignored: m_from_words::<W>(&ignored),
        literal: m_from_words::<W>(&literal),
        trans: trans_v,
        live: live.iter().map(|v| m_from_words::<W>(v)).collect(),
        dfa_start,
        lex_allowed: lex.0,
        lex_prefixes: lex.1,
        store: RwLock::new(EntryStore::default()),
        mem: Mutex::new(Memos::default()),
        tabs: RwLock::new(SessTabs::default()),
        sessions: Mutex::new(SessTable::default()),
    }
}

#[pymethods]
impl RustVerdicts {
    #[new]
    #[pyo3(signature = (action_kind, action_arg, n_states, n_terminals, goto, n_nts, prods,
                        end_id, ignored_mask, literal_mask, trans, live, dfa_start, lexicon=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        action_kind: &Bound<'_, PyBytes>,
        action_arg: &Bound<'_, PyBytes>,
        n_states: usize,
        n_terminals: usize,
        goto: &Bound<'_, PyBytes>,
        n_nts: usize,
        prods: Vec<(u32, u32)>,
        end_id: u32,
        ignored_mask: Vec<u64>,
        literal_mask: Vec<u64>,
        trans: &Bound<'_, PyBytes>,
        live: Vec<Vec<u64>>,
        dfa_start: i32,
        lexicon: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let kind = action_kind.as_bytes().to_vec();
        let ab = action_arg.as_bytes();
        let mut arg = Vec::with_capacity(ab.len() / 4);
        for c in ab.chunks_exact(4) {
            arg.push(u32::from_le_bytes(c.try_into().unwrap()));
        }
        if kind.len() != n_states * n_terminals || arg.len() != n_states * n_terminals {
            return Err(pyo3::exceptions::PyValueError::new_err("action table shape mismatch"));
        }
        let gb = goto.as_bytes();
        let mut goto_tbl = Vec::with_capacity(gb.len() / 4);
        for c in gb.chunks_exact(4) {
            goto_tbl.push(i32::from_le_bytes(c.try_into().unwrap()));
        }
        if goto_tbl.len() != n_states * n_nts {
            return Err(pyo3::exceptions::PyValueError::new_err("goto table shape mismatch"));
        }
        let tb = trans.as_bytes();
        let mut trans_v = Vec::with_capacity(tb.len() / 4);
        for c in tb.chunks_exact(4) {
            trans_v.push(i32::from_le_bytes(c.try_into().unwrap()));
        }
        let lex = parse_lexicon(lexicon)?;
        let width = width_for(n_terminals).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "{n_terminals} terminals exceeds the 512-terminal kernel bound"
            ))
        })?;
        let inner = match width {
            1 => VerdictsAny::W1(build_verdicts::<1>(kind, arg, n_terminals, goto_tbl, n_nts, prods, end_id, ignored_mask, literal_mask, trans_v, live, dfa_start, lex)),
            2 => VerdictsAny::W2(build_verdicts::<2>(kind, arg, n_terminals, goto_tbl, n_nts, prods, end_id, ignored_mask, literal_mask, trans_v, live, dfa_start, lex)),
            4 => VerdictsAny::W4(build_verdicts::<4>(kind, arg, n_terminals, goto_tbl, n_nts, prods, end_id, ignored_mask, literal_mask, trans_v, live, dfa_start, lex)),
            _ => VerdictsAny::W8(build_verdicts::<8>(kind, arg, n_terminals, goto_tbl, n_nts, prods, end_id, ignored_mask, literal_mask, trans_v, live, dfa_start, lex)),
        };
        Ok(RustVerdicts { inner, width })
    }

    #[getter]
    fn width(&self) -> usize {
        self.width
    }

    /// Register one cache entry's CD groups + its ci token ids (the hit_pass
    /// prefix); event masks are u64 word lists. `&self` since kernel v7
    /// (entry store behind RwLock) — no mutable pyclass borrow anywhere.
    fn register(
        &self,
        groups: Vec<(Vec<(Vec<u64>, u32)>, Vec<Vec<u8>>, Vec<u8>, Vec<u32>)>,
        ci_ids: Vec<i32>,
    ) -> PyResult<usize> {
        verdicts_dispatch!(self, v, v.register(groups, ci_ids))
    }

    /// register() with the ci ids as ONE i32-le byte buffer — the hot
    /// registration path (a literal-interior giant's 10k+ ids cross the FFI
    /// as a memcpy instead of per-int extraction).
    fn register_bytes(
        &self,
        groups: Vec<(Vec<(Vec<u64>, u32)>, Vec<Vec<u8>>, Vec<u8>, Vec<u32>)>,
        ci: &Bound<'_, PyBytes>,
    ) -> PyResult<usize> {
        let bytes = ci.as_bytes().to_vec();
        verdicts_dispatch!(self, v, v.register_impl(groups, bytes))
    }

    /// Kernel v7: one GIL-released call = blob parse + VGroup build (THIS
    /// kernel's lexicons) + ci bit-pack + adaptive encode + BLAKE2b-128
    /// entry id + registration (by_id-deduplicated). `key_repr` is
    /// repr(key).encode() computed Python-side (µs) — the id is byte-
    /// identical to grid/mask/cache.py make_entry for the same key/ci.
    /// Doubles as the cross-producer import: a foreign producer registers a
    /// shared entry from its (blob, ci_bytes) with no Python payload path.
    /// Returns (handle, entry_id hex, tag, n_groups).
    fn register_blob(
        &self,
        py: Python<'_>,
        blob: &Bound<'_, PyBytes>,
        ci: &Bound<'_, PyBytes>,
        key_repr: &Bound<'_, PyBytes>,
        vocab_size: u64,
    ) -> PyResult<(u64, String, u8, u32)> {
        let blob_v = blob.as_bytes().to_vec();
        let ci_v = ci.as_bytes().to_vec();
        let key_v = key_repr.as_bytes().to_vec();
        verdicts_dispatch!(self, v, v.register_blob(py, blob_v, ci_v, key_v, vocab_size))
    }

    // ------------------------------------------------- interned addressing (v4)

    /// Intern a full root->top LALR state chain; returns its kidx.
    fn intern_chain(&self, stack: Vec<u32>) -> PyResult<u32> {
        if stack.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("empty stack"));
        }
        Ok(verdicts_dispatch!(self, v, {
            let mem = &mut *v.mem.lock().unwrap();
            v.intern_chain(mem, &stack)
        }))
    }

    /// Intern one child node; `parent = -1` interns a root.
    fn intern_child(&self, parent: i64, state: u32) -> PyResult<u32> {
        verdicts_dispatch!(self, v, {
            let mem = &mut *v.mem.lock().unwrap();
            if parent >= mem.arena.len() as i64 || parent < -1 {
                return Err(pyo3::exceptions::PyValueError::new_err("unknown parent kidx"));
            }
            Ok(v.intern_node(mem, parent, state))
        })
    }

    /// Number of interned nodes (Python caps this and calls reset_interning).
    fn intern_count(&self) -> usize {
        verdicts_dispatch!(self, v, v.mem.lock().unwrap().arena.len())
    }

    /// Drop the arena and every memo (including the v6 session-binding map —
    /// it is kidx-keyed). All outstanding kidx become invalid — the Python
    /// caller bumps its generation and re-interns lazily; live kernel
    /// sessions own their raw state chains and re-intern via the bumped
    /// `Memos::gen` (risk (c)).
    fn reset_interning(&self) {
        verdicts_dispatch!(self, v, {
            let mem = &mut *v.mem.lock().unwrap();
            let gen = mem.gen;
            *mem = Memos::default();
            mem.gen = gen + 1;
        })
    }

    /// Per-step batch at an interned node (persistent memos).
    fn cd_pass_at(&self, py: Python<'_>, handle: usize, kidx: u32) -> PyResult<Py<PyBytes>> {
        verdicts_dispatch!(self, v, v.cd_pass_at(py, handle, kidx))
    }

    /// SS6 warm hit in one call: ci ++ cd-passing ++ eos (i32-le buffer).
    fn hit_pass(&self, py: Python<'_>, handle: usize, kidx: u32, eos_id: i64) -> PyResult<Py<PyBytes>> {
        verdicts_dispatch!(self, v, v.hit_pass(py, handle, kidx, eos_id))
    }

    /// SS2 kernel #4: write the warm hit as a packed uint32 bitmask directly
    /// into `out` (a writable C-contiguous uint32 buffer, e.g. one row of
    /// vLLM's batch bitmask). Every word of `out` is overwritten. One FFI
    /// call per (handle, kidx) — the scheduler-side fill_bitmask hot path.
    /// Warm calls hit the packed-row memo (`Memos::row`) and reduce to a
    /// memcpy plus the per-call eos bit.
    fn fill_bits(
        &self,
        py: Python<'_>,
        handle: usize,
        kidx: u32,
        eos_id: i64,
        out: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let buf = pyo3::buffer::PyBuffer::<u32>::get(out)?;
        verdicts_dispatch!(self, v, {
            v.check_handle(handle)?;
            let mem = &mut *v.mem.lock().unwrap();
            v.check_kidx(mem, kidx)?;
            v.fill_bits_memo(py, mem, handle, kidx, eos_id, &buf)
        })
    }

    /// stack.py::allowed_terminals at an interned node (memoized).
    fn allowed_mask_at(&self, kidx: u32) -> PyResult<Vec<u64>> {
        verdicts_dispatch!(self, v, {
            let mem = &mut *v.mem.lock().unwrap();
            v.check_kidx(mem, kidx)?;
            Ok(v.allowed_at(mem, kidx).to_vec())
        })
    }

    /// stack.py::eos_ok_stack at an interned node (memoized).
    fn eos_ok_at(&self, kidx: u32) -> PyResult<bool> {
        verdicts_dispatch!(self, v, {
            let mem = &mut *v.mem.lock().unwrap();
            v.check_kidx(mem, kidx)?;
            Ok(v.eos_at(mem, kidx))
        })
    }

    /// SS2 `lalr_advance`: reduces+shift from an interned node. Returns
    /// (new kidx, pops into the original chain, pushed (state, sym) frames),
    /// or None when `t` is not viable — mirroring stack.py::shift_terminal.
    fn advance_frames(&self, kidx: u32, t: u32) -> PyResult<Option<(u32, u32, Vec<(u32, u32)>)>> {
        verdicts_dispatch!(self, v, {
            let mem = &mut *v.mem.lock().unwrap();
            v.check_kidx(mem, kidx)?;
            Ok(v.advance_core(mem, kidx, t))
        })
    }

    // ----------------------------------------------------- legacy chain APIs

    /// Per-step batch; returns passing ids as an i32-le buffer (group order).
    fn cd_pass(&self, py: Python<'_>, handle: usize, stack: Vec<u32>) -> PyResult<Py<PyBytes>> {
        if stack.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("empty stack"));
        }
        verdicts_dispatch!(self, v, v.cd_pass(py, handle, &stack))
    }

    /// stack.py::allowed_terminals as a little-endian u64 word list.
    fn allowed_mask(&self, stack: Vec<u32>) -> PyResult<Vec<u64>> {
        if stack.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("empty stack"));
        }
        Ok(verdicts_dispatch!(self, v, {
            let mem = &mut *v.mem.lock().unwrap();
            let kidx = v.intern_chain(mem, &stack);
            v.allowed_at(mem, kidx).to_vec()
        }))
    }

    /// stack.py::eos_ok_stack: ACCEPT reachable via the reduce chain of $end.
    fn eos_ok(&self, stack: Vec<u32>) -> PyResult<bool> {
        if stack.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("empty stack"));
        }
        Ok(verdicts_dispatch!(self, v, {
            let mem = &mut *v.mem.lock().unwrap();
            let kidx = v.intern_chain(mem, &stack);
            v.eos_at(mem, kidx)
        }))
    }

    // ------------------------------------------------------------ v6 sessions

    /// Upload the E6-normative token_bytes table once: one blob + i32-le
    /// offsets (len n_tokens + 1). Ids outside the table map to empty bytes,
    /// which always REJECT.
    fn set_token_bytes(&self, blob: &Bound<'_, PyBytes>, offsets: &Bound<'_, PyBytes>) -> PyResult<()> {
        let blob_v = blob.as_bytes().to_vec();
        let ob = offsets.as_bytes();
        if ob.len() % 4 != 0 || ob.len() < 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "offsets must be i32-le with at least one entry",
            ));
        }
        let mut off = Vec::with_capacity(ob.len() / 4);
        for c in ob.chunks_exact(4) {
            off.push(i32::from_le_bytes(c.try_into().unwrap()));
        }
        if off[0] != 0
            || *off.last().unwrap() as usize != blob_v.len()
            || off.windows(2).any(|w| w[0] > w[1])
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "offsets must be monotone from 0 to len(blob)",
            ));
        }
        verdicts_dispatch!(self, v, {
            let tabs = &mut *v.tabs.write().unwrap();
            tabs.tok_blob = blob_v;
            tabs.tok_off = off;
            Ok(())
        })
    }

    /// Upload the scanner accept tables once: per-state winning terminal
    /// (i32-le, -1 none) + per-state full candidate sets (u64 word lists).
    fn set_dfa_accept(&self, accept: &Bound<'_, PyBytes>, accepts_all: Vec<Vec<u64>>) -> PyResult<()> {
        let ab = accept.as_bytes();
        let mut acc = Vec::with_capacity(ab.len() / 4);
        for c in ab.chunks_exact(4) {
            acc.push(i32::from_le_bytes(c.try_into().unwrap()));
        }
        verdicts_dispatch!(self, v, {
            if acc.len() != v.live.len() || accepts_all.len() != v.live.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "accept table length must match the DFA state count",
                ));
            }
            let tabs = &mut *v.tabs.write().unwrap();
            tabs.dfa_accept = acc;
            tabs.dfa_accepts_all = accepts_all.iter().map(|w| m_from_words(w)).collect();
            Ok(())
        })
    }

    /// New session at the given root->top LALR state chain (empty remainder);
    /// initial status derived in-kernel. Requires both session tables set.
    fn session_new(&self, chain: Vec<u32>, eos_id: u32) -> PyResult<u64> {
        if chain.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("empty stack"));
        }
        verdicts_dispatch!(self, v, {
            let tabs = &*v.tabs.read().unwrap();
            if tabs.tok_off.is_empty() || tabs.dfa_accept.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "session tables not set (set_token_bytes / set_dfa_accept first)",
                ));
            }
            let tbl = &mut *v.sessions.lock().unwrap();
            let mem = &mut *v.mem.lock().unwrap();
            let kidx = v.intern_chain(mem, &chain);
            let status = v.derive_status(mem, tabs, kidx, b"");
            let sid = tbl.next;
            tbl.next += 1;
            tbl.map.insert(sid, Session {
                chain,
                kidx: kidx as i64,
                kgen: mem.gen,
                remainder: Vec::new(),
                status,
                n_generated: 0,
                prev_token: -1,
                eos_id,
                log: Vec::new(),
            });
            Ok(sid)
        })
    }

    fn session_free(&self, sid: u64) -> bool {
        verdicts_dispatch!(self, v, v.sessions.lock().unwrap().map.remove(&sid).is_some())
    }

    /// Apply one token. 0 = REJECTED (no state mutation); else bit0 OK,
    /// bit1 COMPLETE, bit2 UNBOUND (successor (kidx, remainder) has no fill
    /// binding — the Python side binds from T1 or schedules a prefetch).
    fn session_accept(&self, sid: u64, token: i64) -> PyResult<u32> {
        verdicts_dispatch!(self, v, {
            let tbl = &mut *v.sessions.lock().unwrap();
            let SessTable { map, c, .. } = tbl;
            let s = map
                .get_mut(&sid)
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("unknown session"))?;
            let mem = &mut *v.mem.lock().unwrap();
            let tabs = &*v.tabs.read().unwrap();
            let kidx0 = v.sess_kidx(mem, s);
            let mut core = SessCore {
                chain: s.chain.clone(),
                kidx: kidx0,
                remainder: s.remainder.clone(),
                status: s.status,
                eos_id: s.eos_id,
            };
            if !v.step_core(mem, tabs, &mut core, token) {
                c.rejects += 1;
                return Ok(0);
            }
            // commit + rollback-log delta (restore = truncate lcp, re-push popped)
            let lcp = s
                .chain
                .iter()
                .zip(core.chain.iter())
                .take_while(|(a, b)| a == b)
                .count();
            s.log.push(SessLog {
                popped: s.chain[lcp..].to_vec(),
                pushed: (core.chain.len() - lcp) as u32,
                remainder: std::mem::take(&mut s.remainder),
                status: s.status,
                prev_token: s.prev_token,
                n_generated: s.n_generated,
            });
            s.chain = core.chain;
            s.kidx = core.kidx as i64;
            s.kgen = mem.gen;
            s.remainder = core.remainder;
            s.status = core.status;
            s.prev_token = token;
            if token != s.eos_id as i64 {
                s.n_generated += 1; // eos-accepts do NOT bump (guide.py:297)
            }
            c.accepts += 1;
            let mut flags = FLAG_OK;
            if s.status == ST_COMPLETE {
                flags |= FLAG_COMPLETE;
            }
            if !mem.bind.contains_key(&(core.kidx, s.remainder.clone())) {
                flags |= FLAG_UNBOUND;
            }
            Ok(flags)
        })
    }

    /// Longest viable prefix WITHOUT committing (scratch replay of
    /// session_accept semantics, memo writes only).
    fn session_validate(&self, sid: u64, tokens: Vec<i64>) -> PyResult<usize> {
        verdicts_dispatch!(self, v, {
            let tbl = &mut *v.sessions.lock().unwrap();
            let s = tbl
                .map
                .get_mut(&sid)
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("unknown session"))?;
            let mem = &mut *v.mem.lock().unwrap();
            let tabs = &*v.tabs.read().unwrap();
            let kidx0 = v.sess_kidx(mem, s);
            let mut core = SessCore {
                chain: s.chain.clone(),
                kidx: kidx0,
                remainder: s.remainder.clone(),
                status: s.status,
                eos_id: s.eos_id,
            };
            let mut n = 0usize;
            for t in tokens {
                if !v.step_core(mem, tabs, &mut core, t) {
                    break;
                }
                n += 1;
            }
            Ok(n)
        })
    }

    /// Rewind up to `n` accepted tokens (v5 truncation semantics: n past the
    /// log lands on the initial state; each entry restores chain, remainder,
    /// status, prev_token and n_generated exactly — including eos-accepts).
    fn session_rollback(&self, sid: u64, n: u64) -> PyResult<()> {
        verdicts_dispatch!(self, v, {
            let tbl = &mut *v.sessions.lock().unwrap();
            let s = tbl
                .map
                .get_mut(&sid)
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("unknown session"))?;
            let k = (n as usize).min(s.log.len());
            for _ in 0..k {
                let e = s.log.pop().unwrap();
                let keep = s.chain.len() - e.pushed as usize;
                s.chain.truncate(keep);
                s.chain.extend_from_slice(&e.popped);
                s.remainder = e.remainder;
                s.status = e.status;
                s.prev_token = e.prev_token;
                s.n_generated = e.n_generated;
            }
            if k > 0 {
                s.kidx = -1; // chain is authoritative; re-intern lazily
            }
            Ok(())
        })
    }

    /// Back to the initial state (rollback of the whole log).
    fn session_reset(&self, sid: u64) -> PyResult<()> {
        self.session_rollback(sid, u64::MAX)
    }

    /// Warm fill: write the packed bitmask row for the session's current
    /// (kidx, remainder) binding into `out` (eos bit iff status allows
    /// termination and is not COMPLETE — the fill-after-COMPLETE semantics
    /// still consult the entry). Returns the handle, or -1 on an unbound
    /// configuration (the Python miss path binds and retries).
    fn session_fill(&self, py: Python<'_>, sid: u64, out: &Bound<'_, PyAny>) -> PyResult<i64> {
        let buf = pyo3::buffer::PyBuffer::<u32>::get(out)?;
        verdicts_dispatch!(self, v, {
            let tbl = &mut *v.sessions.lock().unwrap();
            let SessTable { map, c, .. } = tbl;
            let s = map
                .get_mut(&sid)
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("unknown session"))?;
            let mem = &mut *v.mem.lock().unwrap();
            let kidx = v.sess_kidx(mem, s);
            match mem.bind.get(&(kidx, s.remainder.clone())).copied() {
                None => {
                    c.fills_miss += 1;
                    Ok(-1)
                }
                Some(h) => {
                    let eos = if s.status == ST_ACCEPTING || s.status == ST_GRAMMAR_END {
                        s.eos_id as i64
                    } else {
                        -1
                    };
                    v.fill_bits_memo(py, mem, h as usize, kidx, eos, &buf)?;
                    c.fills_hit += 1;
                    Ok(h as i64)
                }
            }
        })
    }

    /// Bind the session's current (kidx, remainder) to a registered entry
    /// handle. The caller has validated the T1 key + OBL-KEY1 guard FIRST
    /// (bind-time enforcement); bindings are last-write-wins with identical
    /// values (handles are content-addressed by entry_id).
    fn session_bind(&self, sid: u64, handle: usize) -> PyResult<()> {
        verdicts_dispatch!(self, v, {
            v.check_handle(handle)?;
            let tbl = &mut *v.sessions.lock().unwrap();
            let SessTable { map, c, .. } = tbl;
            let s = map
                .get_mut(&sid)
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("unknown session"))?;
            let mem = &mut *v.mem.lock().unwrap();
            let kidx = v.sess_kidx(mem, s);
            if mem.bind.len() >= 1_000_000 {
                mem.bind.clear(); // unbounded-config backstop; rebind on demand
            }
            mem.bind.insert((kidx, s.remainder.clone()), handle as u32);
            c.binds += 1;
            Ok(())
        })
    }

    /// (kidx, remainder, status, n_generated, prev_token) — status encoding
    /// 0 ACTIVE / 1 ACCEPTING / 2 GRAMMAR_END / 3 COMPLETE.
    #[allow(clippy::type_complexity)]
    fn session_state(&self, py: Python<'_>, sid: u64) -> PyResult<(i64, Py<PyBytes>, u8, u32, i64)> {
        verdicts_dispatch!(self, v, {
            let tbl = &mut *v.sessions.lock().unwrap();
            let s = tbl
                .map
                .get_mut(&sid)
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("unknown session"))?;
            let mem = &mut *v.mem.lock().unwrap();
            let kidx = v.sess_kidx(mem, s);
            Ok((
                kidx as i64,
                PyBytes::new(py, &s.remainder).unbind(),
                s.status,
                s.n_generated,
                s.prev_token,
            ))
        })
    }

    /// (allowed-mask words, remainder) for the session's CURRENT node — the
    /// pure walk/bind inputs, resolved atomically in-kernel so the Python
    /// side never handles a kidx that a concurrent reset could dangle.
    fn session_walk_inputs(&self, py: Python<'_>, sid: u64) -> PyResult<(Vec<u64>, Py<PyBytes>)> {
        verdicts_dispatch!(self, v, {
            let tbl = &mut *v.sessions.lock().unwrap();
            let s = tbl
                .map
                .get_mut(&sid)
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("unknown session"))?;
            let mem = &mut *v.mem.lock().unwrap();
            let kidx = v.sess_kidx(mem, s);
            let a = v.allowed_at(mem, kidx).to_vec();
            Ok((a, PyBytes::new(py, &s.remainder).unbind()))
        })
    }

    /// Drop every (kidx, remainder) -> handle binding (cache-epoch rollover;
    /// also mirrored by the Python _entry_memo cap clear).
    fn clear_bindings(&self) {
        verdicts_dispatch!(self, v, {
            v.mem.lock().unwrap().bind.clear();
        })
    }

    /// Session telemetry counters (Python folds fills_hit into cache.hits).
    fn session_stats(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        verdicts_dispatch!(self, v, {
            let d = PyDict::new(py);
            let tbl = &*v.sessions.lock().unwrap();
            d.set_item("fills_hit", tbl.c.fills_hit)?;
            d.set_item("fills_miss", tbl.c.fills_miss)?;
            d.set_item("accepts", tbl.c.accepts)?;
            d.set_item("rejects", tbl.c.rejects)?;
            d.set_item("binds", tbl.c.binds)?;
            d.set_item("sessions", tbl.map.len())?;
            let mem = &*v.mem.lock().unwrap();
            d.set_item("bindings", mem.bind.len())?;
            Ok(d.unbind())
        })
    }
}

#[pymodule]
fn grid_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustWalker>()?;
    m.add_class::<RustVerdicts>()?;
    m.add_function(wrap_pyfunction!(encode_mask, m)?)?;
    m.add_function(wrap_pyfunction!(entry_id_hex, m)?)?;
    m.add("__kernel_version__", 7)?;
    Ok(())
}
