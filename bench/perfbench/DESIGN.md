# perfbench: the 0.3.x candidate-selection benchmark

Purpose: a fixed, fast-to-run task set on which competing performance
approaches are implemented, measured, and eliminated - so 0.3.x integrates
winners, not hunches. Runs on one machine in minutes, not the 11.3k-schema
full set.

## Ground rules (epoch discipline)

- Correctness is a GATE, not a metric: every candidate must reproduce the
  v0.2.5 error results exactly on all perfbench sets (same pass/reject/
  recorded outcomes). A candidate that trades correctness for speed is
  eliminated regardless of its numbers.
- All candidates measured on the same machine, same run, interleaved, so
  numbers are comparable. Report per set: TTFM p50/p90/p99/max, TBM
  p50/p99/max, peak RSS, and per-phase attribution.
- TTFM is reported as TWO columns per set (E4): *compile-only* (maskbench
  compile_grammar semantics: the five compile phases + GridGuide
  construction) and *first-mask-included* (+ the first compute_mask on the
  initial state). Lazy scanners defer product construction past the compile
  phases and the kernel/genN gates force that first mask to pure Python, so
  a compile-only column alone under-reports exactly the schemas the lazy
  path rescues. Both columns are child-written stats
  (profile_phases.py --first-mask: stats.ttfm_compile_us /
  stats.ttfm_first_us); never recompute them downstream, never publish one
  column without naming it. The old metric keeps its name and meaning.
- Published outcome counts go through bench/perfbench/outcomes.py (the
  classifier is the F1-retraction guard: records with running != null and
  no timeout_s/rc marker are `incomplete`, never ok; at-cap statuses
  without a timeout marker classify as timeouts). Cross-leg claims use its
  compare mode: baseline timeouts have no oracle (timeout -> terminating is
  the sanctioned direction); ok -> anything and declared-class changes are
  gate failures.
- The full 11.3k set is run ONCE at the end on the single winning
  configuration; perfbench is for selection, the full set is for the
  published table. That run happens only after the TTFM definition above is
  settled (it is, as of E4) and reports both columns.

## Task sets (manifest.json, extracted from the v0.2.5 full-run statuses)

- `ttfm_tail_1pct` (113): the top 1% of schemas by compile time; they carry
  77% of total full-set compile time. The target.
- `ttfm_capped` (16): schemas pinned at the 120s compile cap. Any candidate
  that brings these under 1s changes the headline.
- `stratified_200`: 20 schemas per compile-time decile (seed 42). The
  no-regression guard: the median must not get worse while the tail gets
  better.
- `tbm_tail_100`: worst schemas by max single-mask time. Empirically a
  strict subset of ttfm_tail_1pct: both performance surfaces concentrate
  in the same big-grammar schemas.
- `synthetic/` (generated, parametric): isolates mechanisms so we measure
  scaling exponents, not anecdotes.
  - length windows: string minLength/maxLength (0,N), N in
    {16, 64, 128, 256, 1024}
  - terminal count: K constrained-string properties, K in {8, 32, 128, 512}
  - required-key subsets: R required of 2R properties, R in {2, 6, 10}
    (the 2^R object machine)
  - enum width: E string values, E in {10, 100, 1000, 10000}
  - pattern complements + format terminals crossed with the above

## Phase instrumentation (prerequisite, lands before any candidate)

Wrap the compile pipeline with per-phase timers surfaced in the status
JSON: normalize -> schema compile -> spec.load -> projection -> LALR ->
scanner build (per-terminal, so the worst terminal is named). Today's
bottleneck attribution is directional (dev-time measurements point at
scanner subset construction, LALR second); phase data over ttfm_tail_1pct
makes it exact before we bet an epoch on it.

## Candidate ladder (each independently toggleable; measured cumulatively
   and in isolation)

1. Lazy scanner DFA: subset-construct states on demand during decoding
   (RE2-style), memoized. Upfront cost goes to ~zero; llguidance's
   sub-millisecond TTFM at p99 is the existence proof for this class.
2. Counter-based bounded repetition: {m,n} as counter automata instead of
   state expansion; removes the <=64 window budget and the degradation
   machinery (also recovers correctness on degraded terminals).
3. Rust scanner build: port subset construction to grid_core for whatever
   remains eagerly built.
4. Cross-terminal and cross-schema DFA sharing: hash-cons DFA states;
   precompile the 9 FORMAT_PATTERNS once per process; share terminal DFAs
   across schemas keyed by terminal source.
5. Persistent compile cache keyed by schema hash: cold once per deployment
   lifetime. Orthogonal to 1-4; measured as a separate serving scenario.
6. Open lane: any further idea enters as a numbered candidate with the
   same gate and the same report format.

## Store cold/warm protocol (S3) + reserved kernel-keyed namespaces

Every artifact-store measurement runs in a FRESH process per scenario —
same-process "warm" conflates in-memory memos (registry single-flight,
factored._COMPONENTS, T1/T2) with the store and reads ~0.16 ms where a true
redeploy pays spec-load + unpickle. Scenarios per schema
(bench/perfbench/store_coldwarm.py):

- A: flag-on, EMPTY store (cold + persist cost);
- B: flag-on, PRE-POPULATED store (the redeploy warm hit);
- C: flag-off baseline.

A and B are reported separately, never blended (SELECTION.md #8); warm-hit
p50/p99 for B is the number that gates any future default-on decision. The
harness reports store size on disk alongside, and page-cache state is a
sensitivity note (post-purge B is a different measurement). The journal TBM
protocol additionally simulates a redeploy: process 1 serves and flushes its
journal; process 2 restores it under GRID_ADMIT_WARM=1, runs admission
warmup off-batch, and measures the first-request TBM distribution against
the no-journal bounded cold-miss tail.

Two namespaces stay RESERVED (key shapes only, no payloads this epoch), both
keyed on grid.serving.artifact_store.kernel_fingerprint() = blake2b(grid_core
.so) because grid_core exports no version/blob-format constant:

- T2 mask blobs (CANDIDATES #20b): MaskEntryV7.blob is the kernel's own
  register_blob export format; a wrong-key hit is the forbidden
  served-wrong-mask class. Key: (dialect, schema_fp, tokenizer_fingerprint,
  vocab_size, kernel_fingerprint(), blob-format const). Blocked on a
  served-mask-parity gate.
- RustWalker ingestion arenas (trie nodes/trans/accept + accepts_all/live
  word lists): key (scanner key, trie fingerprint, kernel_fingerprint(),
  kernel word width). Blocked on the RUST_SCANNER un-hold / size-gated FFI
  dispatch decision.

## Decision rule

Winner = the smallest candidate set that (a) passes the correctness gate,
(b) brings ttfm_capped under 1s each, (c) cuts ttfm_tail_1pct total by
>=10x, (d) does not regress stratified_200 p50 by more than 10%. Ties
break toward less code.
