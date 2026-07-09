# GRID — Grammar-Railed Decoding
## Implementation Design Document

**Status:** ready for implementation (v1.1 — post-review; findings from the planned 4-lens design review incorporated)
**Companion:** `GUARDRAIL-REDESIGN.md` (the *why*: design-evolution conclusions, chosen methods, proofs, budgets, benchmark rationale). This document is the *what and how*: modules, entities, state machines, interfaces, error taxonomy, tests, and the verification gates that must pass before each milestone proceeds.
**Interface convention:** GRID exposes the Guide / instruction / logits-processor / tokenizer / sampler protocol shapes shared across our internal generation tools. The shapes are defined in `grid/protocols.py` (§4.1) with self-contained conformance tests under `tests/protocols/`; GRID has no runtime dependency on any external constrained-decoding library, and its implementation is original throughout.

---

## 1. Scope

GRID generates SQL (extensible to any LALR(1)-parsable language) from an LLM such that:

- **Soundness:** every emitted token keeps the detokenized output a viable prefix of the role/schema-projected grammar's language.
- **Completeness:** no token is masked whose bytes can extend the current viable prefix (byte-fallback vocabularies).
- **Termination:** EOS is legal iff the output is a complete sentence of the grammar; a stack-cumulative, **token-denominated** reserve prevents budget-exceeded truncation.
- **Near-linear scaling (R):** amortized O(1) guard-rail cost per token, independent of output length *n*; per-step worst case bounded by nesting depth, never by *n*.
- **Auditability:** a hash-chained per-token record of what was permitted/blocked, replayable against versioned grammar artifacts.

Out of mask scope (by proof, companion §4.6): column-level RBAC → post-parse `SemanticChecker`; semantic validity beyond the grammar → same. Out of v1 scope (recorded in §13): distribution-faithfulness arms (GAD/CRANE), BIRD dataset, byte-level jump-forward with re-tokenization, Earley fallback engine, approximate deadline fallbacks.

## 2. Architecture overview

```
                       OFFLINE / PER-DEPLOYMENT
  PolicyBundle ──► RoleProjection (L2) ─┐
  DialectGrammar (L1) ──────────────────┼─► compose ► reduce ► LALR compile ─► CompiledGrammar
  SchemaSnapshot ──► SchemaLexicon (L3) ┘                                        (fingerprinted)
  TokenizerAdapter ──► TokenTrie (per tokenizer fingerprint, immutable)
                       ReserveTable (per (grammar, tokenizer) fingerprint)

                       PER REQUEST
  (role, schema, tokenizer) ──► GrammarRegistry.lookup ─► CompiledGrammar (single-flight on miss)
                                GridGuide(compiled, trie, cache, audit) ─► GridLogitsProcessor

                       PER TOKEN (hot path)
  parser stack ──allowed terminals──► MaskProducer ─► MaskCache (hit | miss: trie walk + write-back)
       │                                  │              + per-step context-dependent residue check
       └── advance on sampled token ◄── sampler ◄── logits.masked_fill_(−inf) ◄──┘
       └── AuditLog.append(config_hash, mask_entry_id, token, blocked_count)   [every step, incl. EOS/Write]
```

Two implementations of the same semantics, bound by differential tests:

- **`grid/_reference/`** — pure-Python executable specification: a brute-force trial-parse loop over the vocabulary. Slow, obviously-correct, used as the oracle in gates G3/G4/G5. Never shipped in the hot path.
- **Fast path** — Python orchestration; hot kernels in `grid_core` (Rust/pyo3) from milestone M4. **Before M4, the Python stand-in operates on the final artifact formats from day one**: `trie/build.py` produces a numpy `uint64` array of DFS-contiguous 8-byte trie nodes plus a token-id table — exactly the buffer `grid_core` consumes zero-copy at M4. The Python walk indexes that array directly (vectorizing only where natural, e.g. batched DFA transitions). The frozen kernel signatures, identical pre- and post-M4:
  - `walk(trie_buf, lexer_state, allowed_bitset, l3_refs) -> (ci_mask, cd_token_list)`
  - `check_context_dependent(cd_token_list, stack_ref) -> mask_bits`
  - `lalr_advance(stack_ref, terminal_id) -> stack_ref`
  - `apply_token_bitmask(logits, mask) -> None (in-place)`

  M4 is a rebind of these four symbols; G7 does **not** gate the Python stand-in.
  Kernel v5 added the scheduler-side `fill_bits` row fill; v5.1 the
  verdict-equivalence CD grouping + packed-row memo; kernel v6 moves the whole
  per-request serving step in-kernel (`session_accept`/`session_fill`, one FFI
  each — warm step 1.33 µs/request measured locally), gated to `audit is None`
  paths; audit-enabled and processor-mode guides keep the v5 Python path.

  *All four symbols are now bound in `grid_core` (kernel v4): `walk` (RustWalker),
  `check_context_dependent` (RustVerdicts.cd_pass_at), `lalr_advance`
  (RustVerdicts.advance_frames — reduces+shift on a persistent interned-stack
  arena, mirrored back into Python StackNodes for the audit hash-chain), and
  `apply_token_bitmask` (fill_bitmask, kernel #4). Kernel v4 adds a persistent,
  structurally-interned stack arena with cross-token memos and a one-call
  `hit_pass` (ci ++ cd-pass ++ eos assembled kernel-side): warm-hit p50 fell
  12.9→3.5 µs on the M-series dev host, meeting G7's `<10 µs` criterion locally
  for the first time. The cold trie walk releases the GIL (`walk_raw` under
  `py.detach`) so it overlaps the GPU forward window — the §6 overlap contract.*

## 3. Package layout

```
grid/
  protocols.py          # tool-family protocol shapes (normative definitions)
  guide.py              # GridGuide, GridState  (Guide protocol implementation)
  processors.py         # GridLogitsProcessor   (tool-family logits-processor shape)
  generate/
    __init__.py         # sql(), cfg()          (singledispatch on model type)
    api.py              # tool-family generation adapter + GRID-owned decode loop (§4.5)
  grammar/
    spec.py             # DialectGrammar (L1) load/parse/validate
    projection.py       # RoleProjection (L2): production subsetting
    reduction.py        # useless-symbol elimination + reducedness assertion
    lexicon.py          # SchemaLexicon (L3): identifier allow-list tries / lazy DFAs
    registry.py         # GrammarRegistry + RegistrySlot: fingerprints, single-flight, namespaces
  lalr/
    compile.py          # LALR(1) table construction, conflict reporting
    tables.py           # ActionTable/GotoTable; allowed_terminals + eos_ok via virtual-stack
                        # reduce-chain simulation (normative algorithm in §6 step 1-2)
    reserve.py          # ReserveTable: token-denominated min-completion costs (§5 E4a)
  lexer/
    dfa.py              # derivative-based lexer DFAs, lazily materialized
    run.py              # LexerRun: immutable value object (§5 E7)
  trie/
    build.py            # TokenTrie from TokenizerAdapter.token_bytes (final artifact format, §2)
    walk.py             # context-independent walk + context-dependent residue split (§6)
  mask/
    producer.py         # MaskProducer: orchestrates walk, residue check, EOS gate, forced spans
    cache.py            # MaskCache: T1/T2, versioned immutable entries, namespaces
    keys.py             # cache-key derivation (OBL-KEY1)
  audit/
    log.py              # AuditLog, AuditRecord, rolling config hash (normative mix, §5 E8)
    replay.py           # audit replay against archived artifacts
  policy/
    bundle.py           # PolicyBundle: RBAC store snapshot → role shapes
    schema.py           # SchemaSnapshot: information_schema → lexicons
    semantic.py         # SemanticChecker: post-parse column-RBAC / schema checks
  models/               # thin adapters (transformers, vllm, llamacpp) re-using the protocol shapes
  samplers.py           # greedy(), multinomial() per the §4.1 sampler contract
  _reference/
    guide.py            # ReferenceGuide: trial-parse oracle (uses the same token_bytes as fast path)
    parser.py           # primary oracle: lark Earley + standard lexer configured to E7's
                        # maximal-munch discipline; lark-LALR contextual lexer as secondary
                        # cross-check only. Oracle disagreements triaged as their own bug class;
                        # G2/G3 comparisons are over terminal sequences (agreed lexing discipline)
grid_core/              # Rust crate (pyo3), from M4: the four §2 kernels
tests/                  # see §9
bench/                  # harness per companion §6; R-microharness from M4 (§10 G7)
```

## 4. Interface contract (tool-family convention)

### 4.1 Protocol shapes — `grid/protocols.py`

The normative definitions live in `grid/protocols.py`; `tests/protocols/` holds
self-contained conformance tests (expected signatures stated in the tests
themselves — no third-party sources are read or shipped).

```python
Instruction = Union[Write, Generate]
# NORMATIVE: instruction `tokens` are torch.LongTensor; GridLogitsProcessor
# additionally normalizes via torch.as_tensor(instruction.tokens) at the boundary.

class Guide(Protocol):
    initial_state: Any
    def get_next_instruction(self, state) -> Instruction: ...
    def get_next_state(self, state, token_id: int) -> Any: ...
    def is_final_state(self, state) -> bool: ...
    def copy(self) -> "Guide": ...

class Tokenizer(Hashable, Protocol):
    eos_token: str; eos_token_id: int; pad_token_id: int
    vocabulary: Dict[str, int]; special_tokens: Set[str]
    def encode(self, prompt) -> Tuple[NDArray, NDArray]: ...
    def decode(self, token_ids) -> List[str]: ...
    def convert_token_to_string(self, token: str) -> str: ...

class Sampler(Protocol):
    samples: int
    def __call__(self, next_token_logits, sequence_weights, rng): ...
    # NORMATIVE return: (next_token_ids (n_seqs,1), ancestors (n_seqs,), weights (n_seqs,))
```

`Write` carries GRID's jump-forward spans **subject to the decode-loop ownership
rules of §4.5** — a logits processor alone cannot append tokens. `Generate(tokens)`
carries the exact allowed-token tensor. **`GridGuide` never returns
`Generate(None)`** (the protocol meaning is "unconstrained"; processors handle it
defensively by skipping masking for that row — prompt tokens are excluded by
anchoring, not by unconstrained instructions).

### 4.2 `GridGuide` — extended Guide surface

Implements the protocol above plus the extended CFG-guide surface:

```python
class GridGuide:
    initial_state: GridState
    def get_next_instruction(self, state: GridState) -> Instruction: ...
    def get_next_state(self, state: GridState, token_id: int) -> GridState: ...
    def is_final_state(self, state: GridState) -> bool: ...          # == can_terminate_state
    def iter_valid_token_ids(self, state, candidate_token_ids) -> Iterator[int]: ...
    def can_terminate_state(self, state: GridState) -> bool: ...     # EOS legal here? (§6 step 2)
    def must_terminate_state(self, state: GridState) -> bool: ...    # only EOS legal here?
    def copy(self) -> "GridGuide": ...
```

**Normative CFG-mode semantics (differ from legacy internal CFG modes; conformance-tested):**
1. GRID returns the **full** exact mask every step (never a first-legal-token shortcut), so sampler parameters are meaningful under constraints; `process_logits` applies a hard in-place `masked_fill_(-inf)`.
2. Token→bytes is GRID's canonical `token_bytes` (§5 E6), used identically by the trie build, the fast path, **and** `ReferenceGuide` — never decode-diffing, which is tokenizer-family-dependent and would make G3 differentials incoherent on llama-family tokenizers.
3. A processor **must never union a multi-token `Write` span into one step's mask** — that admits out-of-order span tokens. Spans are applied one token per step (§4.5).

### 4.3 `GridLogitsProcessor` — the tool-family processor shape

- `process_logits(input_ids: 2D, logits: 2D) -> logits`; `__call__` normalizes array types (torch/numpy/list/tuple; mlx/jax via guarded imports, §12).
- `_seq_start_idx` captured on the first `process_logits` call from `len(input_ids[0])` = the **anchor** for the constrained span (companion G8). Anchoring is processor state; the Guide never sees prompt ids.
- State registry (R requirement — no Θ(n) per-step prefix hashing): `_guide_states: Dict[int, GridState]`, seeded with the empty-prefix key at construction; an unknown state is reconstructed by looking back one token. GRID keys incrementally: per batch row keep `(prev_key, n_prev)`; `new_key = splitmix64(prev_key XOR (token_id * 0x9E3779B97F4A7C15))`. Entries store `(n_generated, last_token)`; a key hit with mismatched length is treated as a miss and the state is **refolded** from the longest cached prefix (handles beam/batch reordering; rare, O(n) worst case, amortized O(1)). Collision policy: 64-bit accidental collision is detected by the stored `(n_generated, last_token)` check and treated as a miss.
- `Generate(None)` (never produced by GRID guides): skip masking for that row.
- **Lifecycle:** the adapter clones the processor with `copy.copy(self.logits_processor)` once per generation; GRID defines `__copy__(self): return self.copy()` — fresh guide (`guide.copy()`), fresh `_guide_states = {seed: initial_state}`, `_seq_start_idx = None`. Without this, sequential generations share the states dict, leak GridStates (pinning E8 stack nodes → unbounded memory), and corrupt anchoring. §9 test: two sequential `g(...)` calls share no processor state.
- Masking: `logits.masked_fill_(mask, float("-inf"))` — hard mask only (companion §1.7).

### 4.4 Entry points and adapter — how the library is called

Entry-point call shapes:

```python
import grid
from grid import generate, samplers

model = grid.models.transformers("Qwen/Qwen2.5-7B-Instruct", device="cuda")

g = generate.cfg(model, open("postgres_subset.lark").read())      # generic CFG parity mode

g = generate.sql(                                                  # policy mode
    model,
    policy=grid.policy.PolicyBundle.from_store("rbac.yaml", role="analyst"),
    schema=grid.policy.SchemaSnapshot.from_dsn("postgresql://..."),
    sampler=samplers.multinomial(temperature=0.7),
    audit=grid.audit.AuditLog(path="audit/"),
)

sql = g("List customers with more than 3 orders", max_tokens=256, seed=42)
for chunk in g.stream("...", max_tokens=256): ...
batch = g(["prompt a", "prompt b"], max_tokens=256)
```

`generate.sql`/`generate.cfg` are `functools.singledispatch` on model type, returning the tool-family adapter: `__call__(prompts, max_tokens=None, stop_at=None, seed=None, **model_kwargs)` and `.stream(...)`, with `SamplingParameters`/`GenerationParameters` dataclasses. `max_tokens` is additionally plumbed into the per-call processor/guide at construction (the reserve check needs `budget_remaining = max_tokens − n_generated`).

**`stop_at` policy (INV-OUT1 compatibility):** `generate.sql` rejects `stop_at` with `ValueError` at call time (a mid-statement stop would violate the parse-on-stop invariant). `generate.cfg` parity mode accepts it; STOP_SEQUENCE stops are excluded from INV-OUT1 and flagged in the audit seal.

### 4.5 Decode-loop ownership (normative — jump-forward's home)

A logits processor can only mask; it cannot append tokens — and external serving stacks own their decode loops. Therefore GRID defines two execution modes:

1. **GRID-owned loop** (transformers path; default for `generate.sql`/`generate.cfg` on local models): a step loop in `grid/generate/api.py`. `Write([t1..tk])` spans are appended **without forward passes**; for each appended token the loop calls `guide.get_next_state` once, registers the intermediate prefix state in `_guide_states`, and emits one `AuditRecord` (`instruction_kind=WRITE`, `mask_entry_id=None`). This is where K5's model-call savings and E15's `MAX_TOKENS_WITH_JUMP_COMPLETE` live.
2. **Processor-only mode** (vLLM and any external loop): `Write` degrades to a **singleton mask per step** — only `forced_ids[0]` is allowed; the remaining span tokens are forced one step at a time. Semantics preserved, no model-call savings. Jump-complete-at-max_tokens is unavailable here; the stop reason is downgraded and the event recorded in the audit seal. G7/G9 performance expectations are scoped per mode.

**Forced-span detection (normative):** a forced span is the maximal chain obtained by iterating §6 steps 1–9 on hypothetical successor states while the mask (excluding EOS) is a singleton, bounded by `J_max` (config, default 8). The `Write` carries exactly that token chain, so each id is in its own step's mask by construction. Byte-level jump-forward with re-tokenization (XGrammar `find_jump_forward_string`) is a recorded v2 optimization — it can produce non-canonical tokenizations that change audit records and `_guide_states` keys.

## 5. Entity catalog and state machines

Conventions: `⊳` marks the initial state; **terminal** states are underlined; transitions carry **triggers**. The single source of truth for every machine is a machine-readable transition table checked into the repo (`grid/_statecharts/*.yaml`: rows of `(state, trigger, next_state)`); the tables below are rendered from it, and §9's tests are generated from the YAML, not from this document. Any transition not in the YAML raises `IllegalTransition` (§7) — no silent defaults. For derived-status machines (E9), tests are observer-style: run generations, assert every observed `(prev, next)` pair is allowed. All fingerprints are BLAKE2b-128 over canonical serializations.

### E1. DialectGrammar (L1)
Source lark/BNF text of the SQL dialect core. Fields: `source`, `terminals`, `productions`, `ignored_terminals` (whitespace/comments — first-class, see §6), `start_symbol`, `fingerprint`.

| State | Trigger → next |
|---|---|
| ⊳DRAFT | `parse()` ok → PARSED; parse error → **INVALID(reason)** |
| PARSED | `validate()` ok → VALIDATED; undefined symbol / non-LALR construct → **INVALID** |
| VALIDATED | `freeze()` → **FROZEN** (immutable + fingerprinted) |

Invariants: FROZEN ⇒ reduced; lists left-recursive where possible (lint `L-REC01`; inherent right recursion allowed, affects only the depth bound); LALR(1)-conflict-free or conflicts explicitly resolved and recorded.

### E2. RoleProjection (L2)
Production subset + clause constraints for one role shape. Fields: `role_shape_hash`, `kept_productions`, `base_fingerprint`.

| State | Trigger → next |
|---|---|
| ⊳DECLARED | `compose(L1)` → COMPOSED; unknown production → **INVALID** |
| COMPOSED | `reduce()` → REDUCED (useless-symbol elimination — **mandatory**) |
| REDUCED | `verify()` → VERIFIED (reducedness + L(G_role) ≠ ∅); empty language → **INVALID(EMPTY_LANGUAGE)** |
| VERIFIED | register → **CACHED** |

Only REDUCED+VERIFIED reaches the LALR compiler.

### E3. SchemaLexicon (L3)
Identifier allow-lists as lexer tries. Fields: `schema_fingerprint`, `categories: Dict[terminal_name, IdentifierTrie]`.

| State | Trigger → next |
|---|---|
| ⊳DECLARED | first use → MATERIALIZING |
| MATERIALIZING | lazy DFA build ok → ACTIVE; build error → **INVALID** (`LexiconBuildError`, §7) |
| ACTIVE | schema fingerprint mismatch → DEPRECATED |
| DEPRECATED | last pinning CompiledGrammar retires → **RETIRED** |

E3 lifetime is **subordinate to E4's refcount**: streams reference lexicons only through a CompiledGrammar, so DEPRECATED lexicons stay readable until every pinning grammar retires (mirrors E4; no use-after-free on in-flight trie walks).

Invariant (**identifier composition rule**, companion §3.4): at identifier positions the mask comes from L3 trie intersection — generic-IDENT cache entries are never consulted there. Enforced structurally by E11's type-distinct keys; violation raises `IdentifierMaskBypassError` in **all** builds.

### E4. CompiledGrammar
LALR tables for one (L1, L2, L3-categories) composition. Fields: `fingerprint` (hash of component fingerprints), `action/goto` tables (**with or without default-reduction compression — decided at compile time and recorded in the artifact**; the §6 step-1 algorithm is correct either way), `terminal_dfas`, `ignored_terminals`, `version`.

| State | Trigger → next |
|---|---|
| ⊳COMPILING | success → READY; LALR conflict → **FAILED(ConflictReport)** |
| READY | superseded by newer fingerprint → DEPRECATED |
| DEPRECATED | last pinned stream ends (refcount 0) → **RETIRED** |

Construction is single-flight per fingerprint via E17. READY objects are immutable and shared.

### E4a. ReserveTable
Min-completion costs, **denominated in model tokens** (a terminal-denominated reserve under-reserves: one identifier terminal can cost many tokens). Per-terminal cost = minimal number of vocabulary tokens spelling that terminal's shortest lexeme (greedy cover over the TokenTrie; for L3 categories, over the shortest *allowed* identifier). Because costs depend on the tokenizer, the ReserveTable is a **separate artifact keyed by (grammar_fingerprint, tokenizer_fingerprint)** and referenced by CompiledGrammar — grammar identity itself stays tokenizer-independent. States: ⊳COMPUTING → **READY** | **FAILED**. G5 asserts: no reserve-stopped generation exceeds `max_tokens`.

### E5. TokenTrie
Byte trie over the vocabulary in the final artifact format (§2), built **exclusively from `TokenizerAdapter.token_bytes`** (E6). One per `tokenizer_fingerprint`; special tokens are excluded from the trie and permanently masked (EOS enters masks only via §6 step 7's explicit union). States: ⊳BUILDING → **READY** (immutable) | **FAILED** (`TrieBuildError`, §7).

### E6. TokenizerAdapter
Wraps a HF/llama tokenizer into the `Tokenizer` protocol, plus the **canonical token→bytes function**:

`token_bytes(token_id) -> bytes`, normative rules: byte-level-BPE unicode↔byte remap tables inverted (GPT-2 style); sentencepiece `▁` and BPE `Ġ` → `0x20`; byte-fallback literals `<0xNN>` → the single byte `0xNN`; the id→token reverse map is built and held by the adapter (the protocol exposes `vocabulary: str→int`). `token_bytes` is used by the trie build, the fast path, and `ReferenceGuide` — one definition, three consumers (G3 depends on this).

| State | Trigger → next |
|---|---|
| ⊳UNVERIFIED | `verify()` over `token_bytes` output: all 256 byte values reachable → VERIFIED_COMPLETE |
| | missing bytes → VERIFIED_DEGRADED (warning `W-COMPLETENESS01`; completeness guarantee formally void, soundness unaffected) |

### E7. LexerRun
**Immutable value object** (4 fields): `dfa_state: int`, `remainder: bytes` (bounded by the longest in-flight lexeme), `hypotheses: Tuple[(terminal, dfa_state), ...]`, `category_context` (identifier-category flag). `advance(bytes) -> (LexerRun, emitted_terminals)` returns a **new instance** — there is no in-place mutation anywhere (many GridStates alive concurrently in `_guide_states`/beams alias LexerRuns; mutation would corrupt siblings). It is a 4-field struct: plain copies, no persistent-tree machinery.

Step-level positions: AT_BOUNDARY ↔ MID_LEXEME (remainder ≠ ε); AMBIGUOUS ⇔ |hypotheses| > 1. Invariant **INV-LEX1**: |hypotheses| ≤ `H_max`, computed at grammar-compile time from the L1/L2 terminal-DFA product (eagerly buildable; L3 identifier categories add at most +1 hypothesis — keyword-vs-identifier — stated as a lemma with a unit test). Runtime assert as backstop → `LexerHypothesisOverflow`.

### E8. ParserStack
Persistent (immutable-node) stack: `StackNode{lalr_state, goto_symbol, parent*, depth, reserve_sum, config_hash, refcount}`. Push = new node O(1); pop = parent pointer; rollback = retained pointer O(1); nodes shared across beams/speculative branches. **Refcount is a plain field with a single invariant — a node is freed only at refcount 0** (no LIVE/SHARED state machine; beam pruning legitimately drops refcounts back to 1).

- `config_hash` — rolling, **pinned normatively for cross-implementation replay (G10)**: `H(node) = low 64 bits of BLAKE2b-128( H(parent) || u32le(lalr_state) || u32le(goto_symbol) )`; `H(root) = 0`. O(1) per push. A cross-implementation test-vector file (stack sequence → expected hashes) is checked in; both the Python path and `grid_core` must pass it. `config_hash` is **audit-only** — never used for mask/cache equality — so 2⁻⁶⁴ collision probability is the accepted policy.
- `reserve_sum` — cumulative token-denominated min-completion cost: `R(node) = R(parent) + cost(pending construct)` from E4a; the stack-top value is the termination reserve.

### E9. GridState (the Guide-protocol state object)
Frozen dataclass: `stack: StackNode`, `lexer: LexerRun`, `n_generated: int`, `prev_token: Optional[int]`, `status: Status` (memoized cache, see below).

**Status is a total pure function** `derive_status(stack, lexer, eos_consumed)` where `eos_consumed ⇔ prev_token == eos_token_id`. The stored field is a memoization of that function; `IllegalTransition` checks compare **derived** values. All statuses are O(1)-derivable — which is why FORCED is *not* a status (it needs mask cardinality, i.e. a trie walk; "forced" is an instruction-level outcome of `get_next_instruction`, §4.5) and reserve exhaustion is *not* a status (budget is session state, not grammar state; it is a processor/adapter-level trigger, §6 step 3).

| Status | Meaning | Legal next (trigger: one `get_next_state` call) |
|---|---|---|
| ⊳ACTIVE | viable, output ∉ L, EOS illegal | ACTIVE, ACCEPTING, GRAMMAR_END |
| ACCEPTING | output ∈ L (per the mid-lexeme-aware EOS rule, §6 step 2) *and* other continuations legal | ACTIVE, ACCEPTING, GRAMMAR_END, **COMPLETE** |
| GRAMMAR_END | only EOS legal by grammar | **COMPLETE** |
| **COMPLETE** | EOS consumed; final | — |
| **DEAD_END** | mask empty — must be unreachable; raises `DeadEndError` (G5 asserts zero) | — |

Self-loops are real (`ACCEPTING→ACCEPTING`: `LIMIT 1` → `LIMIT 10`; `ACTIVE→ACTIVE`: most steps). `ACTIVE→COMPLETE` is impossible (step 11 asserts EOS legality first). Pinned mapping: `is_final_state(s)` ⇔ `can_terminate_state(s)` ⇔ status ∈ {ACCEPTING, GRAMMAR_END, COMPLETE}; `must_terminate_state(s)` ⇔ status ∈ {GRAMMAR_END, COMPLETE}.

### E10. MaskCacheEntry
Immutable, versioned. Fields: `entry_id`, `key`, `payload`, `origin: SEEDED|COMPUTED`, `grammar_version`.

**Deterministic encoding (cross-implementation, G10):** payload sizes compared as `4·|accept|` vs `4·(V−|accept|)` vs `⌈V/8⌉` bytes; ties broken accept-list < reject-list < bitset; token ids sorted ascending; `entry_id = BLAKE2b-128(canonical key bytes || encoding tag || canonical payload)`. Racing writers of one key therefore produce the same `entry_id` — publish is idempotent by construction. Cross-implementation test vectors checked in.

States: ⊳(SEEDED | COMPUTED) → PUBLISHED (immutable forever) → INVALIDATED. **Rollover trigger:** `GrammarRegistry`, on registering a superseding CompiledGrammar/PolicyBundle, swaps the T2 namespace pointer and enqueues the old namespace for archival (replay against a missing archive raises `StaleArtifactError`). Entries are never deleted in place.

### E11. MaskCache
Two tiers with distinct keys and an explicit flow.

**The context-dependent split (soundness precondition for caching).** A token whose bytes cross a terminal boundary *and continue* (`'),'`, `' FROM('`, `'1;'`) has viability depending on the allowed-terminal set *after* shifting the first terminal — i.e. on the parser stack, which no (lexer, A)-keyed entry can capture. Therefore the walk kernel returns two outputs (§2): the **context-independent mask** (tokens fully resolvable within the current lexeme, or ending exactly at one terminal boundary with that terminal ∈ A) — cacheable — and the **context-dependent token list** (boundary-crossing continuations) — checked per step against the live stack via `check_context_dependent`, **never cached**. Without this split, OBL-KEY1 is violated by construction. G3/G4 corpora include multi-terminal tokens explicitly; G7 telemetry reports the residue-list size per state (companion §8.1's nesting-sweep risk).

- **T1** (per-CompiledGrammar, private): key `(lexer_product_state, remainder, allowed_terminal_signature [, schema_fingerprint at identifier positions])`. `lexer_product_state` is the hypothesis-set/product-DFA state, not a single DFA id.
- **T2** (shared across the grammar family): key `(L1 dialect fingerprint, tokenizer_fingerprint, lexer_product_state under the canonical L1 terminal numbering, remainder, allowed_terminal_signature over that numbering [, schema_fingerprint at identifier positions])`. Role projections of one dialect share the L1 terminal numbering (assigned at L1 freeze; projections subset productions, never renumber terminals), so any two roles that reach the same lexer state with the same allowed set share entries — this is what makes T2 the cross-family cache (companion §7.1) instead of a dead tier. The tokenizer fingerprint is in the key because masks are token-id-space-specific (E5: one trie per tokenizer).
- **Flow:** miss → compute → publish to T1 synchronously, T2 asynchronously; T2 hit → copy into T1. **Bounds:** T1 = per-grammar LRU with a configured entry cap; T2 = bounded map; namespace rollover is the bulk-eviction mechanism.
- Identifier-position keys are a **distinct key type** from generic-IDENT keys (cannot collide by construction); consulting a generic-IDENT entry at an identifier position raises `IdentifierMaskBypassError` in all builds.

**Soundness obligation (OBL-KEY1):** two configurations sharing a key must produce byte-identical *context-independent* masks — the key must refine the Myhill–Nerode classes of the (lexer product-DFA × allowed-terminal set × identifier-lexicon) product; the context-dependent residue is exempt because it is never cached. Verified by G4 (cache-on ≡ cache-off, including cross-role T2 hits), not assumed.

### E12. Instruction — `Write`/`Generate` (§4.1). Tokens are `torch.LongTensor` (normative contract).

### E13. GridLogitsProcessor

| State | Trigger → next |
|---|---|
| ⊳FRESH | first `process_logits` captures `_seq_start_idx` → ANCHORED |
| ANCHORED | per-step processing (self-loop); all sequence states COMPLETE → FINISHED; `finish()` from the adapter → FINISHED |
| **FINISHED** | further `process_logits` → `ProcessorReuseError` |

The adapter calls `processor.finish()` whenever the GenerationSession enters STOPPED for **any** reason (STOP_SEQUENCE and ERROR never produce COMPLETE states, so without `finish()` the single-use invariant would be silently unenforced for those stops). §9 test: reuse after a `stop_at` stop raises.

### E14. AuditLog / AuditRecord
Record: `(step, config_hash, mask_entry_id: Optional[EntryId], chosen_token, blocked_count, instruction_kind: GENERATE|WRITE|EOS, prev_record_hash)` — hash-chained. `mask_entry_id = None ⇔ instruction_kind ∈ {WRITE, EOS}`. **Every step appends a record, including each token of a Write span and the EOS step** (§6 steps 11/15) — the EOS record is the chain tail; SEALED requires it for non-error stops (otherwise G10's bit-identical replay would exclude the accepting decision). Log states: ⊳BUFFERED (lock-free ring append) → FLUSHED (async) → **SEALED** (chain head + artifact fingerprints + mode flags, e.g. processor-only downgrades, `stop_at` exclusions). Failure policy: `audit=strict` (flush failure aborts) vs `audit=best_effort` (default; `W-AUDIT01`).

### E15. GenerationSession (adapter-level)
⊳INIT → PROMPT_ENCODED → STREAMING → **STOPPED(reason)**, `reason ∈ {EOS_ACCEPT, MAX_TOKENS_WITH_JUMP_COMPLETE, STOP_SEQUENCE (cfg mode only), ERROR(exc)}`. On any STOPPED: `processor.finish()`. Invariant **INV-OUT1**: every non-ERROR, non-STOP_SEQUENCE stop parses under the same CompiledGrammar (debug: always checked; prod: sampled). `MAX_TOKENS_WITH_JUMP_COMPLETE`: when `budget_remaining ≤ reserve_sum`, the shortest legal completion is jump-forwarded (GRID-owned loop) — the event and residual truncation rate are reported metrics, never silent. In processor-only mode this downgrade is recorded in the audit seal (§4.5).

### E16. PolicyBundle / SchemaSnapshot
⊳LOADED → COMPILED (role shapes hashed, lexicons declared) → ACTIVE → **SUPERSEDED** (in-flight streams pin old versions via E4 refcounts). Fingerprints feed E4/E4a identities — no in-place mutation anywhere in the system.

### E17. RegistrySlot (GrammarRegistry single-flight)
One slot per requested fingerprint (E3 lexicons, E4 grammars, E4a reserves, E5 tries).

| State | Trigger → next |
|---|---|
| ⊳PENDING | build ok → READY; build error → FAILED(err, ttl) |
| READY | artifact superseded → evicted (slot removed; artifact lifecycle continues in E3/E4) |
| FAILED | ttl expiry or component-artifact change → slot removed (next request re-enters PENDING) |

All concurrent waiters on PENDING receive the same result or the same exception. FAILED is **negatively cached** with a TTL to prevent recompile storms on known-bad fingerprints. READY slots hold the strong refs that feed E3/E4 refcounts.

## 6. Per-token hot path (normative pseudocode)

**Processor pre-step** (E13; the Guide never sees prompt ids or budgets):
```
process_logits(input_ids, logits):
  if FRESH: _seq_start_idx ← len(input_ids[0]); ANCHORED          # anchor: constrained span only
  for each row: state ← _guide_states[key] (incremental key, §4.3; refold on miss)
  instr ← guide.get_next_instruction(state)     # steps 1-10 below; reserve trigger evaluated
                                                # here with budget_remaining = max_tokens − n_generated
  apply instr: Generate(mask) → masked_fill_(-inf outside mask); Write → §4.5 mode rules
```

**Guide level** — `get_next_instruction(state)`:
```
  1  A ← allowed_terminals(state.stack)
        # NORMATIVE (LALR default-reductions make raw rows over-approximate):
        #   A = { t : simulate(stack, t) reaches a SHIFT }, where simulate runs the reduce chain
        #   on a virtual stack of state ids (pop |rhs|, GOTO, repeat) until shift or error.
        # Cost: O(|row| × depth) worst case, amortized far less. Ignored terminals (E1) are
        # implicitly in A at every lexeme boundary. Differential unit test vs lark
        # InteractiveParser.accepts() on identical grammars (tests/lalr).
  2  eos_ok ← end-of-input simulation, mid-lexeme aware:
        #   (remainder == ε  OR  the pending lexeme is completable as exactly one winning
        #    hypothesis) AND, after virtually emitting+shifting that pending terminal on a
        #   scratch stack, ACCEPT is reachable via the reduce chain of $end.
        # (After '...FROM t', 't' is a complete IDENT awaiting maximal-munch finalization:
        #  the stack alone would wrongly say EOS is illegal — this rule is what makes
        #  ACCEPTING/can_terminate_state correct. Cases in the G2/G3 corpora.)
  3  if budget_remaining ≤ state.stack.reserve_sum:               # session-level trigger (§4.4)
        if eos_ok: return Write(tensor([eos_id]))
        return Write(tensor(shortest_completion_ids(state) + [eos_id]))
        # jump-complete (companion §3.5): per pending construct take the min-cost production
        # (E4a DP), for open lexemes the shortest trie/L3 lexeme, tokenized greedily via the
        # TokenTrie. Stop reason MAX_TOKENS_WITH_JUMP_COMPLETE. Never a bare EOS away from ACCEPT.
  4  key ← cache_key(state, A)                                    # E11; identifier positions
                                                                  # use the L3 key type
  5  entry ← T1.get(key) or T2.get(key)                           # T2 hit → copy into T1
  6  if miss: (ci_mask, cd_list) ← walk(trie_buf, state.lexer, A, l3_refs)
              publish(key, adaptive_encode(ci_mask))              # T1 sync, T2 async; E10 encoding
     else:    (ci_mask, cd_list) ← entry.payload, entry.cd_list
  7  mask ← ci_mask ∪ check_context_dependent(cd_list, state.stack)   # residue: per-step, uncached
     if eos_ok: mask ← mask ∪ {eos_id}                            # sole entry point for EOS
  8  if |mask| == 0: raise DeadEndError                           # unreachable by theorem; assert
  9  if |mask| == 1: extend to the maximal forced span (§4.5, bound J_max); return Write(span)
 10  return Generate(tensor(sorted(mask)))
```

**Guide level** — `get_next_state(state, token_id)`:
```
 11  if token_id == eos_id:
        assert can_terminate_state(state)                         # E9: only ACCEPTING/GRAMMAR_END
        audit.append(step, config_hash, None, eos_id, blocked_count, EOS, prev_hash)
        return replace(state, prev_token=eos_id, status=COMPLETE) # eos_consumed ⇒ COMPLETE
 12  bytes ← tokenizer.token_bytes(token_id)                      # E6 canonical function
 13  stack' ← state.stack; (lexer', emitted) ← state.lexer.advance(bytes)
        # 0..k terminals; ignored terminals (whitespace/comments) appear in `emitted`
        # but are dropped before step 14 (no parser advance, reserve cost 0)
 14  for t in emitted if t ∉ ignored: stack' ← lalr_advance(stack', t)
        # E8: push/pop with rolling config_hash and reserve_sum maintained per node
 15  audit.append(step, stack'.config_hash, mask_entry_id_or_None, token_id,
                  blocked_count, GENERATE|WRITE, prev_hash)       # every step, incl. Write tokens
 16  return GridState(stack', lexer', n_generated+1, token_id,
                      derive_status(stack', lexer', eos_consumed=False))
```

Cost budget per step (companion §5): steps 1–2 ≈ 0.3–1 µs typical (worst case O(|row|×depth)); 4–7 hit ≈ 1–3 µs, miss ≈ 20–50 µs @128k, plus residue check O(|cd_list|×depth); 13–14 amortized O(1), worst case O(nesting depth); 15 ≈ 0.1–0.3 µs. Nothing in the loop reads anything proportional to *n* (this is why §4.3's incremental state key exists).

**Batch scheduling contract (serving):** masks are computed on CPU overlapped with the GPU forward pass. If a request's mask is not ready at sampling time (worst-case cold miss ≈ full trie walk, ~4 ms), that request is **skipped for the current scheduling round and rejoins the next step** — co-batched requests are never stalled, and an approximate mask is never substituted (no over-approximating deadline fallback exists in v1, by design; §13). Gate G8's adversarial arm verifies this contract.

*Realization (M6, vLLM 0.24 V1): the overlap half is `grid/serving/prefetch.py` — `GridGrammarSession.accept_tokens` schedules the successor state's mask on a worker pool ONLY when that mask is not already T1-warm (`GridGuide.is_mask_warm`); the warm steady state never touches the pool (unconditional scheduling serialized every step behind the single worker's queue — LESSONS 6.5). The cold walk runs with the GIL released (kernel v4 `walk_raw` detach), overlapping the scheduler's remaining CPU work for the step, and `fill_bitmask` waits only for the un-hidden residual (measured, reported in prefetcher stats). The warm fill itself is kernel v5 `fill_bits`: the whole bitmask row (pre-packed per-entry ci bit words ++ live CD-pass bits ++ EOS) written into vLLM's row buffer in one FFI call — µs-scale, no id materialization. Request copies share the template's `MaskProducer` (one kernel, one entry-registration space, one T1 cache; per-request state lives in `GridState`/sessions). vLLM 0.24 exposes no per-step defer hook for a RUNNING request, so the literal "skip this round, rejoin next" is delegated to vLLM's own async grammar-executor admission gating (WAITING requests are held without stalling the batch) while GRID keeps the mask warm by the time the scheduler asks; for a mid-stream cold miss the batch therefore waits the un-hidden residual of that walk — the adversarial G8 arm measures exactly this residual. The single-flight half is `grid/serving/singleflight.py` (E17): one build per fingerprint, N waiters share the result or the same exception, FAILED negatively cached with a TTL.*

## 7. Error taxonomy

| Exception | Raised when | Caller contract |
|---|---|---|
| `GrammarInvalid(reason)` | E1/E2 validation | fix grammar/policy; never at generation time |
| `LALRConflictError(report)` | E4 compile | grammar author fixes; report lists conflict states/lookaheads |
| `EmptyLanguageError` | E2 verify: L(G_role) = ∅ | policy misconfiguration; refuse role |
| `LexiconBuildError` | E3 MATERIALIZING → INVALID | policy/schema author fixes (bad identifier set, encoding) |
| `TrieBuildError` | E5 BUILDING → FAILED | tokenizer-adapter defect; file, don't catch |
| `LexerHypothesisOverflow` | INV-LEX1 breach at runtime | compile-side bug (H_max wrong); file, don't catch |
| `DeadEndError` | empty mask at step 8 | **bug by theorem** — abort generation, dump state; G5 = 0 occurrences |
| `IdentifierMaskBypassError` | generic-IDENT cache entry consulted at an identifier position (E11 key-type guard, **all** builds) | always a bug — abort; G6 injects the condition and asserts it fires |
| `ProcessorReuseError` | E13 FINISHED reuse (incl. after stop_at/ERROR stops via `finish()`) | construct a new generator |
| `AuditFlushError` | E14 strict mode | abort (strict) / warn `W-AUDIT01` (best-effort) |
| `IllegalTransition(entity, from, to)` | any §5 machine (checked against the YAML statecharts) | always a bug; never catch in library code |
| `StaleArtifactError` | audit replay against a missing archived namespace | archival misconfiguration |
| `ValueError("stop_at unsupported in sql mode")` | §4.4 stop_at policy | caller uses cfg mode or drops stop_at |

Warnings (never raised): `W-COMPLETENESS01` (E6 degraded tokenizer), `W-AUDIT01` (best-effort flush failure).

Design rule: **generation-time exceptions are bugs** (DeadEnd, IllegalTransition, HypothesisOverflow, IdentifierMaskBypass). Everything user-fixable fails at compile/verify/call time. This is the single biggest debug-minimization lever: the state machines are checked at the boundaries, so errors surface at construction, not mid-stream.

## 8. Concurrency and immutability model

- **Immutable + shared:** FROZEN grammars, READY CompiledGrammar/ReserveTable/TokenTrie, PUBLISHED cache entries, StackNodes, **LexerRun values** (E7 — advance returns a new instance; the only mutation anywhere is constructing the next value).
- **Per-stream mutable:** the GridState *chain* (each state itself frozen), AuditLog buffer, the processor's `_guide_states` dict and key cursors.
- **Single-flight:** all artifact builds via E17 RegistrySlots.
- **Cache races:** content-hash idempotent publish (E10); no CAS needed; readers never block on writers.
- **Speculative decoding / beams:** rollback = retained StackNode pointer + LexerRun value + audit chain cursor, all O(1). Beam/batch reordering is absorbed by the state registry's refold path (§4.3).

## 9. Testing strategy

```
tests/
  protocols/         # G0: self-contained conformance tests — expected protocol
                     # signatures stated in the tests themselves; array-type
                     # normalization; tensor contract (.tokens.to(device) works on
                     # every instruction a GRID guide emits)
  grammar/           # E1-E3 machines; reduction property tests; statechart-YAML-generated tests
  lalr/              # tables vs lark-LALR; allowed_terminals vs InteractiveParser.accepts();
                     # reduce-closure; E4a reserve DP (token-denominated) incl. tightness
  lexer/ trie/       # unit + property (INV-LEX1, H_max lemma, byte coverage, token_bytes rules)
  mask/              # G3/G4 differentials: fast ≡ reference; cache-on ≡ cache-off (incl. cross-
                     # role T2); context-dependent split; adaptive_encode + config_hash vectors
  guide/             # guide-level differentials and instruction semantics over the
                     # committed four-tokenizer matrix (byte-level BPE, sentencepiece,
                     # tiktoken-style, byte-level)
  sql_samples/       # corpus dir: valid/, invalid/, per-dialect
  processors/        # anchoring, __copy__ isolation (two sequential calls share no state),
                     # finish()/reuse-after-stop_at, Generate(None) row-skip, array types
  generate/          # integration: transformers tiny-model smoke; GRID-owned loop Write spans
                     # (intermediate _guide_states registration, per-token audit records)
  audit/             # chain integrity incl. EOS tail record; replay (G10); cross-impl vectors
  policy/            # RBAC projection, adversarial + property suites (G6), SemanticChecker
```

Techniques: **property-based** (hypothesis) for reduction/masks/keys/encodings; **differential** against `_reference/` and lark (advisory: sqlglot parse, optional dockerized Postgres `EXPLAIN`); **fuzzing** random token walks under the mask (must never wedge); **statechart tests generated from `grid/_statecharts/*.yaml`** — the machine-readable transition tables are the source of truth (§5): explicit-trigger machines get one allowed-transition test per row plus IllegalTransition probes for unlisted pairs; derived-status machines (E9) get observer-style tests (run generations, assert every observed transition is allowed).

## 10. Verification gates

Each gate is a CI job; a milestone may not start until its entry gates are green. "Oracle" = `grid/_reference/` unless stated. Perf gates (G7/G8/G9) run on a **declared cloud runner** — a named provider instance type + image (e.g. Lambda 1×H100 PCIe, Lambda Stack 24.04), recorded in every report's host label — with committed seed lists; the absolute per-step budget is recorded once from ITL measured on the same declared hardware (e.g. 1×H100, Qwen2.5-7B). *(Amended 2026-07: bare-metal pinning — isolated cores, performance governor — dropped from the plan; cross-engine ratios proved host-invariant while absolute constants carry the host label.)*

| Gate | Verifies | Pass criteria |
|---|---|---|
| **G0 Interface conformance** | §4 protocols | self-contained signature tests green (expected shapes stated in the tests); base-processor array-type tests green; tensor-contract assertions green (`.tokens.to(device)` works on every GridGuide instruction); mypy strict |
| **G1 Grammar pipeline** | E1–E4a | PG-subset L1 compiles conflict-free; property test over random production subsets: 100% reducedness, empty-language rejected; fingerprints deterministic across processes; ReserveTable token-denomination verified on fixtures (identifier terminals cost >1 token where true) |
| **G2 Viable-prefix oracle** | parser = Prefix(L) recognizer | 10k corpus sentences: every prefix accepted; 10k mutated: rejected exactly at first invalid terminal; mid-lexeme EOS cases (step-2 rule) correct; differential vs lark over terminal sequences |
| **G3 Mask exactness** | walk + split | fast-path (ci_mask ∪ residue) ≡ oracle trial-parse mask, bit-exact, over ≥10⁵ configs from **grammar-guided random walks under the reference mask** with counter-asserted quotas: ≥20% identifier positions (allowed *and* forbidden, multi-byte), ≥10% remainders mid-UTF-8-codepoint, ≥20% mid-lexeme, ≥5% ambiguous hypothesis sets, **≥10% multi-terminal boundary-crossing tokens, whitespace/comment-spanning tokens included**; committed four-tokenizer matrix (§9); byte-fallback ⇒ zero empty masks at viable states |
| **G4 Cache soundness** | E10/E11, OBL-KEY1 | cache-on ≡ cache-off over randomized replays **including cross-role T2 hits** (two role projections, one dialect) and context-dependent residues; namespace rollover: zero stale hits; racing publish: single entry_id; encoding/config-hash cross-impl vectors green |
| **G5 End-to-end S/C/T** | soundness, completeness, termination | 10k seeded generations, pinned model+tokenizer (byte-fallback BPE ≥100k vocab, e.g. Qwen2.5-0.5B-Instruct) **plus forced-random-walk arm** (uniform over mask, EOS suppressed until length ≥ L, budgets forcing reserve stops): 100% outputs parse under own CompiledGrammar (**binding**; sqlglot/EXPLAIN advisory with triaged log), EOS only at ACCEPT, `DeadEndError` = 0, every jump-complete stop parses and ends at ACCEPT, **no reserve-stopped generation exceeds max_tokens**; coverage counters gate: nesting ≥ D, ≥ k reserve stops, ≥ k multi-byte-identifier events; reserve tightness: sampled `reserve_sum` == oracle shortest-completion token count |
| **G6 RBAC** | E2/E3/SemanticChecker | (a) model-independent mask property test: fuzzed walks to identifier positions — no token sequence completes a forbidden identifier at a lexeme boundary (incl. forbidden-is-prefix-of-allowed: `users` vs `users_public`) — violations **exactly 0**; (b) adversarial prompt suite (secondary); (c) `IdentifierMaskBypassError` injection test fires; (d) column-violation fixtures: SemanticChecker flags 100% |
| **G7 Performance (R)** | companion §6.1 | on the **M4 R-microharness** (recorded/synthetic token-stream replay, no model): 95% CI of mask-latency-vs-position slope has half-width ≤ ε and upper bound ≤ ε (ε = 0.1 µs/1k tokens) at n=16k under the nesting sweep, N ≥ 20 seeded runs; p50 cache-hit < 10 µs; p99 miss < absolute step budget; warm-cache hit rate ≥ 90% reported per nesting depth; **context-dependent residue size reported per nesting depth**; total guard cost linear fit R² > 0.99; cross-role T2 hit factor > 0 |
| **G8 Serving** | batch/overlap | full harness (M6): batch 1/8/32 heterogeneous grammars, per-step p99 within budget; TPOT overhead <2% vs unconstrained @batch 32; TTFT: cold role+schema specialize < 50 ms, warm < 5 ms (companion §6.2); **adversarial cold-miss arm** (cache cleared, maximal identifier position, injected into batch-32): co-batched TPOT degradation < 5% and max step delay bounded via the §6 skip-a-round contract; concurrent cold start: single-flight (1 build, N waiters, same error on FAILED) |
| **G9 Benchmark parity** | bench harness | **binding:** XGrammar + XGrammar-2, llguidance, outlines-current, SynCode, GBNF + unconstrained arm execute on Spider-subset & JSONSchemaBench sample; report auto-generated with the explicit metric list: syntax-validity %, EX delta vs unconstrained, mask-latency percentiles, TTFT, TPOT, throughput @batch, truncation rate, cache telemetry; ablation arms: cache-off, write-back-off, audit-off, jump-forward-off; fairness protocol (companion §6) satisfied. **Tracked KPI (non-gating):** GRID within 2× XGrammar p50 mask latency |
| **G10 Audit replay** | E14 | replay **every step of ≥1,000 generations spanning ≥1 namespace rollover**: bit-identical masks (EOS and Write records included); tamper property test (random record, random field, ≥10³ trials): 100% detection. **G10a** (chain integrity + basic replay smoke) gates M3 exit |

## 11. Milestones

| M | Deliverable | Entry gates | Exit gates |
|---|---|---|---|
| M0 | repo scaffold, protocol shapes + conformance tests, CI, statechart YAML + test generator, `_reference/` guide on toy grammar | — | G0 |
| M1 | grammar pipeline (E1–E4a incl. ReserveTable), PG-subset dialect, policy projections | G0 | G1, G2 |
| M2 | lexer + trie (final artifact format) + MaskProducer with context-dependent split, GridGuide/Processor, GRID-owned decode loop | G1, G2 | G3 |
| M3 | MaskCache T1/T2 + audit log, end-to-end generate() with transformers | G3 | G4, G5, G10a |
| M4 | `grid_core` Rust kernels (the four §2 symbols) + **R-microharness**; same tests rebind | G5 | G3/G4 re-run vs Rust, G7 |
| M5 | RBAC suite + SemanticChecker + policy pipeline hardening | G5 | G6, G10 |
| M6 | vLLM backend (processor-only mode, bitmask kernel, overlap, skip-a-round), full bench harness | G7 | G8, G9 |

M5 and M6 may run in parallel. **Release gate R0 = all of G0–G10 green** — no ship while any gate is red, regardless of milestone bookkeeping.

## 12. Dependencies and pinning

- Runtime: `torch`, `numpy`; `lark` (grammar authoring/validation and test oracles only — the runtime parser is GRID's own LALR engine).
- Dev/test: `pytest`, `hypothesis`, `pytest-benchmark`, `sqlglot` (advisory oracle); `jax`/`mlx` optional (guarded imports in the array-type tests).
- **Interfaces:** the tool-family protocol shapes are defined normatively in `grid/protocols.py` and conformance-tested by self-contained tests under `tests/protocols/`. No external constrained-decoding package is a dependency of GRID at any time; external engines appear only inside `bench/` comparison harnesses.
- `grid_core`: Rust ≥1.75, pyo3, maturin (from M4).

## 13. Decision log and risk traceability

| Item | Decision / gate coverage |
|---|---|
| Nesting-depth context-dependent growth (companion §8.1) | context-dependent split (E11) makes it a **correctness non-issue**; residual *cost* risk gated by G7's per-depth residue-size and hit-rate reporting; fallback (widen keys / more runtime checks) budgeted in M4 |
| Leo/Earley subtleties (companion §8.2) | avoided in v1 — LALR(1) only; Earley fallback deferred until a non-LALR dialect construct forces it |
| Lexer hypothesis bound (companion §8.3) | H_max computed at compile from the L1/L2 terminal-DFA product; L3 adds ≤ +1 (lemma + unit test); runtime assert backstop; G3 fuzz |
| Batch tail (companion §8.4) | no approximate fallback in v1; §6 skip-a-round contract; G8 adversarial cold-miss arm |
| LALR spurious reduces / default reductions | normative virtual-stack simulate() for A and eos_ok (§6 steps 1–2); G5 EOS-only-at-ACCEPT |
| Column RBAC not mask-enforceable (companion §4.6) | SemanticChecker + G6(d); marketed scope = verb/table |
| Distribution faithfulness (GAD/CRANE, companion §4.5/§6) | **deferred post-v1**; bench reports EX delta vs unconstrained; CRANE ablation dropped from G9's list |
| BIRD dataset (companion §6) | **deferred post-v1**; Spider-subset + JSONSchemaBench binding for G9; BIRD tracked for the post-v1 bench |
| Byte-level jump-forward with re-tokenization | **deferred v2**; v1 forced spans = singleton-mask chains (§4.5), bounded J_max |
| Reserve denomination | model tokens (E4a); ReserveTable artifact keyed (grammar_fp, tokenizer_fp); G5 max_tokens assertion |
| `stop_at` | rejected in sql mode; cfg mode: excluded from INV-OUT1, flagged in audit seal |
| jax in the array-type tests | guarded import (decided at M0) |
