# Platform integrations: plan

Goal: GRID available wherever open-weight models are served with pluggable
constrained decoding. Competitors (llguidance, XGrammar, Outlines) each ship
in several stacks; every additional platform is distribution, credibility,
and a fairness surface where our numbers reproduce under someone else's
runner. Ordered by expected reach per unit of work.

## 1. vLLM (in flight)

- Status: processor integration implemented (`grid/models/vllm_processor.py`,
  jump-forward patch sites); `docs/choosing-a-backend.md` is the RFC
  centerpiece the maintainer asked for. Remaining before RFC: the GPU-box
  serving stamp (S1/S2 runtime validation on H100, Lambda workflow) and a
  quiet-machine warm-store retake.
- Sequencing: post-v0.4.0 numbers are in; RFC opens when the serving stamp
  lands. The jsonschemabench PR (#15) stays the visible precursor; if it and
  the RFC both sit silent for two weeks, one polite ping to the structured
  outputs maintainers, then proceed on the other platforms regardless.

## 2. SGLang

- Why second: the same Python backend-slot shape as vLLM (xgrammar and
  llguidance backends already exist to copy the interface from), large
  serving user base, and jump-forward decoding originated there - our S1 API
  maps naturally.
- Work: a `grid` grammar-backend module implementing their
  `BaseGrammarObject` (compile, mask, accept-token, jump-forward hooks),
  wired through their backend registry; reuse the vLLM processor's session
  discipline. Estimate M. Gate: their unit suite + a 50-schema smoke of ours
  through their engine.

## 3. llama.cpp

- Why: the largest local-inference reach; llama.cpp ships its own GBNF
  grammars and llguidance is already integrable as an optional sampler,
  so a grammar-engine slot exists in spirit.
- Approach options, in order of preference:
  a. **Sampler hook via the C API** (`llama_sampler` chain): a thin C ABI
     around grid_core (the kernel is already Rust with a C-friendly FFI
     surface; the Python layer's compile pipeline runs offline and ships the
     kernel a compiled artifact). Grammar compilation happens out-of-process
     (CLI: schema -> .gridc blob); the sampler consumes the blob. This keeps
     llama.cpp dependency-free at build time (dlopen the grid kernel) and
     matches how llguidance integrates.
  b. **GBNF export** (lossy: no constrained terminals, no recorded set) -
     rejected as primary path; acceptable only as a compatibility shim with
     the honesty contract downgraded and documented.
- Work: L (C ABI on grid_core, artifact format freeze, upstream PR with
  sampler + docs). Depends on: kernel arena serialization (S3's reserved
  namespace becomes the .gridc format). Gate: llama.cpp's server JSON-schema
  tests + our corpus smoke through their `llama-server`.

## 4. Typed-pipeline frameworks (DSPy, Instructor) — client-side

- Why: these frameworks declare output types *before* the request — DSPy
  signatures and pydantic models carry the enum/Literal/shape of every
  output field statically — and they already route derived JSON schemas to
  provider structured outputs where available (DSPy's JSONAdapter). That is
  the ideal GRID workload: few schemas × many calls (compile once, warm
  path every call, artifact store across restarts), and the framework's
  parse-retry machinery goes dead because typed fields cannot arrive
  malformed.
- Work: S, no engine changes. An adapter subclass that maps signature ->
  `model_json_schema()` -> `compile_json_schema()` client-side and sends
  the `.grid` source as the request's grammar constraint to any
  GRID-enabled server; keep the framework's own validation scoped to
  `recorded` (usually empty — pydantic-derived schemas are the corpus's
  easy end); `strict=True` at program build makes an unenforceable
  signature a build error rather than a runtime surprise.
- Gate: a DSPy program smoke (signatures with Literal/enum, nested models,
  optional fields) through a GRID-backed vLLM — zero parse failures across
  seeds, recorded-residue plumbing exercised, one optimizer run to confirm
  rollouts never burn tokens on malformed outputs.

## 5. TGI / Ollama / mistral.rs (watch list)

- TGI: guidance module is Outlines-based; a backend abstraction may land
  with their v3 refactor - revisit then.
- Ollama: wraps llama.cpp; inherits #3 for free once merged there.
- mistral.rs: Rust-native, could consume grid_core directly - opportunistic,
  after the C ABI exists.

## Cross-cutting prerequisites

- **Kernel C ABI + artifact format** (unlocks 3 and 4): freeze the arena
  blob format grid_core v8 already defines internally, version it under the
  store's `kernel_fingerprint` discipline.
- **Conformance kit**: a platform-agnostic 50-schema smoke (schemas + valid/
  invalid instances + expected outcomes) any integration runs in its own CI,
  so every platform's GRID build proves the honesty contract locally.
- **Fairness posture**: for each platform PR, the accompanying note names the
  incumbent engines' strengths on that platform (as the jsonschemabench PR
  and README do) - the credibility pattern that has worked.
