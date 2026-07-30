# Choosing a structured-outputs backend: when GRID, when the default

This is the user-facing decision document for serving-stack integrations
(written for the vLLM structured-outputs backend RFC; the criteria apply to
any stack with a pluggable grammar backend). It answers one question: when
should a user pick GRID over just accepting the default backend?

All numbers: full JSONSchemaBench (11,306 real-world schemas), one machine,
current versions (GRID 0.4.0, llguidance 1.7.6, XGrammar 0.2.3), identical
runner semantics. Per-schema statuses are committed in the GRID repo
(`bench/RESULTS-jsonschemabench-v0.4.0rc1.md`).

## Pick GRID when any of these is true

**1. A valid request failing is expensive for you.**
Valid-instance rejection is the failure mode users see as "structured outputs
are broken": the schema is right, the output would be right, the engine
blocks it mid-stream. On the full corpus: XGrammar rejects valid instances on
427 schemas; GRID on 3. If your schemas come from users or third parties
rather than a curated set, this is the row that decides.

**2. You need to know what was NOT enforced.**
Every engine has constraints it cannot express in a token mask. The default
engines either refuse the schema outright or accept silently. GRID returns,
per compiled schema, the names of the constraints it accepted but could not
enforce (`recorded`), so your post-validation checks exactly those and
nothing else. In regulated or audited pipelines this is the difference
between "we validate everything twice" and "we validate the named residue."
Strict mode (`strict=True`) converts records into hard errors if you prefer
the refusal contract.

**3. You cannot afford a compile that never comes back.**
GRID 0.4.0 is, on this corpus, the only engine configuration we have measured
where zero schemas fail to terminate: pathological inputs hit deterministic
construction budgets and come back as declared declines in bounded time, with
machine-independent fire points. A serving process handling arbitrary
customer schemas never wedges a worker on compilation.

**4. Your deployment redeploys.**
With the artifact store enabled, previously-seen schemas reload instead of
recompiling: warm-hit medians of ~17-23ms for schema families whose cold
compiles are seconds. Cold-start on the *first* deployment is unchanged (the
store never fakes a first compile).

## Pick the default (XGrammar) when

- Your schemas are a small curated set you have already verified compile and
  behave correctly and on a fixed set you can test your way past both silent-accept
  risks.
- You need the absolute fastest median compile on huge schema churn and
  llguidance's declare-rate (16% of this corpus) is acceptable — then
  llguidance, not GRID or XGrammar, is the right pick; its 0.7ms average
  TTFM is the reference point we openly chase.


## Integration status

- jsonschemabench/MaskBench engine: PR open
  (guidance-ai/jsonschemabench#15), adapter verified under the unmodified
  upstream runner.
- vLLM: patch-site integration implemented behind `GRID_JUMP` and the
  processor plumbing (`grid/models/vllm_processor.py`); GPU-box serving
  validation is the remaining stamp before the RFC. This document is the
  RFC's centerpiece per maintainer guidance ("the criteria a user should
  have in mind when choosing grid over just accepting the default").
- llama.cpp / SGLang / others: see `docs/integrations-plan.md`.
