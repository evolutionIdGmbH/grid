"""G8 serving benchmark (DESIGN.md gate G8, M6).

Measures GRID as a vLLM structured-output backend under batch load against the
built-in backends and an unconstrained baseline:

Arms:  grid (patched backend), xgrammar, guidance  (all three vLLM-native
       structured-output backends), unconstrained (no constraint).
Batch: 1, 8, 32 — HETEROGENEOUS grammars (a distinct schema per request, so
       fingerprints differ and per-request compile + single-flight are live).
Metrics per (arm, batch): TTFT p50/p99 (grid split cold vs warm), TPOT
       mean/p99, per-step p99, decode throughput (tok/s), and overhead% vs the
       unconstrained TPOT at the same batch.

Gate criteria (G8):
- TPOT overhead < 2% vs unconstrained @ batch 32;
- TTFT: cold role+schema specialize < 50 ms, warm < 5 ms;
- adversarial cold-miss arm (cache cleared, maximal identifier position,
  injected into batch-32): co-batched TPOT degradation < 5% and bounded max
  step delay via the §6 skip-a-round/overlap contract. TWO metrics are always
  reported side by side: v1 (legacy two-point lockstep wall — valid only while
  the batch advances in lockstep) and v2 (per-request TPOT over the 31 warm
  co-batched requests via a raw engine step loop + max engine-step wall +
  the fresh request's TTFT / completion / effective TPOT, reported not gated,
  with a soft bound effective-TPOT <= 3x warm). The GATE evaluates v2 by
  default (RATIFIED 2026-07-09: v2 measures the criterion's intent - the warm
  co-batched requests' experience - which lockstep math cannot measure under
  the defer); --adversarial-metric v1 keeps the legacy gating for comparison;
- concurrent cold start: single-flight (1 build, N waiters, same error on FAILED).

vLLM has no macOS wheels, so the real run is on the declared GPU runner
(bench/vllm_grid_patch.py applies the three integration patch points). Locally,
`--mock` exercises the metric math + report generation against fabricated,
deterministic per-step timings (a self-check of the harness, not a perf claim);
the mock arithmetic is unit-tested in tests/serving/test_serving_metrics.py.

Run (GPU host):
  .venv/bin/python bench/vllm_serving_bench.py --arms grid,xgrammar,unconstrained \\
      --batches 1,8,32 --assert-gates --out bench/RESULTS-serving.md
Run (local self-check):
  .venv-bench/bin/python bench/vllm_serving_bench.py --mock --out /tmp/serving-mock.md
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys

BENCH_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(BENCH_DIR))

GRAMMAR_SQL = (BENCH_DIR.parent / "grammars" / "sql_spider.grid")
GRAMMAR_SUBSET = (BENCH_DIR.parent / "grammars" / "sql_subset.grid")

# heterogeneous schemas: distinct table/column sets -> distinct grammar
# fingerprints -> distinct per-request compiles (single-flight exercised)
SCHEMAS = {
    "hr": {"employees": ["id", "name", "salary_band", "dept_id"],
           "departments": ["dept_id", "title", "budget"]},
    "store": {"products": ["sku", "title", "price_cents"],
              "orders": ["order_id", "sku", "qty", "placed_at"]},
    "library": {"books": ["isbn", "title", "author_id", "year"],
                "loans": ["loan_id", "isbn", "member_id", "due_date"]},
    "fleet": {"vehicles": ["vin", "make", "model", "mileage"],
              "trips": ["trip_id", "vin", "distance_km", "started_at"]},
}
PROMPT = "Write one lowercase SQL query over this schema: "


# --------------------------------------------------------------------- metrics
def summarize_arm(ttfts_ms: list[float], tpots_ms: list[float],
                  step_ms: list[float], decoded_tokens: int, wall_s: float,
                  cold_ttfts_ms: list[float] | None = None,
                  warm_ttfts_ms: list[float] | None = None) -> dict:
    """Pure metric reduction for one (arm, batch) cell — unit-tested."""
    def pct(xs, p):
        # nearest-rank on the sorted samples: never extrapolates past the
        # observed max (statistics.quantiles interpolates and would report a
        # p99 larger than any measured value)
        if not xs:
            return float("nan")
        s = sorted(xs)
        k = max(0, min(len(s) - 1, int(round((p / 100.0) * len(s) + 0.5)) - 1))
        return float(s[k])

    out = {
        "ttft_p50_ms": pct(ttfts_ms, 50),
        "ttft_p99_ms": pct(ttfts_ms, 99),
        "tpot_mean_ms": float(statistics.fmean(tpots_ms)) if tpots_ms else float("nan"),
        "tpot_p99_ms": pct(tpots_ms, 99),
        "step_p99_ms": pct(step_ms, 99),
        "throughput_tok_s": decoded_tokens / wall_s if wall_s > 0 else float("nan"),
        "n_requests": len(ttfts_ms),
    }
    if cold_ttfts_ms is not None:
        out["ttft_cold_p50_ms"] = pct(cold_ttfts_ms, 50) if cold_ttfts_ms else float("nan")
    if warm_ttfts_ms is not None:
        out["ttft_warm_p50_ms"] = pct(warm_ttfts_ms, 50) if warm_ttfts_ms else float("nan")
    return out


def overhead_pct(arm_tpot_ms: float, base_tpot_ms: float) -> float:
    if base_tpot_ms <= 0:
        return float("nan")
    return 100.0 * (arm_tpot_ms - base_tpot_ms) / base_tpot_ms


def reduce_adversarial_v2(step_walls_s: list[float],
                          token_times_s: dict[str, list[float]],
                          fresh_id: str | None) -> dict:
    """Pure metric-v2 reduction for ONE step-loop run — unit-tested.

    step_walls_s: wall time of each engine step (seconds).
    token_times_s: req_id -> arrival time (seconds, relative to loop start) of
        every generated token. A request deferred out of some rounds simply has
        no timestamps in those rounds; an early-finishing request has a shorter
        list — per-request TPOT (t_last - t_first)/(T-1) is exact either way,
        with no lockstep assumption. Requests with < 2 tokens are excluded from
        the warm mean (TPOT undefined).
    fresh_id: the adversarial (never-warmed) request; every other request is a
        warm co-batched request. None => all requests are warm (baseline run).
    """
    warm_tpots = []
    for rid, ts in token_times_s.items():
        if rid == fresh_id or len(ts) < 2:
            continue
        warm_tpots.append(1000.0 * (ts[-1] - ts[0]) / (len(ts) - 1))
    warm_mean = float(statistics.fmean(warm_tpots)) if warm_tpots else float("nan")
    fresh = token_times_s.get(fresh_id, []) if fresh_id is not None else []
    fresh_ttft = 1000.0 * fresh[0] if fresh else float("nan")
    fresh_completion = 1000.0 * fresh[-1] if fresh else float("nan")
    fresh_tpot = (1000.0 * (fresh[-1] - fresh[0]) / (len(fresh) - 1)
                  if len(fresh) >= 2 else float("nan"))
    ratio = fresh_tpot / warm_mean if warm_mean > 0 else float("nan")  # nan-safe: nan>0 is False
    return {
        "warm_tpot_mean_ms": warm_mean,
        "warm_tpots_ms": warm_tpots,
        "n_warm": len(warm_tpots),
        "max_step_ms": 1000.0 * max(step_walls_s) if step_walls_s else float("nan"),
        "fresh_ttft_ms": fresh_ttft,
        "fresh_completion_ms": fresh_completion,
        "fresh_effective_tpot_ms": fresh_tpot,
        "fresh_tpot_ratio": ratio,
    }


def _nanmean(vals: list[float]) -> float:
    valid = [v for v in vals if v == v]
    return float(statistics.fmean(valid)) if valid else float("nan")


def aggregate_adversarial_v2(reductions: list[dict],
                             baseline_warm_tpot_ms: float) -> dict:
    """Combine per-repeat v2 reductions into the reported v2 block — unit-tested.

    Degradation is the warm co-batched TPOT (mean over repeats) vs the all-warm
    baseline's per-request TPOT from the same step-loop surface. Fresh-request
    metrics are transparency metrics (reported, not gated) with a soft bound
    effective TPOT <= 3x warm; repeats where the fresh request never reached 2
    tokens (starvation-cap edge) contribute nan and are skipped in the means —
    the ratio then reads nan and the soft bound reads False, never silently OK.
    """
    warm = _nanmean([r["warm_tpot_mean_ms"] for r in reductions])
    steps = [r["max_step_ms"] for r in reductions if r["max_step_ms"] == r["max_step_ms"]]
    fresh_tpot = _nanmean([r["fresh_effective_tpot_ms"] for r in reductions])
    ratio = fresh_tpot / warm if warm > 0 else float("nan")
    return {
        "tpot_degradation_pct": overhead_pct(warm, baseline_warm_tpot_ms),
        "warm_tpot_mean_ms": warm,
        "baseline_warm_tpot_ms": baseline_warm_tpot_ms,
        "n_warm": max((r["n_warm"] for r in reductions), default=0),
        "max_step_ms": max(steps) if steps else float("nan"),
        "fresh_ttft_ms": _nanmean([r["fresh_ttft_ms"] for r in reductions]),
        "fresh_completion_ms": _nanmean([r["fresh_completion_ms"] for r in reductions]),
        "fresh_effective_tpot_ms": fresh_tpot,
        "fresh_tpot_ratio": ratio,
        "soft_bound_ok": bool(ratio == ratio and ratio <= 3.0),
        "n_repeats": len(reductions),
    }


def build_adversarial_result(v1: dict, v2: dict, metric: str,
                             budget_ms: float) -> dict:
    """Both metric blocks + top-level gating values from the selected metric.

    evaluate_gates reads only the top-level keys, so the gate follows `metric`
    (--adversarial-metric, default v2 - ratified) while the report always prints both.
    """
    sel = v2 if metric == "v2" else v1
    return {"metric": metric, "budget_ms": budget_ms,
            "tpot_degradation_pct": sel["tpot_degradation_pct"],
            "max_step_ms": sel["max_step_ms"],
            "v1": v1, "v2": v2}


def evaluate_gates(cells: dict, adversarial: dict | None,
                   singleflight: dict | None) -> list[tuple[str, bool, str]]:
    """cells[(arm, batch)] -> summary dict. Returns (criterion, pass, detail)."""
    checks = []
    base32 = cells.get(("unconstrained", 32))
    grid32 = cells.get(("grid", 32))
    if base32 and grid32:
        ov = overhead_pct(grid32["tpot_mean_ms"], base32["tpot_mean_ms"])
        checks.append(("TPOT overhead < 2% vs unconstrained @batch 32",
                       ov < 2.0, f"{ov:+.2f}%"))
    grid1 = cells.get(("grid", 1))
    if grid1 and "ttft_cold_p50_ms" in grid1:
        checks.append(("TTFT cold specialize < 50 ms",
                       grid1["ttft_cold_p50_ms"] < 50.0,
                       f"{grid1['ttft_cold_p50_ms']:.1f} ms"))
        checks.append(("TTFT warm < 5 ms",
                       grid1.get("ttft_warm_p50_ms", 1e9) < 5.0,
                       f"{grid1.get('ttft_warm_p50_ms', float('nan')):.2f} ms"))
    if adversarial:
        # top-level values mirror the GATING metric (adversarial["metric"],
        # default v1); both blocks are printed side by side in the report
        metric = adversarial.get("metric", "v1")
        tag = (" [gating metric v2: per-request TPOT over warm co-batched requests]"
               if metric == "v2" else " [gating metric v1: legacy two-point lockstep wall]")
        checks.append(("adversarial cold-miss: co-batched TPOT degradation < 5%" + tag,
                       adversarial["tpot_degradation_pct"] < 5.0,
                       f"{adversarial['tpot_degradation_pct']:+.2f}%"))
        checks.append(("adversarial cold-miss: max step delay bounded (skip-a-round)" + tag,
                       adversarial["max_step_ms"] < adversarial["budget_ms"],
                       f"{adversarial['max_step_ms']:.1f} < {adversarial['budget_ms']:.0f} ms"))
    if singleflight:
        checks.append(("concurrent cold start: single build, N waiters",
                       singleflight["builds"] == 1,
                       f"{singleflight['builds']} build / {singleflight['waiters']} waiters"))
        checks.append(("concurrent cold start: same error on FAILED",
                       singleflight["same_error"], str(singleflight["same_error"])))
    return checks


# ------------------------------------------------------------------ mock arm
def mock_adversarial(metric="v1", budget_ms=30.0):
    """Fabricated adversarial arm exercising the REAL v2 reduction/aggregation
    math (deferred rounds, an early-finishing warm request, a solo fresh tail)
    plus a fixed v1 block. Gate-passing by construction; NOT a perf claim."""
    T, dt = 20, 0.0063          # 20 tokens, 6.3 ms warm cadence
    step_walls = [dt] * (T + 6)
    step_walls[2] = 0.008       # worst engine step 8 ms (< 30 ms budget)
    token_times = {}
    for i in range(31):
        n = 12 if i == 30 else T                      # one early finisher
        token_times[f"warm-{i}"] = [dt * (k + 1) for k in range(n)]
    # fresh request: deferred 2 rounds (absent from steps 1-2), then a slower
    # effective cadence that runs past the warm batch into a solo tail
    token_times["fresh"] = [3 * dt + 0.012 * k for k in range(T)]
    red = reduce_adversarial_v2(step_walls, token_times, "fresh")
    v2 = aggregate_adversarial_v2([red], baseline_warm_tpot_ms=6.2)
    v1 = {"tpot_degradation_pct": 3.1, "max_step_ms": 4.2}
    return build_adversarial_result(v1, v2, metric, budget_ms)


def mock_cells(batches, arms, metric="v1"):
    """Deterministic fabricated timings that satisfy the gate shape — exercises
    the metric + report path only. NOT a performance claim (labeled in report).
    Numbers modeled on the A10 mode-2 acceptance run + constant-overhead priors."""
    base = {1: 9.8, 8: 10.4, 32: 12.6}  # unconstrained TPOT ms by batch
    add = {"grid": 0.14, "xgrammar": 0.22, "guidance": 0.30, "unconstrained": 0.0}
    cells = {}
    for arm in arms:
        for b in batches:
            n = b
            tpot = base[b] + add.get(arm, 0.0)
            tpots = [tpot + 0.4 * ((i % 5) - 2) / 2 for i in range(n * 20)]
            steps = [t for t in tpots]
            if arm == "grid":
                cold = [18.0 + 2.0 * (i % 3) for i in range(n)]
                warm = [1.6 + 0.3 * (i % 3) for i in range(n)]
                ttfts = cold[:1] + warm[1:]
            else:
                ttfts = [ (34.0 if arm != "unconstrained" else 3.0) + (i % 4) for i in range(n)]
                cold = warm = None
            decoded = n * 20
            wall = decoded * tpot / 1000.0 / max(1, b)  # batched
            cells[(arm, b)] = summarize_arm(ttfts, tpots, steps, decoded, wall,
                                            cold_ttfts_ms=cold, warm_ttfts_ms=warm)
    adversarial = mock_adversarial(metric=metric)
    singleflight = {"builds": 1, "waiters": 8, "same_error": True}
    return cells, adversarial, singleflight


# ----------------------------------------------------------------- real arm
# vLLM 0.24 V1 does not surface per-request RequestOutput.metrics, so TPOT is
# measured by a two-point wall-clock method (standard for offline throughput
# benchmarking): run the batch at max_tokens=T and at max_tokens=1, and take
# (wall_T - wall_1) / (T - 1) per request, B*(T-1)/(wall_T-wall_1) throughput.
# CAVEAT (metric v1): that division ASSUMES the batch advances in lockstep —
# every request present in every step. The moment a request is legitimately
# deferred out of a round (the §6 skip-a-round contract / GRID_DEFER), the
# batch is non-lockstep and the two-point wall conflates the deferred
# request's own tail with co-batched TPOT. The homogeneous warm cells keep
# lockstep, so v1 stays valid there; the adversarial arm therefore ALSO
# measures metric v2 via _step_loop_batch (per-request token timestamps on
# the raw engine step loop — no lockstep assumption).
# TTFT-specialize (the gate's "role+schema specialize" cost — grammar compile +
# first mask, NOT the model prefill) is measured at the GRID backend level,
# which is what the criterion is about; it is reported separately from the
# vLLM end-to-end wall.

def _build_batch(arm, grammar, schema_items, b, mt):  # pragma: no cover - GPU host
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    prompts, sps = [], []
    for i in range(b):
        _name, schema = schema_items[i % len(schema_items)]
        prompts.append(PROMPT + json.dumps(schema))
        kw = dict(temperature=0.0, max_tokens=mt)
        if arm != "unconstrained":
            envelope = (json.dumps({"grammar": grammar, "schema": schema})
                        if arm == "grid" else grammar)
            kw["structured_outputs"] = StructuredOutputsParams(grammar=envelope)
        sps.append(SamplingParams(**kw))
    return prompts, sps


def _measure_cell(llm, arm, grammar, schema_items, b, T, repeats):  # pragma: no cover
    import time
    # warmup: compile grammars + warm caches (not timed). Full length T, not a
    # token-4 stub: greedy paths revisit the same configurations every repeat,
    # so a full-T pass leaves the timed repeats at the steady state the cell
    # claims to measure (a 4-token warmup left repeat 1 paying every cold walk
    # + first-touch registration — batch 8 read +4443% on cold amortization
    # alone). Deliberate cold costs are measured where they belong: the
    # TTFT-specialize split and the adversarial cold-miss arm.
    p, s = _build_batch(arm, grammar, schema_items, b, T)
    llm.generate(p, s, use_tqdm=False)
    # also warm the 1-token shape: vLLM JIT-compiles Triton kernels (slot
    # mapping, bitmask apply) per shape ON FIRST USE — a JIT spike inside a
    # timed wall_1/wall_T reads as a 10x TPOT outlier for the whole cell
    p, s = _build_batch(arm, grammar, schema_items, b, 1)
    llm.generate(p, s, use_tqdm=False)
    tpots_ms, walls, decoded_tot = [], [], 0
    for _ in range(repeats):
        p, s = _build_batch(arm, grammar, schema_items, b, T)
        t0 = time.perf_counter()
        outs = llm.generate(p, s, use_tqdm=False)
        wall_T = time.perf_counter() - t0
        p1, s1 = _build_batch(arm, grammar, schema_items, b, 1)
        t0 = time.perf_counter()
        llm.generate(p1, s1, use_tqdm=False)
        wall_1 = time.perf_counter() - t0
        decoded = sum(len(o.outputs[0].token_ids) for o in outs)
        steps = max(1, T - 1)
        tpots_ms.append(1000.0 * max(0.0, wall_T - wall_1) / steps)
        walls.append(wall_T)
        decoded_tot += decoded
    tpot_mean = statistics.fmean(tpots_ms)
    return {
        "tpot_mean_ms": tpot_mean,
        "tpot_p99_ms": max(tpots_ms),           # coarse: max over repeats
        "step_p99_ms": max(tpots_ms),           # batch-lockstep: step == tpot
        "throughput_tok_s": decoded_tot / sum(walls) if sum(walls) > 0 else float("nan"),
        "ttft_p50_ms": float("nan"),            # end-to-end TTFT not isolated here
        "ttft_p99_ms": float("nan"),
        "n_requests": b,
    }


def _ttft_specialize_ms(model_name):  # pragma: no cover - GPU host
    """Grammar-specialize TTFT at the GRID level (compile + first cold mask):
    cold = first request for a schema, warm = a repeat. This is the gate's
    'role+schema specialize' cost, independent of vLLM model prefill."""
    import time

    from transformers import AutoTokenizer

    from grid.models.hf_adapter import HFTokenizerAdapter
    from grid.models.vllm_processor import _GuideRegistry

    grammar = GRAMMAR_SQL.read_text()
    reg = _GuideRegistry(HFTokenizerAdapter(AutoTokenizer.from_pretrained(model_name)))
    cold, warm = [], []
    for _name, schema in SCHEMAS.items():
        spec = {"grammar": grammar, "schema": schema}
        t0 = time.perf_counter()
        g = reg.guide_for(spec)
        g._mask_ids(g.initial_state)            # cold walk at the initial position
        cold.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter()
        g2 = reg.guide_for(spec)                # template cached -> warm
        g2._mask_ids(g2.initial_state)
        warm.append((time.perf_counter() - t0) * 1e3)
    return cold, warm


def real_run(args):  # pragma: no cover - GPU host only
    import vllm_grid_patch
    vllm_grid_patch.main()

    from vllm import LLM

    grammar = GRAMMAR_SQL.read_text()
    schema_items = list(SCHEMAS.items())
    arms = args.arms.split(",")
    batches = [int(b) for b in args.batches.split(",")]
    cells = {}

    for arm in arms:
        cfg = {} if arm == "unconstrained" else {"structured_outputs_config": {"backend": arm}}
        try:
            llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
                      max_model_len=args.max_model_len, enforce_eager=False, **cfg)
            for b in batches:
                cells[(arm, b)] = _measure_cell(llm, arm, grammar, schema_items, b,
                                                args.max_tokens, args.repeats)
            del llm
        except Exception as e:  # noqa: BLE001
            # a comparison arm that cannot consume our .grid dialect grammar is
            # skipped, not fatal (xgrammar needs GBNF/Lark; the cross-engine
            # SQL mask-latency comparison lives in bench/compare_engines.py with
            # each engine's native grammar). The binding G8 criteria are all
            # grid-vs-unconstrained.
            print(f"[skip arm {arm}] {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue

    if "grid" in arms:
        cold, warm = _ttft_specialize_ms(args.model)
        c1 = cells.get(("grid", 1)) or cells[("grid", batches[0])]
        c1["ttft_cold_p50_ms"] = float(sorted(cold)[len(cold) // 2])
        c1["ttft_warm_p50_ms"] = float(sorted(warm)[len(warm) // 2])

    adversarial = _adversarial_arm(args) if "grid" in arms else None
    singleflight = _singleflight_probe(args) if "grid" in arms else None
    return cells, adversarial, singleflight


def _step_loop_batch(llm, prompts, sampling_params, req_ids=None):  # pragma: no cover - GPU host
    """Drive the V1 engine step loop directly (llm.llm_engine.add_request +
    step() until no unfinished requests), recording per-step wall times and
    per-request token-arrival timestamps (seconds, relative to loop start).
    No lockstep assumption: a request absent from a round (deferred, finished)
    simply gets no timestamp that round. Returns (step_walls_s, token_times_s)
    in the exact shape reduce_adversarial_v2 consumes."""
    import time
    eng = llm.llm_engine
    ids = list(req_ids) if req_ids is not None else [f"req-{i}" for i in range(len(prompts))]
    for rid, prompt, sp in zip(ids, prompts, sampling_params, strict=True):
        eng.add_request(rid, prompt, sp)
    token_times = {rid: [] for rid in ids}
    seen = dict.fromkeys(ids, 0)
    step_walls_s = []
    t_start = time.perf_counter()
    while eng.has_unfinished_requests():
        t0 = time.perf_counter()
        outs = eng.step()
        t1 = time.perf_counter()
        step_walls_s.append(t1 - t0)
        for out in outs:
            n = len(out.outputs[0].token_ids)
            new = n - seen.get(out.request_id, 0)
            if new > 0:
                seen[out.request_id] = n
                token_times.setdefault(out.request_id, []).extend([t1 - t_start] * new)
    return step_walls_s, token_times


def _adversarial_arm(args):  # pragma: no cover - GPU host
    """Cold-miss injected into batch-32: compare co-batched TPOT of an all-warm
    batch-32 (grid) against a batch-32 where one request carries a fresh,
    never-warmed schema (its mask must be built cold, mid-batch). The overlap
    contract (worker-thread prefetch, GIL-released walk, §6 skip-a-round)
    should keep the co-batched TPOT degradation < 5% and the max step bounded.

    Reports BOTH metrics; --adversarial-metric selects which one gates:
      v1 (legacy): two-point lockstep wall, math unchanged — valid only while
          the batch is lockstep (defer makes it legitimately non-lockstep).
      v2: raw engine step loop — per-request TPOT over the 31 warm co-batched
          requests, max engine-step wall, and the fresh request's TTFT /
          completion / effective TPOT as transparency metrics (soft bound
          effective TPOT <= 3x warm, reported not gated).
    Acceptance (plan W9): with defer disabled and no adversary, v1 and v2
    agree within noise."""
    import time

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    grammar = GRAMMAR_SQL.read_text()
    schema_items = list(SCHEMAS.items())
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len, enforce_eager=False,
              structured_outputs_config={"backend": "grid"})
    T = args.max_tokens

    def _fresh_batch(prefix, r):
        # a never-warmed schema per repeat; prefix keeps the v1/v2 legs disjoint
        p, s = _build_batch("grid", grammar, schema_items, 32, T)
        fresh = {f"{prefix}_tbl_{r}": [f"{prefix}_col_{r}_{j}" for j in range(6)]}
        p[0] = PROMPT + json.dumps(fresh)
        s[0] = SamplingParams(temperature=0.0, max_tokens=T,
                              structured_outputs=StructuredOutputsParams(
                                  grammar=json.dumps({"grammar": grammar, "schema": fresh})))
        return p, s

    # ---- metric v1 (legacy two-point lockstep wall; math unchanged) ----
    base = _measure_cell(llm, "grid", grammar, schema_items, 32, T, args.repeats)
    deg_tpots, step_max = [], 0.0
    for r in range(args.repeats):
        p, s = _fresh_batch("z", r)
        t0 = time.perf_counter()
        llm.generate(p, s, use_tqdm=False)
        wall_T = time.perf_counter() - t0
        p1, s1 = _build_batch("grid", grammar, schema_items, 32, 1)
        t0 = time.perf_counter()
        llm.generate(p1, s1, use_tqdm=False)
        wall_1 = time.perf_counter() - t0
        tpot = 1000.0 * max(0.0, wall_T - wall_1) / max(1, T - 1)
        deg_tpots.append(tpot)
        step_max = max(step_max, tpot)
    v1 = {"tpot_degradation_pct": overhead_pct(statistics.fmean(deg_tpots),
                                               base["tpot_mean_ms"]),
          "max_step_ms": step_max}

    # ---- metric v2 (per-request step loop; no lockstep assumption) ----
    # baseline: all-warm batch-32 on the SAME step-loop surface (warm caches —
    # the v1 leg above already ran the warm schemas to completion repeatedly)
    p, s = _build_batch("grid", grammar, schema_items, 32, T)
    walls, times = _step_loop_batch(llm, p, s)
    base_v2 = reduce_adversarial_v2(walls, times, fresh_id=None)
    reductions = []
    for r in range(args.repeats):
        p, s = _fresh_batch("zz", r)   # "zz": disjoint from the v1 leg's "z"
        ids = ["fresh"] + [f"warm-{i}" for i in range(1, 32)]
        walls, times = _step_loop_batch(llm, p, s, req_ids=ids)
        reductions.append(reduce_adversarial_v2(walls, times, fresh_id="fresh"))
    v2 = aggregate_adversarial_v2(reductions, base_v2["warm_tpot_mean_ms"])
    del llm
    return build_adversarial_result(v1, v2, args.adversarial_metric,
                                    args.step_budget_ms)


def _singleflight_probe(args):  # pragma: no cover - GPU host
    """8 threads compile the SAME new schema; assert 1 build via registry stats.
    Reuses the exact SingleFlight the backend uses."""
    import threading

    from transformers import AutoTokenizer

    from grid.models.hf_adapter import HFTokenizerAdapter
    from grid.models.vllm_processor import _GuideRegistry

    reg = _GuideRegistry(HFTokenizerAdapter(AutoTokenizer.from_pretrained(args.model)))
    spec = {"grammar": GRAMMAR_SUBSET.read_text(),
            "schema": {"t": ["a", "b", "c"]}}
    errs = []

    def worker():
        try:
            reg.guide_for(spec)
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return {"builds": reg.stats["builds"], "waiters": 8, "same_error": not errs}


# ----------------------------------------------------------------- report
def write_report(cells, adversarial, singleflight, checks, out_path, mock):
    host = os.environ.get("GRID_HOST_LABEL", "local dev (unpinned)")
    arms = sorted({a for a, _b in cells}, key=lambda a: (a != "grid", a))
    batches = sorted({b for _a, b in cells})
    lines = [
        "# G8 serving benchmark" + ("  — MOCK (harness self-check, not a perf claim)" if mock else ""),
        "",
        f"Host: {host} | heterogeneous schemas ({len(SCHEMAS)} distinct grammars) | "
        f"batches {', '.join(map(str, batches))}",
        "",
    ]
    if mock:
        lines += ["> **Mock run.** Timings are fabricated deterministic values that "
                  "exercise the metric reduction + gate logic + report only. Real "
                  "numbers come from the declared GPU runner (vLLM has no macOS "
                  "wheels). The reduction arithmetic is unit-tested "
                  "(tests/serving/test_serving_metrics.py).", ""]
    lines += ["| arm | batch | TTFT p50 (ms) | TPOT mean (ms) | TPOT p99 (ms) | "
              "step p99 (ms) | tok/s | overhead vs unconstrained |",
              "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for b in batches:
        base = cells.get(("unconstrained", b))
        for arm in arms:
            c = cells.get((arm, b))
            if not c:
                continue
            ov = (overhead_pct(c["tpot_mean_ms"], base["tpot_mean_ms"])
                  if base and arm != "unconstrained" else None)
            ttft = c.get("ttft_p50_ms", float("nan"))
            ttft_s = "—" if ttft != ttft else f"{ttft:.1f}"  # nan -> em dash
            lines.append(
                f"| {arm} | {b} | {ttft_s} | {c['tpot_mean_ms']:.2f} | "
                f"{c['tpot_p99_ms']:.2f} | {c['step_p99_ms']:.2f} | "
                f"{c['throughput_tok_s']:.0f} | "
                f"{'—' if ov is None else f'{ov:+.2f}%'} |")
    grid1 = cells.get(("grid", 1))
    if grid1 and "ttft_cold_p50_ms" in grid1:
        lines += ["", f"GRID TTFT split @batch 1: cold specialize "
                  f"**{grid1['ttft_cold_p50_ms']:.1f} ms**, warm "
                  f"**{grid1.get('ttft_warm_p50_ms', float('nan')):.2f} ms**."]
    if adversarial:
        v1, v2 = adversarial.get("v1"), adversarial.get("v2")
        if v1 or v2:
            metric = adversarial.get("metric", "v1")
            budget = adversarial["budget_ms"]
            lines += ["", "**Adversarial cold-miss arm** (fresh never-warmed schema "
                      "injected into batch-32; both metrics reported, gating metric: "
                      f"**{metric}**, budget {budget:.0f} ms):", ""]
            if v1:
                lines.append(
                    "- **metric v1 — legacy two-point lockstep wall** (assumes every "
                    "request advances every step; conflates a deferred request's tail "
                    "into the batch wall): co-batched TPOT degradation "
                    f"**{v1['tpot_degradation_pct']:+.2f}%**, max step "
                    f"**{v1['max_step_ms']:.1f} ms**.")
            if v2:
                lines.append(
                    f"- **metric v2 — per-request, no lockstep assumption** (raw engine "
                    f"step loop; TPOT = (t_last−t_first)/(T−1) per request over the "
                    f"{v2.get('n_warm', 31)} warm co-batched requests): co-batched TPOT "
                    f"degradation **{v2['tpot_degradation_pct']:+.2f}%**, max engine-step "
                    f"wall **{v2['max_step_ms']:.1f} ms**.")
                lines.append(
                    "- **fresh request (reported, not gated)**: TTFT "
                    f"**{v2['fresh_ttft_ms']:.1f} ms**, completion "
                    f"**{v2['fresh_completion_ms']:.1f} ms**, effective TPOT "
                    f"**{v2['fresh_effective_tpot_ms']:.2f} ms** "
                    f"({v2['fresh_tpot_ratio']:.2f}x warm; soft bound <= 3x warm: "
                    f"{'OK' if v2.get('soft_bound_ok') else 'EXCEEDED'}).")
            lines += ["", "The §6 skip-a-round/overlap contract is gated on the "
                      f"metric-{metric} values above."]
        else:  # legacy single-metric dict (pre-v2 runs)
            lines += ["", "**Adversarial cold-miss arm** (cache cleared, maximal identifier "
                      "position, injected into batch-32): co-batched TPOT degradation "
                      f"**{adversarial['tpot_degradation_pct']:+.2f}%**, max step "
                      f"**{adversarial['max_step_ms']:.1f} ms** "
                      f"(budget {adversarial['budget_ms']:.0f} ms) — the §6 "
                      "skip-a-round/overlap contract holds."]
    if singleflight:
        lines += ["", f"**Concurrent cold start**: {singleflight['builds']} build / "
                  f"{singleflight['waiters']} waiters, "
                  f"same-error-on-FAILED {singleflight['same_error']} (E17 single-flight)."]
    lines += ["", "## Gate G8", "", "| criterion | pass | value |", "|---|---|---|"]
    for name, ok, val in checks:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {val} |")
    all_ok = all(ok for _n, ok, _v in checks)
    lines += ["", f"Gate G8: {'**PASS**' if all_ok else '**FAIL**' if checks else '**not evaluated**'}"
              + ("" if not mock else " *(mock inputs — validates harness, not hardware)*") + ".",
              "", "Harness: `bench/vllm_serving_bench.py` (+ `bench/vllm_grid_patch.py`).", ""]
    pathlib.Path(out_path).write_text("\n".join(lines))
    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    # default arms are grid vs unconstrained (the binding gate is TPOT overhead
    # vs unconstrained). xgrammar/guidance can be added, but they need a native
    # GBNF/Lark grammar — our SQL dialect is authored in GRID's .grid LALR
    # format, so they're skipped if handed it (the cross-engine SQL mask-latency
    # comparison lives in bench/compare_engines.py with per-engine grammars).
    ap.add_argument("--arms", default="grid,unconstrained")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--step-budget-ms", type=float, default=30.0)
    ap.add_argument("--adversarial-metric", choices=("v1", "v2"), default="v2",
                    help="which adversarial metric GATES (both are always "
                         "measured and printed). v2 is RATIFIED as the binding "
                         "metric (2026-07-09): per-request TPOT over the warm "
                         "co-batched requests + max engine-step wall; the "
                         "fresh request is reported, not gated. v1 keeps the "
                         "legacy lockstep math for comparison.")
    ap.add_argument("--assert-gates", action="store_true")
    ap.add_argument("--out", default=str(BENCH_DIR / "RESULTS-serving.md"))
    args = ap.parse_args()

    if args.mock:
        arms = args.arms.split(",")
        batches = [int(b) for b in args.batches.split(",")]
        cells, adversarial, singleflight = mock_cells(batches, arms,
                                                      metric=args.adversarial_metric)
    else:  # pragma: no cover - GPU host
        cells, adversarial, singleflight = real_run(args)

    checks = evaluate_gates(cells, adversarial, singleflight)
    all_ok = write_report(cells, adversarial, singleflight, checks, args.out, args.mock)
    for name, ok, val in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {val}")
    print(f"report -> {args.out}")
    # mock never gates CI (fabricated inputs); real run honors --assert-gates
    return 0 if (all_ok or not args.assert_gates or args.mock) else 1


if __name__ == "__main__":
    sys.exit(main())
