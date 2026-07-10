"""Integration-floor calibration for batched-serving TPOT overhead.

Runs vLLM's NATIVE xgrammar backend through the exact two-point TPOT
method (bench/vllm_serving_bench.py) on grammars xgrammar consumes natively
(per-request JSON schemas, heterogeneous like the serving-under-batch-load SQL
arm), against the same unconstrained baseline.

Purpose: attribute GRID's measured overhead-vs-unconstrained at batch 32.
vLLM 0.24 fills structured-output bitmasks per request per step through
Python (serial below fill_bitmask_parallel_threshold=128), so every backend
pays a per-request plumbing tax the unconstrained arm does not. The number
this script prints is that tax for the reference native backend on the same
host/model/batch — the floor any backend behind this interface can reach.

Run (GPU host):
  .venv/bin/python bench/vllm_xgr_floor.py --batches 8,32 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

# Heterogeneous NON-COMPLETING schemas (minItems >> max_tokens): an
# object-shaped schema closes and forces EOS early, and the two-point method
# then divides a short run by the full T — xgrammar read *negative* overhead
# on the first calibration attempt. A 500-item array cannot complete within
# 96 tokens, pinning both arms to identical step counts. (The serving-under-batch-load
# SQL arm does not need this: whitespace keeps SQL statements paddable to max_tokens, and
# the recorded tok/s confirms full-length decodes for grid and unconstrained.)
SCHEMAS = {
    "ints": {"type": "array", "items": {"type": "integer"}, "minItems": 500},
    "nums": {"type": "array", "items": {"type": "number"}, "minItems": 500},
    "strs": {"type": "array", "items": {"type": "string"}, "minItems": 500},
    "pairs": {"type": "array", "minItems": 500,
              "items": {"type": "object",
                        "properties": {"k": {"type": "string"},
                                       "v": {"type": "integer"}},
                        "required": ["k", "v"],
                        "additionalProperties": False}},
}
PROMPT = "Return one long JSON array for this schema: "


def _json_schema(schema):
    return schema


def _build(arm, b, mt):
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    items = list(SCHEMAS.items())
    prompts, sps = [], []
    for i in range(b):
        _n, schema = items[i % len(items)]
        prompts.append(PROMPT + json.dumps(schema))
        kw = dict(temperature=0.0, max_tokens=mt)
        if arm == "xgrammar":
            kw["structured_outputs"] = StructuredOutputsParams(
                json=_json_schema(schema))
        sps.append(SamplingParams(**kw))
    return prompts, sps


def _cell(llm, arm, b, T, repeats):
    llm.generate(*_build(arm, b, T), use_tqdm=False)  # full-T warmup, untimed
    tpots = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        outs = llm.generate(*_build(arm, b, T), use_tqdm=False)
        wall_T = time.perf_counter() - t0
        t0 = time.perf_counter()
        llm.generate(*_build(arm, b, 1), use_tqdm=False)
        wall_1 = time.perf_counter() - t0
        tpots.append(1000.0 * max(0.0, wall_T - wall_1) / max(1, T - 1))
        short = sum(1 for o in outs if len(o.outputs[0].token_ids) < T)
        if short:  # early termination voids the fixed-T denominator
            print(f"  WARNING: {short}/{b} {arm} requests terminated early",
                  flush=True)
    return statistics.fmean(tpots)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--batches", default="8,32")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    args = ap.parse_args()

    from vllm import LLM

    batches = [int(b) for b in args.batches.split(",")]
    cells = {}
    for arm in ("xgrammar", "unconstrained"):
        cfg = ({"structured_outputs_config": {"backend": "xgrammar"}}
               if arm == "xgrammar" else {})
        llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
                  max_model_len=2048, enforce_eager=False, **cfg)
        for b in batches:
            cells[(arm, b)] = _cell(llm, arm, b, args.max_tokens, args.repeats)
            print(f"  cell {arm} b={b}: TPOT {cells[(arm, b)]:.2f} ms", flush=True)
        del llm

    print("\n# xgrammar integration floor (two-point TPOT, native JSON schemas)")
    for b in batches:
        base, xgr = cells[("unconstrained", b)], cells[("xgrammar", b)]
        ov = 100.0 * (xgr - base) / base if base > 0 else float("nan")
        print(f"batch {b}: xgrammar {xgr:.2f} ms vs unconstrained {base:.2f} ms "
              f"-> XGR_FLOOR_OVERHEAD b={b}: {ov:+.2f}%", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
