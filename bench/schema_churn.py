"""Schema churn: mask-cache footprint, hit rate, and cold-start cost on real SQL.

Teacher-forced replay of the Spider dev gold queries (canonical lowercase form,
the grammar's dialect) in a shuffled order, so consecutive requests usually hit
DIFFERENT schemas: 166 databases, one grammar, one tokenizer. A schema's guide
(its L3 lexicons + producer) is built on first sight — that build and the
schema's first query are the cold start. Per step we time the full next-token
mask (hit vs miss via the T1 counters) and then advance with the gold token.

Two configurations, each in its own process (module-level kernel state must
not leak between them):
  --t2 off   per-schema T1 caches only
  --t2 on    per-schema T1 + one shared cross-schema T2 tier (the serving
             registry's configuration: schema-independent entries are reused)

Also reports the audit trail's cost in isolation: per-record append time
(BLAKE2b chaining, Python) and serialized bytes per record.

  .venv-bench/bin/python bench/schema_churn.py --spider tmp/spider-data/spider_data --t2 on \\
      --out tmp/review/churn-t2on.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import resource
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np  # noqa: E402
from spider_coverage import normalize  # noqa: E402


def pct(xs, q):
    return float(np.percentile(np.asarray(xs), q)) if xs else float("nan")


def entry_bytes(e) -> int:
    """Payload bytes held per entry. Kernel-v7 entries keep only the packed CI
    bytes + the kernel blob (their ci_tokens/cd_groups are lazy decode views —
    never touched here, so the measurement does not inflate the cache)."""
    if hasattr(e, "blob"):
        return len(e.ci_bytes or b"") + len(e.blob or b"")
    ci = e.ci_tokens
    n = ci.nbytes if hasattr(ci, "nbytes") else 4 * len(ci)
    return n + sum(4 * len(g.token_ids) for g in e.cd_groups)


def audit_microbench(n: int = 200_000) -> dict:
    from grid.audit.log import GENERATE, AuditLog

    log = AuditLog()
    rng = random.Random(0)
    t0 = time.perf_counter()
    for i in range(n):
        log.append(i, rng.getrandbits(63), f"{rng.getrandbits(128):032x}", rng.randrange(151_000),
                   rng.randrange(151_000), GENERATE)
    dt = time.perf_counter() - t0
    t1 = time.perf_counter()
    assert log.verify_chain()
    dv = time.perf_counter() - t1
    line = json.dumps(log.records[-1].__dict__, separators=(",", ":"))
    return {"append_us": 1e6 * dt / n, "verify_us": 1e6 * dv / n, "bytes_per_record_json": len(line) + 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spider", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--t2", choices=["on", "off"], default="on")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="first N queries of the stream (smoke)")
    ap.add_argument("--split", default="dev", help="gold file stem: dev (20 dbs) | train_spider (140 dbs)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from spider_ex import _grammar_artifacts, schema_snapshot
    from transformers import AutoTokenizer

    from grid.guide import GridGuide
    from grid.mask.cache import MaskCacheT2
    from grid.models.hf_adapter import HFTokenizerAdapter

    hf_tok = AutoTokenizer.from_pretrained(args.tokenizer)
    adapter = HFTokenizerAdapter(hf_tok)
    t0 = time.perf_counter()
    tables, dfa, trie = _grammar_artifacts(adapter)
    grammar_build_s = time.perf_counter() - t0

    dev = json.load(open(f"{args.spider}/{args.split}.json"))
    schemas = {db["db_id"]: db for db in json.load(open(f"{args.spider}/tables.json"))}
    order = random.Random(args.seed).sample(dev, len(dev))[: args.limit]
    t2 = MaskCacheT2() if args.t2 == "on" else None

    guides: dict[str, GridGuide] = {}
    build_s: list[float] = []
    hit_lat, miss_lat, t2_lat = [], [], []  # T1 hit | cold walk | T1 miss served by T2
    first_q_hits = first_q_steps = later_hits = later_steps = 0
    first_q_wall, later_wall = [], []
    rejected = steps = 0
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    trace = []
    for qi, ex in enumerate(order):
        sql = normalize(ex["query"])
        if sql is None:
            continue
        db_id = ex["db_id"]
        first = db_id not in guides
        tq = time.perf_counter()
        if first:
            tb = time.perf_counter()
            snap = schema_snapshot(schemas[db_id])
            guides[db_id] = GridGuide(tables=tables, dfa=dfa, trie=trie, adapter=adapter,
                                      lexicons=snap.lexicons(tables),
                                      schema_fingerprint=snap.fingerprint, mask_t2=t2)
            build_s.append(time.perf_counter() - tb)
        g = guides[db_id]
        cache = g.producer.cache
        state = g.initial_state
        for tok in hf_tok(sql, add_special_tokens=False)["input_ids"]:
            m0, h2 = cache.misses, (t2.hits if t2 else 0)
            ts = time.perf_counter()
            ids, _ = g._mask_ids(state)
            dt = time.perf_counter() - ts
            miss = cache.misses > m0
            if not miss:
                hit_lat.append(dt)
            elif t2 is not None and t2.hits > h2:
                t2_lat.append(dt)
            else:
                miss_lat.append(dt)
            steps += 1
            if first:
                first_q_steps += 1
                first_q_hits += not miss
            else:
                later_steps += 1
                later_hits += not miss
            if not bool((ids == tok).any()):
                rejected += 1
                break
            state = g.get_next_state(state, tok)
        (first_q_wall if first else later_wall).append(time.perf_counter() - tq)
        if (qi + 1) % 100 == 0 or qi + 1 == len(order):
            ents = [e for gg in guides.values() for e in gg.producer.cache._t1.values()]
            trace.append({
                "queries": qi + 1, "schemas": len(guides), "t1_entries": len(ents),
                "t1_mb": sum(entry_bytes(e) for e in ents) / 2**20,
                "t2_entries": len(t2._map) if t2 else 0,
                "t2_mb": (sum(entry_bytes(e) for e in t2._map.values()) / 2**20) if t2 else 0.0,
                "rss_growth_mb": (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss0) / 2**20,
                "t1_hit_rate_so_far": len(hit_lat) / max(1, steps),
                "cold_walk_rate_so_far": len(miss_lat) / max(1, steps),
            })
            print(json.dumps(trace[-1]), file=sys.stderr, flush=True)

    out = {
        "config": {"t2": args.t2, "tokenizer": args.tokenizer, "seed": args.seed, "split": args.split,
                   "queries": len(order), "schemas": len(guides)},
        "grammar_build_s": grammar_build_s,
        "schema_build_ms": {"p50": 1e3 * pct(build_s, 50), "p90": 1e3 * pct(build_s, 90),
                            "max": 1e3 * max(build_s)},
        "steps": steps, "rejected_replays": rejected,
        "t1_hit_rate": len(hit_lat) / max(1, steps),
        "t2_hit_rate": len(t2_lat) / max(1, steps),
        "cold_walk_rate": len(miss_lat) / max(1, steps),
        "t1_hit_rate_first_query_of_schema": first_q_hits / max(1, first_q_steps),
        "t1_hit_rate_later_queries": later_hits / max(1, later_steps),
        "hit_us": {"p50": 1e6 * pct(hit_lat, 50), "p90": 1e6 * pct(hit_lat, 90), "p99": 1e6 * pct(hit_lat, 99)},
        "t2_hit_us": {"p50": 1e6 * pct(t2_lat, 50), "p90": 1e6 * pct(t2_lat, 90),
                      "p99": 1e6 * pct(t2_lat, 99)},
        "cold_walk_us": {"p50": 1e6 * pct(miss_lat, 50), "p90": 1e6 * pct(miss_lat, 90),
                         "p99": 1e6 * pct(miss_lat, 99)},
        "all_us": {q: 1e6 * pct(hit_lat + t2_lat + miss_lat, q) for q in (50, 90, 99)},
        "query_wall_ms": {"first_of_schema_p50": 1e3 * pct(first_q_wall, 50),
                          "first_of_schema_p90": 1e3 * pct(first_q_wall, 90),
                          "later_p50": 1e3 * pct(later_wall, 50), "later_p90": 1e3 * pct(later_wall, 90)},
        "trace": trace,
        "audit": audit_microbench(),
        "host": "local dev: Apple M3 Max, 36 GB (unpinned)",
    }
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "trace"}, indent=1))


if __name__ == "__main__":
    main()
