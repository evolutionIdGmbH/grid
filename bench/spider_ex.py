"""Spider execution-accuracy (EX) harness — the G9 accuracy arms, locally runnable.

Per dev question, each arm generates SQL for the question's database and the
result sets are executed against the Spider SQLite database and compared to the
gold query's results (order-sensitive iff the gold has ORDER BY; floats rounded).

Arms:
- grid            GRID v0.0.7: constrained generation (mode-1 GRID-owned loop,
                  jump-forward Write spans, token-denominated reserve, audit on;
                  Spider dialect grammar + per-database L3 lexicons -> every
                  emitted identifier schema-valid by construction) PLUS one
                  SemanticChecker-guided constrained retry when the checker
                  finds binding violations (kept iff it does not increase them).
- unconstrained   the same model, prompt, and greedy decoding, no constraints
                  (HF generate) — the EX-delta baseline G9 requires.
- grid-repair-off / grid-cache-off / grid-audit-off / grid-jf-off
                  Ablation arms (checker-guided retry off / write-back cache
                  disabled / audit trail off / jump-forward spans disabled via
                  j_max=1). All ablation arms run without the retry so they
                  isolate the mask machinery; EX is retry- and cache-invariant
                  by construction except for the repair-off delta itself.

Metrics per arm: syntax-valid % (sqlite EXPLAIN), execution-ok %, EX %, EX-delta
vs unconstrained, truncation rate, output tokens, gen tok/s, wall time.

Run (0.5B locally; the binding G9 run repoints --model/--device on the GPU box):
  .venv-bench/bin/python bench/spider_ex.py --spider <dir> --sample 100 \\
      --arms grid,unconstrained --out bench/RESULTS-spider.md
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import re
import sqlite3
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import torch  # noqa: E402

GRAMMAR = (pathlib.Path(__file__).parent.parent / "grammars" / "sql_spider.grid").read_text()


# ---------------------------------------------------------------- model


class KVCachedModel:
    """TransformersModel-compatible callable with incremental KV cache.

    The GRID-owned loop calls model(prompt+generated) each step; this wrapper
    feeds only the new suffix when the previous call was a prefix (Write spans
    may extend by several tokens at once), resetting on divergence."""

    def __init__(self, model, adapter, device: str) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = adapter
        self.device = device
        self.vocab_size = max(adapter.vocabulary.values()) + 1
        self._ids: list[int] = []
        self._past = None

    def reset(self) -> None:
        self._ids, self._past = [], None

    @torch.no_grad()
    def __call__(self, token_ids: list[int]) -> torch.Tensor:
        ids = [int(t) for t in token_ids] or [self.tokenizer.eos_token_id]
        n = len(self._ids)
        if self._past is not None and len(ids) > n and ids[:n] == self._ids:
            new = ids[n:]
        else:
            self.reset()
            new = ids
        inp = torch.tensor([new], dtype=torch.long, device=self.device)
        out = self.model(inp, past_key_values=self._past, use_cache=True)
        self._past = out.past_key_values
        self._ids = ids
        logits = out.logits[0, -1, :].float().cpu()
        if logits.shape[0] < self.vocab_size:
            logits = torch.cat([logits, torch.full((self.vocab_size - logits.shape[0],), float("-inf"))])
        return logits[: self.vocab_size]


class DisabledCache:
    """G9 cache-off ablation: MaskCache interface, never stores, never hits."""

    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0

    def get(self, key):
        self.misses += 1
        return None

    def publish(self, entry):
        return entry

    def invalidate_namespace(self) -> None:  # pragma: no cover
        pass


# ---------------------------------------------------------------- prompt & data


def load_spider(spider_dir: str) -> tuple[list[dict], dict[str, dict]]:
    dev = json.load(open(os.path.join(spider_dir, "dev.json")))
    schemas = {db["db_id"]: db for db in json.load(open(os.path.join(spider_dir, "tables.json")))}
    return dev, schemas


def schema_prompt(db: dict) -> str:
    lines = []
    for ti, tname in enumerate(db["table_names_original"]):
        cols = [c for (t, c) in db["column_names_original"] if t == ti]
        lines.append(f"table {tname.lower()} ( {', '.join(c.lower() for c in cols)} )")
    return "\n".join(lines)


def build_prompt(tokenizer, db: dict, question: str) -> str:
    system = (
        "You translate questions to SQLite SQL for the given schema. "
        "Write exactly one SQL query, all lowercase, no explanation. "
        "When joining tables, alias them as t1, t2, ... and qualify columns."
    )
    user = f"Schema:\n{schema_prompt(db)}\n\nQuestion: {question}\nSQL:"
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )


_IDENT = re.compile(r"[a-z_][a-z0-9_]*")


def schema_snapshot(db: dict):
    """SchemaSnapshot from tables.json, filtered to the grammar's identifier
    language — Spider has columns like ``Official_ratings_(millions)`` whose
    spelling no COLUMN_NAME lexeme can ever scan; the L3 composition rule now
    rejects such words at guide build (they'd otherwise dead-end generation
    mid-identifier). Dropped names are simply not generatable, matching the
    grammar's own language."""
    from grid.policy.schema import SchemaSnapshot

    d: dict[str, list[str]] = {}
    for ti, tname in enumerate(db["table_names_original"]):
        tl = tname.lower()
        if not _IDENT.fullmatch(tl):
            continue
        d[tl] = [
            cl for (t, c) in db["column_names_original"]
            if t == ti and _IDENT.fullmatch(cl := c.lower())
        ]
    return SchemaSnapshot.from_dict(d)


# ---------------------------------------------------------------- execution / EX


def run_sql(db_file: str, sql: str, deadline_s: float = 5.0):
    """-> ("ok", rows) | ("error", msg). Read-only, progress-handler timeout."""
    try:
        conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True, timeout=deadline_s)
    except Exception as e:
        return "error", f"connect: {e}"
    t0 = time.monotonic()
    conn.set_progress_handler(lambda: 1 if time.monotonic() - t0 > deadline_s else 0, 20_000)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(5_000)
        return "ok", rows
    except Exception as e:
        return "error", str(e)[:160]
    finally:
        conn.close()


def _canon_cell(v):
    if isinstance(v, float):
        return round(v, 4)
    return v


def results_match(gold_rows, pred_rows, ordered: bool) -> bool:
    g = [tuple(_canon_cell(c) for c in r) for r in gold_rows]
    p = [tuple(_canon_cell(c) for c in r) for r in pred_rows]
    if ordered:
        return g == p
    return sorted(map(repr, g)) == sorted(map(repr, p))


def explain_ok(db_file: str, sql: str) -> bool:
    st, _ = run_sql(db_file, "explain " + sql, deadline_s=3.0)
    return st == "ok"


# ---------------------------------------------------------------- arms


_ARTIFACTS: dict = {}  # grammar-level (db-independent) + per-db caches


def _grammar_artifacts(adapter):
    got = _ARTIFACTS.get("base")
    if got is None:
        from grid.grammar import spec as gspec
        from grid.grammar.projection import RoleProjection
        from grid.lalr.compile import compile_tables
        from grid.lexer.dfa import build_scanner
        from grid.trie.build import build_trie

        grammar = gspec.load(GRAMMAR)
        proj = RoleProjection.full(grammar).build()
        tables = compile_tables(proj, frozenset({"TABLE_NAME", "COLUMN_NAME"}))
        dfa = build_scanner(grammar.terminals, grammar.terminal_order)
        got = _ARTIFACTS["base"] = (tables, dfa, build_trie(adapter))
    return got


def _db_artifacts(adapter, db: dict):
    got = _ARTIFACTS.get(db["db_id"])
    if got is None:
        from grid.lalr.reserve import ReserveTable

        tables, dfa, _trie = _grammar_artifacts(adapter)
        snap = schema_snapshot(db)
        lex = snap.lexicons(tables)
        reserve = ReserveTable(tables=tables, dfa=dfa, adapter=adapter, lexicons=lex)
        got = _ARTIFACTS[db["db_id"]] = (lex, reserve, snap.fingerprint)
    return got


def _db_checker(adapter, db: dict):
    """Alias-aware SemanticChecker per db (grid/policy/semantic.py)."""
    key = ("checker", db["db_id"])
    got = _ARTIFACTS.get(key)
    if got is None:
        from grid.policy.semantic import SemanticChecker

        tables, dfa, _trie = _grammar_artifacts(adapter)
        got = _ARTIFACTS[key] = SemanticChecker(tables, dfa, schema_snapshot(db))
    return got


def build_repair_prompt(tokenizer, db: dict, question: str, bad_sql: str, details: list[str]) -> str:
    system = (
        "You translate questions to SQLite SQL for the given schema. "
        "Write exactly one SQL query, all lowercase, no explanation. "
        "When joining tables, alias them as t1, t2, ... and qualify columns."
    )
    user = (
        f"Schema:\n{schema_prompt(db)}\n\nQuestion: {question}\n"
        f"A previous attempt was:\n{bad_sql}\n"
        f"It is invalid: {'; '.join(details)}.\n"
        "Write a corrected SQL query.\nSQL:"
    )
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )


def gen_grid_repair(model, adapter, hf_tok, db: dict, question: str, prompt: str,
                    max_tokens: int) -> dict:
    """grid + one SemanticChecker-guided constrained retry (the W4 repair arm).

    The mask guarantees grammar + schema-lexicon validity; the checker catches
    the provably-mask-unenforceable layer (alias/column binding). On violations,
    regenerate once with the violations quoted back; keep the retry iff it has
    no more violations than the original."""
    first = gen_grid(model, adapter, db, prompt, set(), max_tokens)
    checker = _db_checker(adapter, db)
    v1 = checker.check(first["sql"]) if first["sql"] else []
    if not v1 or first["stop"].startswith("ERROR"):
        return {**first, "repaired": False, "violations": len(v1)}
    details = [f"{v.kind}: {v.detail}" for v in v1[:4]]
    rprompt = build_repair_prompt(hf_tok, db, question, first["sql"], details)
    second = gen_grid(model, adapter, db, rprompt, set(), max_tokens)
    v2 = checker.check(second["sql"]) if second["sql"] else v1
    best = second if len(v2) <= len(v1) else first
    return {
        **best,
        "tokens": first["tokens"] + second["tokens"],
        "seconds": first["seconds"] + second["seconds"],
        "repaired": best is second,
        "violations": min(len(v1), len(v2)),
    }


def _generator(model, adapter, db: dict, ablate: set[str]):
    """One GridSequenceGeneratorAdapter per (db, ablations), cached: each call
    clones the processor (guide.copy shares the warm mask cache — E11's designed
    reuse) with a fresh audit chain per generation."""
    from grid.audit.log import AuditLog
    from grid.generate.api import GridSequenceGeneratorAdapter
    from grid.guide import GridGuide
    from grid.processors import GridLogitsProcessor
    from grid.samplers import greedy

    key = (db["db_id"], frozenset(ablate))
    got = _ARTIFACTS.get(key)
    if got is None:
        tables, dfa, trie = _grammar_artifacts(adapter)
        lex, reserve, fingerprint = _db_artifacts(adapter, db)
        guide = GridGuide(
            tables=tables, dfa=dfa, trie=trie, adapter=adapter,
            lexicons=lex, schema_fingerprint=fingerprint,
            reserve=reserve, audit=None if "audit" in ablate else AuditLog(),
            j_max=1 if "jumpforward" in ablate else 8,
            mask_cache=DisabledCache() if "cache" in ablate else None,
        )
        processor = GridLogitsProcessor(adapter, guide)
        got = _ARTIFACTS[key] = GridSequenceGeneratorAdapter(model, processor, greedy(), mode="sql")
    return got


def gen_grid(model, adapter, db: dict, prompt: str, ablate: set[str], max_tokens: int) -> dict:
    gen = _generator(model, adapter, db, ablate)
    model.reset()
    t0 = time.monotonic()
    result = gen(prompt, max_tokens=max_tokens, seed=0)
    dt = time.monotonic() - t0
    return {
        "sql": result.text.strip(),
        "stop": result.stop_reason,
        "tokens": len(result.token_ids),
        "seconds": dt,
        "truncated": result.stop_reason != "EOS_ACCEPT",
    }


@torch.no_grad()
def gen_unconstrained(hf_model, hf_tok, prompt: str, device: str, max_tokens: int) -> dict:
    ids = hf_tok(prompt, return_tensors="pt").to(device)
    t0 = time.monotonic()
    out = hf_model.generate(
        **ids, max_new_tokens=max_tokens, do_sample=False,
        pad_token_id=hf_tok.eos_token_id,
    )
    dt = time.monotonic() - t0
    new = out[0][ids["input_ids"].shape[1]:]
    text = hf_tok.decode(new, skip_special_tokens=True).strip()
    # first statement only: cut at ';' or a blank line; strip code fences
    text = text.replace("```sql", " ").replace("```", " ").strip()
    text = re.split(r";|\n\s*\n", text)[0].strip()
    truncated = len(new) >= max_tokens
    return {"sql": text, "stop": "MAX_TOKENS" if truncated else "STOP",
            "tokens": int(len(new)), "seconds": dt, "truncated": truncated}


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spider", required=True, help="spider_data dir (dev.json, tables.json, database/)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--arms", default="grid,unconstrained",
                    help="comma list: grid,unconstrained,grid-repair-off,"
                         "grid-cache-off,grid-audit-off,grid-jf-off")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default="tmp/spider-out")
    ap.add_argument("--host-label", default=os.environ.get("GRID_HOST_LABEL", "local dev (unpinned)"),
                    help="host description recorded in the report header")
    args = ap.parse_args()

    dev, schemas = load_spider(args.spider)
    rng = random.Random(args.seed)
    sample = rng.sample(dev, min(args.sample, len(dev)))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from grid.models.hf_adapter import HFTokenizerAdapter

    print(f"loading {args.model} on {args.device}", file=sys.stderr)
    hf_tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float16 if args.device != "cpu" else torch.float32
    hf_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    adapter = HFTokenizerAdapter(hf_tok)
    kv_model = KVCachedModel(hf_model, adapter, args.device)
    _grammar_artifacts(adapter)  # tables/DFA/trie once up front

    os.makedirs(args.dump, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    stats = {a: {"n": 0, "syntax": 0, "exec": 0, "ex": 0, "trunc": 0,
                 "tokens": 0, "gen_s": 0.0} for a in arms}
    per_q: list[dict] = []

    for qi, ex in enumerate(sample):
        db = schemas[ex["db_id"]]
        db_file = os.path.join(args.spider, "database", ex["db_id"], f"{ex['db_id']}.sqlite")
        prompt = build_prompt(hf_tok, db, ex["question"])
        gold_ordered = "order by" in ex["query"].lower()
        g_status, gold_rows = run_sql(db_file, ex["query"])
        if g_status != "ok":
            continue  # gold must execute for EX to be defined

        row: dict = {"i": qi, "db": ex["db_id"], "question": ex["question"], "gold": ex["query"]}
        for arm in arms:
            ablate = set()
            if arm.startswith("grid"):
                if "cache-off" in arm:
                    ablate.add("cache")
                if "audit-off" in arm:
                    ablate.add("audit")
                if "jf-off" in arm:
                    ablate.add("jumpforward")
                try:
                    if arm == "grid":
                        # v0.0.7: the checker-guided retry is part of the system
                        r = gen_grid_repair(kv_model, adapter, hf_tok, db,
                                            ex["question"], prompt, args.max_tokens)
                    else:
                        # ablation arms (incl. grid-repair-off) isolate the mask
                        # machinery: no retry
                        r = gen_grid(kv_model, adapter, db, prompt, ablate, args.max_tokens)
                except Exception as e:
                    r = {"sql": "", "stop": f"ERROR:{type(e).__name__}", "tokens": 0,
                         "seconds": 0.0, "truncated": True, "error": str(e)[:200]}
            else:
                r = gen_unconstrained(hf_model, hf_tok, prompt, args.device, args.max_tokens)
            sql = r["sql"]
            syn = bool(sql) and explain_ok(db_file, sql)
            status, rows = run_sql(db_file, sql) if syn else ("error", "syntax")
            ex_ok = status == "ok" and results_match(gold_rows, rows, gold_ordered)
            s = stats[arm]
            s["n"] += 1
            s["syntax"] += syn
            s["exec"] += status == "ok"
            s["ex"] += ex_ok
            s["trunc"] += bool(r["truncated"])
            s["tokens"] += r["tokens"]
            s["gen_s"] += r["seconds"]
            row[arm] = {**r, "syntax_ok": syn, "exec_ok": status == "ok", "ex": ex_ok}
        per_q.append(row)
        with open(os.path.join(args.dump, f"q{qi:04d}.json"), "w") as f:
            json.dump(row, f, indent=1)
        if (qi + 1) % 10 == 0:
            done = {a: f"EX {s['ex']}/{s['n']}" for a, s in stats.items()}
            print(f"[{qi + 1}/{len(sample)}] {done}", file=sys.stderr)

    print_report(args, arms, stats)
    if args.out:
        write_report(args.out, args, arms, stats)
        print(f"report -> {args.out}", file=sys.stderr)


def _rows(args, arms, stats):
    out = []
    ex_base = None
    if "unconstrained" in stats and stats["unconstrained"]["n"]:
        s = stats["unconstrained"]
        ex_base = s["ex"] / s["n"]
    for arm in arms:
        s = stats[arm]
        n = max(1, s["n"])
        ex_rate = s["ex"] / n
        out.append({
            "arm": arm, "n": s["n"],
            "syntax": s["syntax"] / n, "exec": s["exec"] / n, "ex": ex_rate,
            "delta": (ex_rate - ex_base) if ex_base is not None else float("nan"),
            "trunc": s["trunc"] / n,
            "tok": s["tokens"] / n,
            "tps": s["tokens"] / s["gen_s"] if s["gen_s"] else float("nan"),
        })
    return out


def print_report(args, arms, stats) -> None:
    for r in _rows(args, arms, stats):
        print(f"{r['arm']:>18}: n={r['n']} syntax {r['syntax']:.1%} exec {r['exec']:.1%} "
              f"EX {r['ex']:.1%} (delta {r['delta']:+.1%}) trunc {r['trunc']:.1%} "
              f"tok/q {r['tok']:.0f} gen tok/s {r['tps']:.1f}")


def write_report(path, args, arms, stats) -> None:
    lines = [
        "# Spider dev — execution accuracy (EX), GRID-constrained vs unconstrained",
        "",
        f"Model: `{args.model}` ({args.device}, greedy) | sample: {stats[arms[0]]['n']} dev questions "
        f"(seed {args.seed}) | max_tokens {args.max_tokens} | grammar: `grammars/sql_spider.grid` "
        f"(100% dev-gold coverage) + per-database L3 lexicons | host: {args.host_label}",
        "",
        "EX = predicted and gold result sets match on the Spider SQLite database "
        "(order-sensitive iff gold has ORDER BY). Syntax-valid = sqlite EXPLAIN "
        "accepts. GRID generations parse by construction and every identifier is "
        "schema-valid via the L3 lexicons; its failures are semantic, not syntactic.",
        "",
        "| arm | n | syntax-valid | executes | EX | EX-delta | truncated | tok/query | gen tok/s |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in _rows(args, arms, stats):
        lines.append(
            f"| {r['arm']} | {r['n']} | {r['syntax']:.1%} | {r['exec']:.1%} | {r['ex']:.1%} | "
            f"{r['delta']:+.1%} | {r['trunc']:.1%} | {r['tok']:.0f} | {r['tps']:.1f} |"
        )
    lines += [
        "",
        "Arms `grid-cache-off`, `grid-audit-off`, `grid-jf-off` are the G9 ablations "
        "(write-back cache / audit trail / jump-forward spans); EX is identical by "
        "construction — the column that moves is gen tok/s.",
        "",
        "Binding G9 numbers run on the declared cloud runner with the reference model "
        "(DESIGN.md SS10); this harness repoints via --model/--device.",
    ]
    pathlib.Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
