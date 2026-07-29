"""Scanner-build byte-identity digest gate (wave-a E2-scanner-dedup).

Pins the OUTPUT of both scanner-build arms per grammar, so a refactor of the
shared subset-construction core can be proven behavior-preserving bit-for-bit:

  eager     build_scanner(..., factored=False)          -> dense ScannerDFA
  factored  build_factored_scanner(...) default budget  -> dense ScannerDFA,
            or a bounded deterministic BFS probe of the LazyProductDFA when
            the product breaches the budget (probe order is the facade's own
            FIFO/ascending-class order, so the digest is run-stable)
  comps     per-terminal TerminalDFA subset-construction outputs, hashed in
            terminal_order (keyed by the (pattern, is_literal) memo identity)

Frozensets are serialized as sorted tuples (set repr order is not canonical);
everything else digested is nested tuples of ints/bools, whose repr is stable
across processes (int hashing is PYTHONHASHSEED-independent, which the
builders' own block ordering already relies on). GrammarInvalid outcomes are
recorded as class+message text — message parity is part of the gate.

Workers scrub GRID_PERF_* from their environment: arm selection is explicit
per call and the factored budget stays at the in-code default, so ambient
shell flags cannot skew a leg.

Usage:
    python bench/perfbench/diff_scanner_digest.py --sets stratified_200,ttfm_capped \
        --builtin --timeout 60 --jobs 4 --out tmp/scanner-digest-pre
    ... refactor ...
    python bench/perfbench/diff_scanner_digest.py --sets ... --out tmp/scanner-digest-post
    python bench/perfbench/diff_scanner_digest.py --compare \
        tmp/scanner-digest-pre tmp/scanner-digest-post
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

MANIFEST = os.path.join(os.path.dirname(__file__), "manifest.json")
DATA_DIR = os.environ.get(
    "GRID_JSB_DATA",
    os.path.join(os.path.dirname(__file__), "..", "..", "tmp",
                 "jsb-src", "data"))

# digest the tree this script lives in (see profile_phases.py: the venv's grid
# install points at the main checkout, so a worktree run needs the pin)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_LAZY_PROBE_STATES = 200


class _Timeout(BaseException):
    """SIGALRM payload; BaseException so no library except-clause eats it."""


# ---------------------------------------------------------------- builtin leg
# In-repo floor corpus, runnable without tmp/jsb-src: the checked-in grammars,
# the wide->64-terminal variant (tests/conftest.py wide_source), and the
# synthetic terminal sets of tests/lexer/test_factored_differential.py
# (window/zombie/priority families + its JSON schema corpus).

_WINDOW_PATTERNS = [
    "a{3}", "a{2,4}", "a{2,}", "xa{0,2}", "[0-9]{2,3}x", "(ab){2}",
    "a{1,2}b{1,2}", "[a-z]{1,16}x", '"[a-zA-Z0-9]{0,32}"', "[a-f]{4,64}",
]
_ZOMBIE_PATTERNS = ["a|[^\\x00-\\xff]b", "z[^\\x00-\\xff]y", "x([^\\x00-\\xff]y)?q"]
_JSON_SCHEMAS = [
    {"type": "object",
     "properties": {"name": {"type": "string", "pattern": "^[a-z]{2,8}$"},
                    "n": {"type": "integer"}},
     "required": ["name"]},
    {"enum": ["red", "green", "blue", "a longer literal", 1, 2.5, True, None]},
    {"type": "string", "format": "date-time"},
    {"type": "object",
     "properties": {"a": {"type": "string", "minLength": 2, "maxLength": 6},
                    "b": {"type": "array", "items": {"type": "number"}}}},
]


def builtin_units() -> list[list]:
    units: list[list] = []
    gdir = os.path.join(_ROOT, "grammars")
    for fn in sorted(os.listdir(gdir)):
        if fn.endswith(".grid"):
            units.append(["grid", os.path.join(gdir, fn)])
    units.append(["wide", ""])
    for i in range(len(_JSON_SCHEMAS)):
        units.append(["json", str(i)])
    for pat in _WINDOW_PATTERNS + _ZOMBIE_PATTERNS:
        units.append(["rx", [pat]])
    units.append(["rx", ["[a-z]{1,16}x", "[a-y]{2,8}", "z{0,4}q"]])   # window product
    units.append(["prio", ""])   # literal-vs-regex tie family (is_literal coverage)
    units.append(["rx", ["a*"]])         # empty-match GrammarInvalid outcome
    units.append(["rx", ["a{4,2}"]])     # bad-repeat GrammarInvalid outcome
    return units


def _unit_grammar(kind: str, payload):
    """-> grammar-like object with .terminals / .terminal_order."""
    from grid.grammar import spec
    from grid.grammar.spec import Terminal

    if kind in ("file", "grid", "wide", "json"):
        if kind == "file":
            with open(payload) as f:
                schema = json.load(f)
            if isinstance(schema, dict) and "schema" in schema and "tests" in schema:
                schema = schema["schema"]   # wrapped maskbench layout
        elif kind == "json":
            schema = _JSON_SCHEMAS[int(payload)]
        if kind in ("file", "json"):
            from grid.jsonschema import compile_json_schema
            src, _rec = compile_json_schema(schema)
        elif kind == "grid":
            with open(payload) as f:
                src = f.read()
        else:   # wide: conftest.py wide_source construction
            with open(os.path.join(_ROOT, "grammars", "sql_subset.grid")) as f:
                sql = f.read()
            kws = [f'"w{i:02d}"' for i in range(60)]
            src = (sql
                   + "wide_stmt: wide_kw | wide_stmt wide_kw\n"
                   + "wide_kw: " + " | ".join(kws) + "\n"
                   ).replace('stmt: query ";"', 'stmt: query ";" | wide_stmt ";"')
        return spec.load(src)

    if kind == "rx":
        terms = {
            f"T{i}": Terminal(name=f"T{i}", pattern=p, is_literal=False,
                              ignored=False, decl_index=i)
            for i, p in enumerate(payload)
        }
    elif kind == "prio":
        terms = {
            "RX": Terminal(name="RX", pattern="a[b]", is_literal=False,
                           ignored=False, decl_index=0),
            "LIT": Terminal(name="LIT", pattern="ab", is_literal=True,
                            ignored=False, decl_index=1),
            "RX2": Terminal(name="RX2", pattern="ab|cd", is_literal=False,
                            ignored=False, decl_index=2),
        }
    else:   # pragma: no cover
        raise ValueError(kind)

    class _G:
        terminals = terms
        terminal_order = tuple(terms)
    return _G


# ---------------------------------------------------------------- digests

def _sfs(fss) -> tuple:
    """Frozensets -> sorted tuples: the canonical serialization."""
    return tuple(tuple(sorted(fs)) for fs in fss)


def digest_dense(d) -> str:
    payload = repr((d.start, d.trans, d.accept, _sfs(d.accepts_all),
                    _sfs(d.live), d.h_max))
    return "dense:" + hashlib.sha256(payload.encode()).hexdigest()


def digest_lazy(p) -> str:
    """Over-budget facade: full component digests + global classes + a bounded
    BFS probe (states 0..cap in the facade's own discovery order — probing i
    in order with classes ascending IS the eager discovery order, so the
    probed prefix is deterministic)."""
    comp_reprs = tuple(
        (c.trans, c.class_of, c.accepting, c.co_acc, c.matches_empty)
        for c in p.comps)
    rows = []
    i = 0
    while i < len(p._states) and i < _LAZY_PROBE_STATES:
        rows.append(tuple(p._class_step(i, g) for g in range(p._n_g)))
        i += 1
    payload = repr((comp_reprs, p._gclass_of, p.h_max, rows,
                    tuple(p.accept[:i]), _sfs(p.accepts_all[:i]),
                    _sfs(p.live[:i])))
    return "lazy:" + hashlib.sha256(payload.encode()).hexdigest()


def digest_comps(terminals, order) -> str:
    from grid.lexer.factored import _component
    h = hashlib.sha256()
    for name in order:
        t = terminals[name]
        c = _component(t.pattern, t.is_literal)
        h.update(repr((t.pattern, t.is_literal, c.trans, c.class_of,
                       c.accepting, c.co_acc, c.matches_empty)).encode())
    return "comps:" + h.hexdigest()


def unit_record(kind: str, payload, timeout_s: int) -> dict:
    from grid.errors import GrammarInvalid
    from grid.lexer.dfa import ScannerDFA, build_scanner
    from grid.lexer.factored import build_factored_scanner

    rec: dict = {"stage": "load"}

    def _alarm(*a):
        raise _Timeout()

    old_h = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout_s)
    t0 = time.monotonic()
    try:
        try:
            g = _unit_grammar(kind, payload)
        except _Timeout:
            raise
        except Exception as e:   # Unsupported / compile errors: no scanner arms
            rec["compile"] = f"{type(e).__name__}:{str(e)[:300]}"
            return rec
        rec["stage"] = "eager"
        try:
            rec["eager"] = digest_dense(
                build_scanner(g.terminals, g.terminal_order, factored=False))
        except GrammarInvalid as e:
            rec["eager"] = f"GrammarInvalid:{e}"
        rec["stage"] = "factored"
        try:
            fact = build_factored_scanner(g.terminals, g.terminal_order)
            rec["factored"] = (digest_dense(fact) if isinstance(fact, ScannerDFA)
                               else digest_lazy(fact))
        except GrammarInvalid as e:
            rec["factored"] = f"GrammarInvalid:{e}"
        rec["stage"] = "comps"
        try:
            rec["comps"] = digest_comps(g.terminals, g.terminal_order)
        except GrammarInvalid as e:
            rec["comps"] = f"GrammarInvalid:{e}"
        return rec
    except _Timeout:
        rec["timeout"] = rec.pop("stage")
        return rec
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_h)
        rec.pop("stage", None)
        rec["secs"] = round(time.monotonic() - t0, 3)


def _unit_id(kind: str, payload) -> str:
    if kind == "file":
        return payload
    if kind in ("grid",):
        return "builtin:grid:" + os.path.basename(payload)
    if kind == "rx":
        return "builtin:rx:" + "\x1f".join(payload)
    return f"builtin:{kind}:{payload}"


def worker(list_file: str, out_file: str, timeout_s: int) -> None:
    for k in [k for k in os.environ if k.startswith("GRID_PERF_")]:
        del os.environ[k]
    with open(list_file) as f:
        units = json.load(f)
    with open(out_file, "w") as out:
        for kind, payload in units:
            try:
                rec = unit_record(kind, payload, timeout_s)
            except Exception as e:   # harness failure, not a build outcome
                rec = {"harness_error": f"{type(e).__name__}: {e}"}
            rec["id"] = _unit_id(kind, payload)
            out.write(json.dumps(rec) + "\n")
            out.flush()


# ---------------------------------------------------------------- parent

def schema_path(schema_id: str) -> str:
    if "---" in schema_id:
        split, name = schema_id.split("---", 1)
        p = os.path.join(DATA_DIR, split, name + ".json")
        if os.path.exists(p):
            return p
    # remaining ids (BFCL_*, JME_*, ...) are maskbench-layout files
    return os.path.join(DATA_DIR, "..", "maskbench", "data",
                        schema_id + ".json")


def load_run(out_dir: str) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for fn in sorted(os.listdir(out_dir)):
        if not (fn.startswith("chunk") and fn.endswith(".jsonl")):
            continue
        with open(os.path.join(out_dir, fn)) as f:
            for line in f:
                rec = json.loads(line)
                by_id[rec["id"]] = rec
    return by_id


_FIELDS = ("compile", "eager", "factored", "comps", "timeout", "harness_error")


def compare(dir_a: str, dir_b: str) -> None:
    a, b = load_run(dir_a), load_run(dir_b)
    bad = 0
    for missing, which in ((set(a) - set(b), dir_b), (set(b) - set(a), dir_a)):
        for uid in sorted(missing):
            print(f"MISSING in {which}: {uid}")
            bad += 1
    for uid in sorted(set(a) & set(b)):
        ra, rb = a[uid], b[uid]
        diffs = [(f, ra.get(f), rb.get(f)) for f in _FIELDS
                 if ra.get(f) != rb.get(f)]
        if diffs:
            bad += 1
            print(f"DIFF {uid}")
            for f, va, vb in diffs:
                print(f"  {f}: {str(va)[:160]}  !=  {str(vb)[:160]}")
    n = len(set(a) & set(b))
    if bad:
        print(f"FAIL: {bad} divergent/missing of {n} shared units")
        sys.exit(1)
    print(f"OK: {n} units bit-identical")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=2, metavar=("LIST", "OUT"))
    ap.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"))
    ap.add_argument("--sets", default="")
    ap.add_argument("--builtin", action="store_true",
                    help="include the in-repo floor corpus")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default="tmp/scanner-digest")
    args = ap.parse_args()

    if args.worker:
        worker(args.worker[0], args.worker[1], args.timeout)
        return
    if args.compare:
        compare(*args.compare)
        return

    units: list[list] = []
    if args.builtin:
        units.extend(builtin_units())
    if args.sets:
        with open(MANIFEST) as f:
            sets = json.load(f)["sets"]
        seen: set[str] = set()
        for name in args.sets.split(","):
            for sid in sets[name]:
                p = schema_path(sid)
                if p not in seen:
                    seen.add(p)
                    units.append(["file", p])

    os.makedirs(args.out, exist_ok=True)
    chunks = [units[i::args.jobs] for i in range(args.jobs)]
    procs = []
    for i, chunk in enumerate(chunks):
        lf = os.path.join(args.out, f"chunk{i}.json")
        of = os.path.join(args.out, f"chunk{i}.jsonl")
        with open(lf, "w") as f:
            json.dump(chunk, f)
        procs.append(subprocess.Popen(
            [sys.executable, __file__, "--worker", lf, of,
             "--timeout", str(args.timeout)]))
    for p in procs:
        p.wait()

    counts: dict[str, int] = {}
    for rec in load_run(args.out).values():
        for f in _FIELDS:
            if f in rec:
                key = f if f in ("compile", "timeout", "harness_error") else (
                    f + ":" + rec[f].split(":", 1)[0])
                counts[key] = counts.get(key, 0) + 1
    print(f"n={len(units)}", json.dumps(counts, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
