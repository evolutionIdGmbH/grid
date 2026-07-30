"""GRID_PERF_HASHCONS differential tests (0.3.x candidate #17).

Flag-off is the oracle everywhere the oracle terminates: normalize output
must be JSON-identical, compiles must match in outcome + recorded set, and
grammar text must be byte-equal ('dedupe' alone) or start-anchored
isomorphic ('norm' shares normalized subtrees, so the compiler's id() memo
skips duplicate rule families legacy built and deduped away — renumbering
later rules without changing the derivation structure). The five committed
capped schemas are the hang-family regression anchors: legacy normalize
never finishes on them, so they are asserted fast + deterministic flag-on.
"""

import json
import pathlib
import signal
import sys
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "bench"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "bench" / "perfbench"))

from diff_hashcons import grammar_isomorphic  # noqa: E402

import grid.jsonschema.normalize as N  # noqa: E402
from grid.jsonschema.compiler import (  # noqa: E402
    SchemaCompiler,
    Unsupported,
    compile_schema,
)
from grid.jsonschema.normalize import (  # noqa: E402
    Unmergeable,
    _hashcons_components,
    normalize,
)

DATA = pathlib.Path(__file__).parent / "data" / "hashcons"
OFF = frozenset()
NORM = frozenset({"norm"})
DEDUPE = frozenset({"dedupe"})
BOTH = frozenset({"norm", "dedupe"})


class _OracleTimeout(BaseException):
    """BaseException: normalize's _valid swallows Exception subclasses."""


class oracle_deadline:
    """The flag-off arm is the oracle but may be exponentially slow; a case
    is only comparable when the oracle finishes."""

    def __init__(self, seconds: int = 30) -> None:
        self.seconds = seconds

    def __enter__(self):
        def bail(*a):
            raise _OracleTimeout()

        self._prev = signal.signal(signal.SIGALRM, bail)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._prev)
        return False


# ------------------------------------------------------------ flag parsing

def test_flag_parsing():
    assert _hashcons_components("") == frozenset()
    assert _hashcons_components("0") == frozenset()
    assert _hashcons_components("1") == BOTH
    assert _hashcons_components("all") == BOTH
    assert _hashcons_components("norm") == NORM
    assert _hashcons_components("norm,dedupe") == BOTH
    assert _hashcons_components("bogus,norm") == NORM


def test_memo_restored_after_run():
    normalize({"type": "object"}, hashcons=NORM)
    assert N._MEMO is None


# ------------------------------------------------------- digest soundness

def test_digest_type_discrimination():
    memo = N._HashconsMemo()
    ds = {k: N._digest(v, memo) for k, v in {
        "int": 1, "float": 1.0, "bool": True, "str_1": "1",
        "str_true": "true", "none": None,
    }.items()}
    assert len(set(ds.values())) == len(ds)
    assert N._digest({"a": [1]}, memo) != N._digest({"a": [1.0]}, memo)
    assert N._digest([], memo) != N._digest({}, memo)
    # key ORDER is semantic: merge2 output order and the compiler's
    # properties walk both inherit it
    assert N._digest({"a": 1, "b": 2}, memo) != \
        N._digest({"b": 2, "a": 1}, memo)
    assert N._digest({"a": 1, "b": 2}, memo) == \
        N._digest(dict([("a", 1), ("b", 2)]), memo)


def test_digest_undigestable():
    memo = N._HashconsMemo()
    assert N._digest({"a": {1, 2}}, memo) is None      # non-JSON value
    cyc: dict = {}
    cyc["x"] = cyc
    assert N._digest(cyc, memo) is None                # cyclic graph


def test_digest_shared_dag_linear():
    memo = N._HashconsMemo()
    node = {"type": "string"}
    for _ in range(200):                # 2^200 tree expansion as a DAG
        node = {"properties": {"a": node, "b": node}}
    assert N._digest(node, memo) is not None


# ------------------------------------------- normalize output equivalence

CURATED = [
    {"allOf": [
        {"type": "object", "properties": {"a": {"type": "integer"}},
         "required": ["a"]},
        {"properties": {"b": {"type": "string"}}, "required": ["b"]},
    ]},
    {"allOf": [{"enum": [1, 2, 3]}, {"not": {"const": 2}}]},
    {"allOf": [{"type": "string"}, {"type": "integer"}]},
    {"$defs": {"base": {"type": "object",
                        "properties": {"a": {"type": "integer"}},
                        "required": ["a"]}},
     "allOf": [{"$ref": "#/$defs/base"},
               {"properties": {"b": {"type": "string"}}}]},
    {"$defs": {"s": {"type": "string", "minLength": 1}},
     "$ref": "#/$defs/s", "maxLength": 4},
    {"type": "object",
     "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
     "dependencies": {"a": ["b"], "b": ["a"]}},
    {"if": {"properties": {"k": {"const": "x"}}, "required": ["k"]},
     "then": {"required": ["v"]}, "else": {"required": ["w"]},
     "type": "object", "properties": {"k": {"type": "string"},
                                      "v": {}, "w": {}}},
    {"not": {"type": "string"}},
    {"anyOf": [{"type": "string"}, {"type": "integer"}], "minLength": 2},
    {"$schema": "http://json-schema.org/draft-04/schema#",
     "$defs": {"t": {"type": "integer"}},
     "properties": {"a": {"$ref": "#/$defs/t", "minimum": 3}},
     "exclusiveMinimum": True, "minimum": 5, "type": "integer"},
    {"anyOf": [
        {"type": "object", "properties": {"t": {"const": "a"}}},
        {"type": "object", "properties": {"t": {"type": "string"}}},
    ]},
]


def _ref_sibling_chain(n: int, extra_depth: int = 0) -> dict:
    """$ref-with-siblings chain: each hop re-merges and re-normalizes, the
    profiled o12175 blowup shape."""
    defs = {"d0": {"type": "object",
                   "properties": {"a": {"type": "integer"}},
                   "required": ["a"]}}
    for i in range(1, n):
        defs[f"d{i}"] = {"$ref": f"#/$defs/d{i - 1}",
                         "minProperties": 1,
                         "properties": {f"p{i}": {"type": "string"}}}
    root: dict = {"$defs": defs, "$ref": f"#/$defs/d{n - 1}",
                  "maxProperties": 64}
    for _ in range(extra_depth):
        root = {"type": "object", "properties": {"w": root},
                "$defs": defs} if "$defs" not in root else \
            {"type": "object", "properties": {"w": root}}
    return root


def _allof_product(width: int) -> dict:
    return {"allOf": [
        {"type": "object",
         "properties": {f"k{i}": {"type": "string", "maxLength": 8}},
         "required": [f"k{i}"]}
        for i in range(width)
    ]}


def _deep_repeated(depth: int) -> dict:
    """Identical subtree content at every level: hits at many depths, with
    near-cutoff levels exercising the rel-rejection + (digest, depth) path."""
    leaf = {"allOf": [{"type": "object",
                       "properties": {"q": {"type": "integer"}}},
                      {"required": ["q"]}]}
    node: dict = dict(leaf)
    for _ in range(depth):
        node = {"type": "object", "properties": {"x": node, "y": dict(leaf)}}
    return node


EQUIV_CASES = CURATED + [
    _ref_sibling_chain(8),
    _ref_sibling_chain(12, extra_depth=3),
    _allof_product(10),
    _deep_repeated(70),     # crosses the depth>64 cutoff
    _deep_repeated(63),     # ends exactly at the boundary
]


@pytest.mark.parametrize("idx", range(len(EQUIV_CASES)))
def test_normalize_equivalence(idx):
    schema = EQUIV_CASES[idx]
    with oracle_deadline(60):
        old = normalize(schema, hashcons=OFF)
    new = normalize(schema, hashcons=NORM)
    assert json.dumps(old, sort_keys=True) == json.dumps(new, sort_keys=True)


def test_unmergeable_message_identical():
    # both sides carry `format`, twice, so the (cached) failure re-raises
    # with the same content-determined message
    branch = {"allOf": [{"format": "date"}, {"format": "uri"}]}
    schema = {"properties": {"a": dict(branch), "b": dict(branch)}}
    with oracle_deadline(30):
        old = normalize(schema, hashcons=OFF)
    new = normalize(schema, hashcons=NORM)
    assert json.dumps(old, sort_keys=True) == json.dumps(new, sort_keys=True)
    for _hc in (OFF, NORM):
        with pytest.raises(Unmergeable) as ei:
            N.merge2({"format": "date"}, {"format": "uri"}, {})
        assert "two distinct formats" in str(ei.value)


def test_merge2_cached_failure_reraises_fresh():
    prev = N._MEMO
    N._MEMO = N._HashconsMemo()
    try:
        excs = []
        for _ in range(2):
            with pytest.raises(Unmergeable) as ei:
                N.merge2({"format": "date"}, {"format": "uri"}, {})
            excs.append(ei.value)
        assert str(excs[0]) == str(excs[1])
        assert excs[0] is not excs[1]
    finally:
        N._MEMO = prev


# ------------------------------------------------- compile equivalence

@pytest.mark.parametrize("idx", range(len(EQUIV_CASES)))
def test_compile_equivalence(idx):
    schema = EQUIV_CASES[idx]
    with oracle_deadline(60):
        try:
            old = compile_schema(schema, hashcons=OFF)
        except Unsupported as e:
            old = ("unsupported", str(e))
    try:
        new = compile_schema(schema, hashcons=BOTH)
    except Unsupported as e:
        new = ("unsupported", str(e))
    if isinstance(old, tuple) and old[0] == "unsupported":
        assert new == old
        return
    old_src, old_rec = old
    new_src, new_rec = new
    assert old_rec == new_rec
    assert old_src == new_src or grammar_isomorphic(old_src, new_src), \
        f"text diverged (not isomorphic)\nOLD:\n{old_src[:800]}\n" \
        f"NEW:\n{new_src[:800]}"


@pytest.mark.parametrize("idx", range(len(EQUIV_CASES)))
def test_compile_dedupe_alone_byte_identical(idx):
    schema = EQUIV_CASES[idx]
    with oracle_deadline(60):
        try:
            old = compile_schema(schema, hashcons=OFF)
        except Unsupported as e:
            old = ("unsupported", str(e))
    try:
        new = compile_schema(schema, hashcons=DEDUPE)
    except Unsupported as e:
        new = ("unsupported", str(e))
    assert old == new


def test_nested_draft_memo_scoping():
    # a draft-04 $schema inside a 2020-12 compile: the compiler's nested
    # normalize (patternProperties overlap merge) must get a fresh memo with
    # ITS root/draft mode, never entries from the outer run
    schema = {
        "type": "object",
        "$defs": {"t": {"type": "object",
                        "properties": {"n": {"type": "integer"}}}},
        "properties": {
            "ax": {"$schema": "http://json-schema.org/draft-04/schema#",
                   "$ref": "#/$defs/t", "required": ["n"]},
        },
        "patternProperties": {
            "^a": {"minProperties": 1},
        },
    }
    with oracle_deadline(30):
        old = compile_schema(schema, hashcons=OFF)
    new = compile_schema(schema, hashcons=BOTH)
    assert old[1] == new[1]
    assert old[0] == new[0] or grammar_isomorphic(old[0], new[0])


# ----------------------------------------------------- dedupe worklist

def _mk_compiler(rules: dict[str, list[str]], order: list[str]):
    c = SchemaCompiler({}, hashcons=OFF)
    c.rules = {k: list(v) for k, v in rules.items()}
    c.rule_order = list(order)
    return c


def _dedupe_both(rules, order, start):
    a = _mk_compiler(rules, order)
    sa = a._dedupe_rules(start)
    b = _mk_compiler(rules, order)
    sb = b._dedupe_rules_worklist(start)
    assert (sa, a.rules, a.rule_order) == (sb, b.rules, b.rule_order)
    return sa, a.rules, a.rule_order


def test_dedupe_worklist_alias_chain():
    # two parallel 300-deep chains, identical shape: must merge pairwise;
    # legacy needs ~300 fixpoint passes, the worklist requeues locally
    rules: dict[str, list[str]] = {}
    order: list[str] = []
    for pfx in ("a", "b"):
        for i in range(300):
            nxt = f"{pfx}{i + 1}" if i < 299 else None
            rules[f"{pfx}{i}"] = [f'"," X {nxt}'] if nxt else ['"," X']
            order.append(f"{pfx}{i}")
    rules["top"] = ["a0 b0"]
    order.insert(0, "top")
    _, merged, morder = _dedupe_both(rules, order, "top")
    assert set(morder) == {"top"} | {f"a{i}" for i in range(300)}
    assert merged["top"] == ["a0 a0"]


def test_dedupe_worklist_self_recursion_merges():
    rules = {
        "top": ["a b"],
        "a": ["X", "a Y"],
        "b": ["X", "b Y"],
    }
    order = ["top", "a", "b"]
    _, merged, morder = _dedupe_both(rules, order, "top")
    assert morder == ["top", "a"]
    assert merged["top"] == ["a a"]


def test_dedupe_worklist_cyclic_twins_do_not_merge():
    # mutually-recursive twins are NOT @SELF-canonical: they must survive,
    # exactly as legacy (only self-references are canonicalized)
    rules = {
        "top": ["a1 b1"],
        "a1": ["X a2"], "a2": ["X a1"],
        "b1": ["X b2"], "b2": ["X b1"],
    }
    order = ["top", "a1", "a2", "b1", "b2"]
    _, merged, morder = _dedupe_both(rules, order, "top")
    assert morder == order


def test_dedupe_worklist_requeue_representative():
    # a later-order rule merged into an earlier one whose own signature then
    # collides: exercises requeue-of-dependents and min-order representative
    rules = {
        "r0": ["r2 Z"],
        "r1": ["r3 Z"],
        "r2": ["X"],
        "r3": ["X"],
        "top": ["r0 r1"],
    }
    order = ["top", "r0", "r1", "r2", "r3"]
    _, merged, morder = _dedupe_both(rules, order, "top")
    assert morder == ["top", "r0", "r2"]
    assert merged["top"] == ["r0 r0"]


def test_dedupe_worklist_eps_alts():
    rules = {
        "top": ["a b"],
        "a": ["|EPS|", '"," X a'],
        "b": ["|EPS|", '"," X b'],
    }
    order = ["top", "a", "b"]
    _dedupe_both(rules, order, "top")


# --------------------------------------------------------- hang canary

# expected flag-on outcomes: legacy normalize never finishes on any of
# these, so the deterministic declared outcomes below ARE the untimed
# legacy outcomes (the virtual-_n guard charges skipped twin constructions,
# so o39217's rule-budget verdict matches what legacy would conclude)
CANARY = [
    ("o12175.json", "unsupported"),
    ("o39217.json", "unsupported"),
    ("o11667.json", "unsupported"),
    ("cloudify.json", "unsupported"),
    ("wp_105_Normalized.json", "ok"),
]


@pytest.mark.parametrize("fname,expect", CANARY)
def test_hang_canary(fname, expect):
    # all five hang legacy normalize past the 120s bench cap; flag-on they
    # must resolve fast and deterministically (the sanctioned improvement)
    with open(DATA / fname) as f:
        schema = json.load(f)
    if isinstance(schema, dict) and "schema" in schema and "tests" in schema:
        schema = schema["schema"]
    t0 = time.monotonic()
    try:
        src, _ = compile_schema(schema, hashcons=BOTH)
        got = "ok"
        assert src.startswith("%start start")
    except Unsupported:
        got = "unsupported"
    dt = time.monotonic() - t0
    assert got == expect
    assert dt < 10, f"{fname}: {dt:.1f}s"
