"""G10 audit-replay gate (DESIGN.md E14/G10) — the full-scale run.

Criteria (binding):
- replay EVERY step of >= 1,000 generations spanning >= 1 namespace rollover:
  bit-identical record chains (masks compare via content-addressed
  mask_entry_id + blocked_count inside the hash-chained records; EOS and Write
  records included);
- tamper property test: random record, random field, >= 10^3 trials,
  100% detection.

Setup: SQL-subset grammar, MockTokenizer vocabulary, MockModel logits with a
seeded multinomial sampler — the mode-1 GRID-owned loop exactly as
grid/generate/api.py drives it (Write spans, reserve/budget writes, EOS), one
guide copy per generation SHARING the template's write-back mask cache. At the
halfway generation the cache namespace rolls over (E10): entries recompute and
re-publish under identical content hashes, so replays of pre-rollover
generations must still be bit-identical — that is the property this gate pins.

Run:  .venv-bench/bin/python bench/g10_replay.py [--gens 1000] [--assert-gates]
Report: bench/RESULTS-g10.md
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
import random
import sys
import time

import torch

BENCH_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(BENCH_DIR.parent))

from grid.generate import build_guide  # noqa: E402
from grid.guide import COMPLETE  # noqa: E402
from grid.models.mock import MockModel  # noqa: E402
from grid.models.tokenizer_adapter import MockTokenizer  # noqa: E402
from grid.protocols import Generate, Write  # noqa: E402
from grid.samplers import multinomial  # noqa: E402

# the tests' SQL mock vocabulary (tests/conftest.py SQL_TOKENS): multi-byte,
# boundary-crossing, and mid-identifier tokens included
SQL_TOKENS = (
    "select", "sel", "ect", "insert", "update", "delete", "from", "where", "and", "or",
    "limit", "into", "values", "set", " ", "*", ",", ";", "=", "<", ">", "(", ")",
    "users", "orders", "user_id", "name", "email", "total", "id", "salaries",
    " from ", " where ", "select ", "'x'", "'", "1", "42", "0",
    "s;", "rs;", "us", "ers", " users", ",name", "us_x", "sala", "ries",
)

MAX_TOKENS = 40


def generate_one(template, model, sampler, seed: int):
    """The mode-1 loop of grid/generate/api.py::_generate_one, verbatim
    semantics, on a fresh guide copy sharing the template's mask cache."""
    guide = template.copy()
    guide.max_new_tokens = MAX_TOKENS
    rng = torch.Generator()
    rng.manual_seed(seed)
    state = guide.initial_state
    out: list[int] = []
    while True:
        instr = guide.get_next_instruction(state)
        if isinstance(instr, Write):
            for t in (int(x) for x in instr.tokens):
                state = guide.get_next_state(state, t)
                out.append(t)
                if state.status == COMPLETE:
                    break
            if state.status == COMPLETE:
                break
            continue
        assert isinstance(instr, Generate)
        logits = model(out)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.as_tensor(instr.tokens, dtype=torch.long)] = False
        tok_ids, _anc, _w = sampler(logits.masked_fill(mask, float("-inf")).unsqueeze(0),
                                    torch.zeros(1), rng)
        t = int(tok_ids[0, 0])
        state = guide.get_next_state(state, t)
        out.append(t)
        if state.status == COMPLETE:
            break
    return out, guide.audit


# -- key-format replay header (W3) -------------------------------------------
# The cache-key format the log's mask_entry_ids were recorded under:
#   v1 = legacy raw remainder-bytes generic keys (GRID_GENN_KEYS=0)
#   v2 = genN normalized generic keys (grid/mask/producer.py cache_key)
# entry_id hashes repr(key), so a log replays bit-identically only under ITS
# OWN key format; replaying a v1 log on a genN producer serves entries under
# the v1 keys AND dual-key-checks every consulted config — proving the genN
# merge is a pure re-keying (identical entry payload bytes, ids deterministic).
KEY_FORMAT_RAW, KEY_FORMAT_GENN = "v1", "v2"


def producer_key_format(producer) -> str:
    return KEY_FORMAT_GENN if getattr(producer, "_genn_keys", False) else KEY_FORMAT_RAW


def replay_header(template) -> dict:
    return {"key_format": producer_key_format(template.producer)}


def dual_key_check(prod, configs) -> int:
    """The pure-re-keying proof over every (remainder, A) a replay consulted:
    the entry served under the legacy raw key and under the genN key must
    carry byte-identical canonical mask payloads (adaptive tag + payload) and
    identical cd-group token partitions, and each key form's entry_id must be
    deterministic under recomputation (entry_id formula untouched). Identifier
    configs must key identically under both formats (E11 path untouched)."""
    from grid.mask.cache import adaptive_encode, make_entry

    native = producer_key_format(prod)
    checked = 0
    try:
        for remainder, A in sorted(configs):
            prod.set_genn_keys(False)
            k1, e1 = prod.cache_key(remainder, A), prod._entry_for(remainder, A)
            prod.set_genn_keys(True)
            k2, e2 = prod.cache_key(remainder, A), prod._entry_for(remainder, A)
            if k1[0] == "ident":
                assert k1 == k2 and e1.entry_id == e2.entry_id, \
                    f"ident key changed across formats at {remainder!r}"
                continue
            assert adaptive_encode(e1.ci_tokens, prod.vocab_size) == \
                adaptive_encode(e2.ci_tokens, prod.vocab_size), \
                f"entry bytes differ across key formats at {remainder!r}"

            def part(e):
                return sorted(tuple(sorted(int(t) for t in g.token_ids)) for g in e.cd_groups)

            assert part(e1) == part(e2), \
                f"cd token partition differs across key formats at {remainder!r}"
            for k, e in ((k1, e1), (k2, e2)):
                assert make_entry(k, list(e.ci_tokens), e.cd_entries,
                                  prod.vocab_size).entry_id == e.entry_id, \
                    f"entry_id not deterministic under recomputation for {k!r}"
            checked += 1
    finally:
        prod.set_genn_keys(native == KEY_FORMAT_GENN)
    return checked


def replay_records(template, records, header: dict | None = None) -> list[str]:
    """Replay dispatcher: ``header`` carries the log's key_format (None means
    the producer's native format — same-process logs). An unknown format is a
    hard error. A format differing from the producer's native one replays
    under the LOG's format (entry_ids must match the recorded chain) while
    capturing every consulted config for the dual-key re-keying check."""
    prod = template.producer
    fmt = producer_key_format(prod) if header is None else header.get("key_format")
    if fmt not in (KEY_FORMAT_RAW, KEY_FORMAT_GENN):
        raise ValueError(f"unsupported replay key_format: {fmt!r}")
    native = producer_key_format(prod)
    if fmt == native:
        return _replay_chain(template, records)
    configs: set[tuple[bytes, frozenset]] = set()
    orig = type(prod).cache_key

    def capture(remainder, A):
        configs.add((bytes(remainder), A))
        return orig(prod, remainder, A)

    prod.set_genn_keys(fmt == KEY_FORMAT_GENN)
    prod.cache_key = capture  # instance attr shadows the method for this replay
    try:
        chain = _replay_chain(template, records)
    finally:
        del prod.cache_key
        prod.set_genn_keys(native == KEY_FORMAT_GENN)
    dual_key_check(prod, configs)
    return chain


def _replay_chain(template, records) -> list[str]:
    """Re-drive a guide copy along the recorded chosen tokens, mirroring the
    generation loop's instruction cadence; returns the rebuilt record-hash
    chain. Any structural divergence (span shape, unexpected termination)
    raises — G10 wants bit-identical, not almost."""
    guide = template.copy()
    guide.max_new_tokens = MAX_TOKENS
    state = guide.initial_state
    i = 0
    n = len(records)
    while i < n:
        instr = guide.get_next_instruction(state)
        if isinstance(instr, Write):
            for t in (int(x) for x in instr.tokens):
                if i >= n:
                    raise AssertionError("replay Write span ran past the record chain")
                if t != records[i].chosen_token:
                    raise AssertionError(
                        f"replay diverged at step {i}: write {t} != recorded "
                        f"{records[i].chosen_token}")
                state = guide.get_next_state(state, t)
                i += 1
                if state.status == COMPLETE:
                    break
            if state.status == COMPLETE:
                break
            continue
        t = records[i].chosen_token
        state = guide.get_next_state(state, t)
        i += 1
        if state.status == COMPLETE:
            break
    if i != n:
        raise AssertionError(f"replay consumed {i} of {n} records")
    return [r.record_hash for r in guide.audit.records]


def tamper_trials(logs, trials: int, rng: random.Random) -> int:
    from grid.audit.log import AuditLog

    fields = ["step", "config_hash", "chosen_token", "blocked_count", "instruction_kind"]
    detected = 0
    for _ in range(trials):
        base = rng.choice(logs)
        log = AuditLog(records=list(base.records), sealed=base.sealed,
                       seal_info=dict(base.seal_info))
        i = rng.randrange(len(log.records))
        field = rng.choice(fields)
        rec = log.records[i]
        cur = getattr(rec, field)
        new = cur + 1 if isinstance(cur, int) else ("WRITE" if cur != "WRITE" else "EOS")
        log.records[i] = dataclasses.replace(rec, **{field: new})
        if not log.verify_chain():
            detected += 1
    return detected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=1000)
    ap.add_argument("--tamper-trials", type=int, default=1000)
    ap.add_argument("--assert-gates", action="store_true")
    ap.add_argument("--out", default=str(BENCH_DIR / "RESULTS-g10.md"))
    args = ap.parse_args()

    tok = MockTokenizer(SQL_TOKENS)
    source = (BENCH_DIR.parent / "grammars" / "sql_subset.grid").read_text()
    template = build_guide(source, tok, audit=True)
    sampler = multinomial(1.0)
    rollover_at = args.gens // 2

    t0 = time.perf_counter()
    logs, tokens_total, rollovers = [], 0, 0
    for g in range(args.gens):
        if g == rollover_at:
            template.producer.cache.invalidate_namespace()
            rollovers += 1
        out, audit = generate_one(template, MockModel(tok, seed=g), sampler, seed=g)
        assert audit.verify_chain(), f"generation {g}: chain does not verify"
        assert len(audit.records) == len(out), f"generation {g}: records != tokens"
        logs.append(audit)
        tokens_total += len(out)
    gen_s = time.perf_counter() - t0

    # replay every generation against the POST-rollover cache state
    header = replay_header(template)  # all logs recorded under the native format
    t0 = time.perf_counter()
    identical = 0
    for g, log in enumerate(logs):
        rebuilt = replay_records(template, log.records, header=header)
        original = [r.record_hash for r in log.records]
        if rebuilt == original:
            identical += 1
        else:
            print(f"REPLAY MISMATCH at generation {g}", file=sys.stderr)
    replay_s = time.perf_counter() - t0

    # W3 dual-key compat arm: record a handful of generations under the v1
    # (legacy raw-key) format, then replay them through the genN producer —
    # the v1 header routes them through the dual-key path; chains must still
    # be bit-identical (entries served under the log's keys, contents proven
    # identical under both key forms).
    v1_gens, v1_identical = 0, 0
    if header["key_format"] == KEY_FORMAT_GENN:
        v1_gens = 8
        template.producer.set_genn_keys(False)
        v1_logs = [
            generate_one(template, MockModel(tok, seed=10_000 + g), sampler, 10_000 + g)[1]
            for g in range(v1_gens)
        ]
        template.producer.set_genn_keys(True)
        for g, log in enumerate(v1_logs):
            rebuilt = replay_records(template, log.records, header={"key_format": KEY_FORMAT_RAW})
            if rebuilt == [r.record_hash for r in log.records]:
                v1_identical += 1
            else:
                print(f"V1-LOG REPLAY MISMATCH at generation {g}", file=sys.stderr)

    rng = random.Random(20260709)
    detected = tamper_trials(logs, args.tamper_trials, rng)

    host = os.environ.get("GRID_HOST_LABEL", "local dev (unpinned)")
    ok_replay = identical == args.gens
    ok_tamper = detected == args.tamper_trials
    ok_roll = rollovers >= 1
    ok_v1 = v1_identical == v1_gens
    lines = [
        "# G10 audit replay — full-scale run (E14)",
        "",
        f"Host: {host} | grammar: `grammars/sql_subset.grid` | MockTokenizer "
        f"({len(SQL_TOKENS)} tokens) | mode-1 GRID-owned loop, max_tokens {MAX_TOKENS} "
        f"| key format: {header['key_format']}",
        "",
        f"- v1-log dual-key compat: **{v1_identical}/{v1_gens} bit-identical** "
        "(legacy-key logs replayed through the genN producer; every consulted "
        "config byte-compared under both key forms)",
        f"- generations: **{args.gens}** (seeded multinomial over MockModel logits), "
        f"{tokens_total:,} audited steps total (Write and EOS records included)",
        f"- namespace rollovers spanned: **{rollovers}** (at generation {rollover_at}; "
        "entries recompute content-addressed, replays of pre-rollover generations must "
        "still match)",
        f"- replay: **{identical}/{args.gens} bit-identical record chains** "
        f"(chain hash sequences compared record-by-record; {replay_s:.1f}s)",
        f"- tamper property: **{detected}/{args.tamper_trials} detected** "
        "(random record x random field per trial)",
        f"- generation wall: {gen_s:.1f}s",
        "",
        f"Gate G10: {'**PASS**' if (ok_replay and ok_tamper and ok_roll and ok_v1) else '**FAIL**'} "
        "(criteria: every step of >=1,000 generations replayed bit-identical across "
        ">=1 namespace rollover; tamper detection 100% over >=10^3 trials; v1-format "
        "logs replay bit-identical via the dual-key path).",
        "",
        "Harness: `bench/g10_replay.py` (G10a smoke-scale versions of these "
        "properties run in CI: tests/audit/test_audit.py).",
        "",
    ]
    pathlib.Path(args.out).write_text("\n".join(lines))
    print("\n".join(lines[4:14]))
    print(f"report -> {args.out}")
    if args.assert_gates and not (ok_replay and ok_tamper and ok_roll and ok_v1):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
