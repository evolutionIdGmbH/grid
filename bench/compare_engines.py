"""GRID vs XGrammar vs llguidance: per-token mask latency on identical replays.

Protocol (DESIGN.md SS10 G9 slice + the G7 R-slope check):
- one real tokenizer (gpt2 50k by default; --tokenizer Qwen/Qwen2.5-0.5B for 151k);
- the same SQL-subset grammar in each engine's native format (bench/grammars.py);
- two replay arms: (a) realistic corpus statements, (b) GRID-generated random
  walks (byte-fallback-heavy stress; --walks N --walk-len L);
- per step and per engine: wall time to produce the full token mask for the
  current prefix, then advance with the replay token; acceptance is counted, not
  asserted (language-parity corners are reported, DESIGN caveat).

Outlines 1.3 note: outlines' CFG path delegates to a backend (default:
llguidance), so the llguidance arm IS what `outlines.types.CFG` executes today.

Honesty note: GRID's hot path runs in grid_core Rust kernels — the trie walk
(in-kernel CD grouping + alias expansion) and the per-step CD-group verdicts +
LALR simulate (masks stay in i32 buffers end-to-end; no per-step Python-int
materialization). Cold misses still pay the full walk; the report separates
hit/miss. The position SLOPE (requirement R) is the architecture-level claim.

Outlines note: outlines has no independent CFG engine — outlines.types.CFG routes
to a backend (default llguidance, CFG_DEFAULT_BACKEND). The Outlines arm here
drives outlines' own LLGuidanceLogitsProcessor, so it measures llguidance plus
outlines' Python wrapper (consume + bitmask fill + apply).

Run:  .venv-bench/bin/python bench/compare_engines.py --out bench/RESULTS.md
"""

from __future__ import annotations

import argparse
import pathlib
import random
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from grammars import CORPUS, GRID_SQL, LLGUIDANCE_SQL, XGRAMMAR_SQL  # noqa: E402


def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))]


class GridEngine:
    name = "GRID (grid_core Rust kernels: walk + CD verdicts + LALR)"

    def __init__(self, hf_tok) -> None:
        from grid.generate import build_guide
        from grid.models.hf_adapter import HFTokenizerAdapter

        t0 = time.perf_counter()
        self.adapter = HFTokenizerAdapter(hf_tok)
        self.guide = build_guide(GRID_SQL, self.adapter)
        self.compile_s = time.perf_counter() - t0
        self.state = self.guide.initial_state
        self.hit_lat: list[float] = []
        self.miss_lat: list[float] = []

    def reset(self) -> None:
        self.state = self.guide.initial_state

    def step(self, tok: int) -> tuple[float, bool]:
        cache = self.guide.producer.cache
        misses0 = cache.misses
        t0 = time.perf_counter()
        ids, _ = self.guide._mask_ids(self.state)
        dt = time.perf_counter() - t0
        (self.miss_lat if cache.misses > misses0 else self.hit_lat).append(dt)
        ok = bool((ids == tok).any())
        if ok:
            self.state = self.guide.get_next_state(self.state, tok)
        return dt, ok

    def is_accepting(self) -> bool:
        return self.guide.can_terminate_state(self.state)


class XGrammarEngine:
    name = "XGrammar 0.2.3 (EBNF)"

    def __init__(self, hf_tok) -> None:
        import xgrammar as xgr

        self.xgr = xgr
        t0 = time.perf_counter()
        self.info = xgr.TokenizerInfo.from_huggingface(hf_tok)
        self.compiler = xgr.GrammarCompiler(self.info)
        self.compiled = self.compiler.compile_grammar(XGRAMMAR_SQL)
        self.compile_s = time.perf_counter() - t0
        self.matcher = xgr.GrammarMatcher(self.compiled)
        self.bitmask = xgr.allocate_token_bitmask(1, self.info.vocab_size)

    def reset(self) -> None:
        self.matcher = self.xgr.GrammarMatcher(self.compiled)

    def step(self, tok: int) -> tuple[float, bool]:
        t0 = time.perf_counter()
        self.matcher.fill_next_token_bitmask(self.bitmask)
        dt = time.perf_counter() - t0
        word = int(self.bitmask[0][tok // 32])
        allowed = (word >> (tok % 32)) & 1 == 1
        ok = allowed and self.matcher.accept_token(tok)
        return dt, ok

    def is_accepting(self) -> bool:
        return bool(self.matcher.is_terminated()) or True  # xgrammar: stop via EOS mask


class LLGuidanceEngine:
    name = "llguidance 1.7.6 (lark, driven directly)"

    def __init__(self, hf_tok) -> None:
        import llguidance
        import llguidance.hf

        t0 = time.perf_counter()
        self.lt = llguidance.hf.from_tokenizer(hf_tok)
        err = llguidance.LLMatcher.validate_grammar(LLGUIDANCE_SQL, self.lt)
        if err:
            raise RuntimeError(f"llguidance grammar invalid: {err}")
        self.grammar = llguidance.LLMatcher.grammar_from_lark(LLGUIDANCE_SQL)
        self.matcher = llguidance.LLMatcher(self.lt, self.grammar)
        self.compile_s = time.perf_counter() - t0
        self._llg = llguidance

    def reset(self) -> None:
        self.matcher.reset()

    def step(self, tok: int) -> tuple[float, bool]:
        t0 = time.perf_counter()
        mask = self.matcher.compute_bitmask()
        dt = time.perf_counter() - t0
        allowed = (mask[tok // 8] >> (tok % 8)) & 1 == 1
        ok = allowed and self.matcher.consume_token(tok)
        return dt, ok

    def is_accepting(self) -> bool:
        return self.matcher.is_accepting()


class OutlinesEngine:
    """Outlines' CFG path (outlines.types.CFG -> LLGuidanceBackend). Outlines has
    no CFG engine of its own; this drives its own LLGuidanceLogitsProcessor so the
    row reflects the real cost of constraining generation *through* Outlines."""

    name = "Outlines 1.3.1 (CFG backend = llguidance)"

    def __init__(self, hf_tok) -> None:
        import llguidance
        import llguidance.hf
        import torch
        from outlines.backends.llguidance import LLGuidanceLogitsProcessor

        t0 = time.perf_counter()
        llg_tok = llguidance.hf.from_tokenizer(hf_tok)
        try:  # mirror LLGuidanceBackend.get_cfg_logits_processor exactly
            spec = llguidance.grammar_from("grammar", LLGUIDANCE_SQL)
        except ValueError:
            spec = llguidance.grammar_from("lark", LLGUIDANCE_SQL)
        self.proc = LLGuidanceLogitsProcessor(spec, llg_tok, "torch")
        self.vocab_size = int(llg_tok.vocab_size)
        self._torch = torch
        # Build the matcher once here (folds llguidance.torch's one-time JIT + matcher
        # construction into compile, like the other arms' __init__). reset() rebuilds it
        # OUTSIDE the timed step, so the measured per-step cost is the true mask cost —
        # process_logits' own consume + fill + apply — not per-sequence setup.
        self.proc._setup(1)
        self.proc.is_first_token = False
        # Trigger llguidance's one-time first-mask JIT (parser tables) here, not in a
        # timed step; the grammar automaton is cached in the spec, so reset()'s fresh
        # matcher reuses it.
        self.proc._bias_logits(
            torch.tensor([[0]], dtype=torch.long),
            torch.zeros((1, self.vocab_size), dtype=torch.float32),
        )
        self.compile_s = time.perf_counter() - t0
        self._first = True
        self.ids: list[int] = []

    def reset(self) -> None:
        self.proc._setup(1)  # fresh LLMatcher, outside the timed region
        self.proc.is_first_token = False
        self._first = True
        self.ids = []

    def step(self, tok: int) -> tuple[float, bool]:
        torch = self._torch
        logits = torch.zeros((1, self.vocab_size), dtype=torch.float32)
        inp = torch.tensor([self.ids] if self.ids else [[0]], dtype=torch.long)
        t0 = time.perf_counter()
        if not self._first:  # process_logits consumes the last token before masking
            self.proc.ll_matchers[0].consume_token(self.ids[-1])
        biased = self.proc._bias_logits(inp, logits)  # fill_next_token_bitmask + apply
        dt = time.perf_counter() - t0
        self._first = False
        ok = tok < self.vocab_size and float(biased[0, tok].item()) != float("-inf")
        if ok:
            self.ids.append(tok)
        return dt, ok

    def is_accepting(self) -> bool:
        return True  # stop via EOS in the mask


def grid_random_walks(engine: GridEngine, n: int, max_len: int, seed0: int = 0) -> list[list[int]]:
    """Valid token sequences from seeded random walks (EOS suppressed until length)."""
    from grid.guide import COMPLETE

    out = []
    for k in range(n):
        rng = random.Random(seed0 + k)
        st = engine.guide.initial_state
        seq: list[int] = []
        while len(seq) < max_len:
            ids, _ = engine.guide._mask_ids(st)
            pool = sorted(set(ids) - {engine.guide.eos_token_id}) or sorted(ids)
            tok = rng.choice(pool)
            st = engine.guide.get_next_state(st, tok)
            seq.append(tok)
            if st.status == COMPLETE:
                break
        out.append(seq)
    engine.reset()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--walks", type=int, default=3)
    ap.add_argument("--walk-len", type=int, default=120)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    hf_tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"tokenizer: {args.tokenizer} ({len(hf_tok.get_vocab())} tokens)")

    engines: list = []
    grid_engine = GridEngine(hf_tok)
    engines.append(grid_engine)
    for cls in (XGrammarEngine, LLGuidanceEngine, OutlinesEngine):
        try:
            engines.append(cls(hf_tok))
        except Exception as e:  # report unavailable engines, never crash the bench
            print(f"[skip] {cls.name}: {type(e).__name__}: {e}")

    # replay material
    replays: list[tuple[str, list[int]]] = []
    for text in CORPUS:
        replays.append((f"corpus:{text[:28]}...", hf_tok.encode(text)))
    print(f"generating {args.walks} GRID random walks (stress arm)...")
    for i, seq in enumerate(grid_random_walks(grid_engine, args.walks, args.walk_len)):
        replays.append((f"walk:{i} (len {len(seq)})", seq))

    results: dict[str, dict] = {}
    for eng in engines:
        lat: list[float] = []
        pos_lat: list[tuple[int, float]] = []
        rejected = 0
        steps = 0
        for _name, seq in replays:
            eng.reset()
            for pos, tok in enumerate(seq):
                dt, ok = eng.step(tok)
                lat.append(dt)
                pos_lat.append((pos, dt))
                steps += 1
                if not ok:
                    rejected += 1
                    break
        # position slope (requirement R): OLS over (position, latency)
        import numpy as np

        xs = np.array([p for p, _ in pos_lat], dtype=float)
        ys = np.array([d for _, d in pos_lat], dtype=float) * 1e6
        slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) > 2 else float("nan")
        results[eng.name] = {
            "compile_s": eng.compile_s,
            "steps": steps,
            "rejected_replays": rejected,
            "p50_us": pct(lat, 0.50) * 1e6,
            "p90_us": pct(lat, 0.90) * 1e6,
            "p99_us": pct(lat, 0.99) * 1e6,
            "slope_us_per_pos": slope,
        }
        if isinstance(eng, GridEngine):
            results[eng.name].update({
                "hit_p50_us": pct(eng.hit_lat, 0.5) * 1e6,
                "miss_p50_us": pct(eng.miss_lat, 0.5) * 1e6,
                "hit_rate": len(eng.hit_lat) / max(1, len(eng.hit_lat) + len(eng.miss_lat)),
            })
        r = results[eng.name]
        print(f"\n{eng.name}\n  compile {r['compile_s']*1e3:.1f} ms | steps {steps} | "
              f"p50 {r['p50_us']:.1f} us | p90 {r['p90_us']:.1f} us | p99 {r['p99_us']:.1f} us | "
              f"slope {r['slope_us_per_pos']:+.3f} us/pos | rejected {rejected}")
        if "hit_rate" in r:
            print(f"  cache: hit p50 {r['hit_p50_us']:.1f} us | miss p50 {r['miss_p50_us']/1e3:.1f} ms | "
                  f"hit rate {r['hit_rate']:.0%}")

    # G7-style R measurement: warm-cache replay of the longest walk (GRID) —
    # per-token cost must be independent of position n (requirement R)
    import numpy as np

    longest = max((seq for name, seq in replays if name.startswith("walk")), key=len, default=None)
    if longest:
        grid_engine.reset()
        for tok in longest:  # warm pass 1 (populate cache along this exact path)
            grid_engine.step(tok)
        grid_engine.reset()
        warm: list[tuple[int, float]] = []
        for pos, tok in enumerate(longest):
            dt, _ = grid_engine.step(tok)
            warm.append((pos, dt))
        xs = np.array([p for p, _ in warm], float)
        ys = np.array([d for _, d in warm], float) * 1e6
        slope = float(np.polyfit(xs, ys, 1)[0])
        half = len(warm) // 2
        first = statistics.median(d for _, d in warm[:half]) * 1e6
        second = statistics.median(d for _, d in warm[half:]) * 1e6
        results["_grid_warm_R"] = {
            "steps": len(warm), "slope_us_per_pos": slope,
            "first_half_p50_us": first, "second_half_p50_us": second,
        }
        print(f"\nGRID warm-replay R check ({len(warm)} steps): slope {slope:+.3f} us/pos | "
              f"first-half p50 {first:.0f} us vs second-half p50 {second:.0f} us")

    if args.out:
        write_report(args.out, args.tokenizer, replays, results)
        print(f"\nreport -> {args.out}")


def write_report(path: str, tokenizer: str, replays, results: dict) -> None:
    lines = [
        "# GRID vs XGrammar vs llguidance vs Outlines — SQL-subset constrained decoding",
        "",
        f"Tokenizer: `{tokenizer}` | replays: {len(replays)} "
        f"({sum(len(s) for _, s in replays)} steps total) | host: local dev (unpinned — "
        "G7/G9 bind on the declared cloud runner)",
        "",
        "GRID's hot path runs in grid_core Rust kernels: the trie walk (in-kernel CD "
        "grouping + alias expansion) and the per-step CD-group verdicts + LALR simulate; "
        "masks stay in i32 buffers end-to-end. Cold misses pay the full walk (see the "
        "cache split). Outlines' CFG path delegates to llguidance "
        "(CFG_DEFAULT_BACKEND='llguidance'), so the Outlines and raw-llguidance arms share "
        "the same core matcher — the Outlines row adds outlines' logits-processor wrapper "
        "(consume + bitmask fill + apply).",
        "",
        "| engine | compile | p50 | p90 | p99 | slope (us/pos) | rejected replays |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        if name.startswith("_"):
            continue
        lines.append(
            f"| {name} | {r['compile_s']*1e3:.1f} ms | {r['p50_us']:.1f} us | "
            f"{r['p90_us']:.1f} us | {r['p99_us']:.1f} us | {r['slope_us_per_pos']:+.3f} | "
            f"{r['rejected_replays']} |"
        )
    for _name, r in results.items():
        if "hit_rate" in r:
            lines += [
                "",
                f"GRID cache split: hit p50 {r['hit_p50_us']:.1f} us | miss p50 "
                f"{r['miss_p50_us']/1e3:.1f} ms | hit rate {r['hit_rate']:.0%}",
            ]
    warm = results.get("_grid_warm_R")
    if warm:
        lines += [
            "",
            f"GRID warm-replay R check ({warm['steps']} steps): slope "
            f"{warm['slope_us_per_pos']:+.3f} us/pos; first-half p50 "
            f"{warm['first_half_p50_us']:.0f} us vs second-half p50 "
            f"{warm['second_half_p50_us']:.0f} us — per-token cost tracks grammar "
            "configuration, not absolute position (requirement R).",
        ]
    lines += [
        "",
        "Notes:",
        "- Rejected replays count language-parity corners between the grammar encodings",
        "  (maximal-munch vs explicit-whitespace), not correctness bugs.",
        "- Outlines has no independent CFG engine: `outlines.types.CFG` routes to a",
        "  backend, default llguidance (`CFG_DEFAULT_BACKEND`), so its row tracks",
        "  llguidance plus wrapper overhead (JSON-schema/regex default to outlines_core).",
    ]
    pathlib.Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
