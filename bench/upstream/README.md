# Upstream PR kit: GRID adapter for guidance-ai/jsonschemabench

Contents of the intended PR:

1. `grid_engine.py` -> copy to `maskbench/maskbench/grid_engine.py`
2. `runner.py` registration (add alongside the other engines):

```python
    parser.add_argument("--grid", action="store_true", help="Use GRID")
    parser.add_argument("--grid-strict", action="store_true",
                        help="Use GRID in strict (declared) mode")
...
    if args.grid or args.grid_strict:
        from .grid_engine import GridEngine
        assert not engine, "Multiple engines specified"
        engine = GridEngine(strict=args.grid_strict)
```

3. `requirements` addition: `grid-guardrail`

Run: `python -m maskbench.runner --grid --tokenizer <model> data/`

Numbers to accompany the PR (full 11,306-schema set, one machine,
llguidance 1.7.6 / XGrammar 0.2.3 / GRID 0.4.0 - zero timeouts,
passing 10,159, TTFM avg 66ms):
see `bench/RESULTS-jsonschemabench-v0.4.0rc1.md` in the GRID repo.
