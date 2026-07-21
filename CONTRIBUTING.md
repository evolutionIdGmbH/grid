# Contributing

- `python -m venv .venv && .venv/bin/pip install -e ".[dev]"`
- `pytest tests/ -q` must be green; correctness changes to `grid/jsonschema/`
  must keep `tests/jsonschema_bridge/test_official_suite.py` green (the
  honesty contract: no false-rejects, silent acceptance only when recorded).
- Benchmark claims follow the repo convention: pinned engine versions,
  declared runners, full error distributions (see bench/RESULTS-*.md).
- The 0.2.x line accepts correctness-only changes; performance work targets
  0.3.x (see DESIGN-JSON-COVERAGE.md for the epoch rules).
