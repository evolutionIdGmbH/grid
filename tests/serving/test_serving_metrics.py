"""G8 harness metric reduction + gate logic (pure arithmetic; no vLLM).

The serving numbers come from the GPU box, but the reduction from raw per-step
timings to TTFT/TPOT percentiles + overhead% + gate verdicts is code that must
be right regardless of hardware — pinned here."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))

import vllm_serving_bench as B  # noqa: E402


def test_summarize_basic_percentiles():
    s = B.summarize_arm(
        ttfts_ms=[10, 20, 30, 40],
        tpots_ms=[5.0] * 100,
        step_ms=[5.0] * 100,
        decoded_tokens=100,
        wall_s=1.0,
    )
    assert abs(s["tpot_mean_ms"] - 5.0) < 1e-9
    assert s["throughput_tok_s"] == 100.0
    assert s["n_requests"] == 4
    assert 5.0 <= s["ttft_p99_ms"] <= 40.0


def test_overhead_pct_sign_and_zero():
    assert B.overhead_pct(10.2, 10.0) > 0
    assert B.overhead_pct(10.0, 10.0) == 0.0
    assert B.overhead_pct(9.8, 10.0) < 0


def test_cold_warm_ttft_split_recorded():
    s = B.summarize_arm([18, 1.6, 1.7], [10] * 10, [10] * 10, 10, 1.0,
                        cold_ttfts_ms=[18.0], warm_ttfts_ms=[1.6, 1.7])
    assert s["ttft_cold_p50_ms"] == 18.0
    assert s["ttft_warm_p50_ms"] < 5.0


def test_gate_evaluation_passes_on_good_inputs():
    cells, adv, sf = B.mock_cells([1, 8, 32], ["grid", "xgrammar", "unconstrained"])
    checks = B.evaluate_gates(cells, adv, sf)
    assert checks, "no criteria evaluated"
    failed = [(n, v) for n, ok, v in checks if not ok]
    assert not failed, f"mock inputs should pass every gate, failed: {failed}"
    # the specific binding criteria are present
    names = " ".join(n for n, _o, _v in checks)
    assert "TPOT overhead < 2%" in names
    assert "single build" in names
    assert "skip-a-round" in names


def test_gate_flags_overhead_regression():
    cells, adv, sf = B.mock_cells([32], ["grid", "unconstrained"])
    # force grid TPOT to +10% over unconstrained @32
    base = cells[("unconstrained", 32)]["tpot_mean_ms"]
    cells[("grid", 32)]["tpot_mean_ms"] = base * 1.10
    checks = B.evaluate_gates(cells, adv, sf)
    ov = next(ok for n, ok, _v in checks if "TPOT overhead" in n)
    assert ov is False, "10% overhead must fail the <2% gate"


def test_mock_report_writes_and_passes(tmp_path):
    cells, adv, sf = B.mock_cells([1, 8, 32], ["grid", "xgrammar", "unconstrained"])
    checks = B.evaluate_gates(cells, adv, sf)
    out = tmp_path / "serving.md"
    all_ok = B.write_report(cells, adv, sf, checks, str(out), mock=True)
    assert all_ok
    text = out.read_text()
    assert "MOCK" in text and "Gate G8" in text
    assert "adversarial cold-miss" in text.lower()
    assert "| grid | 32 |" in text
