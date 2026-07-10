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


# ------------------------------------------------- adversarial metric v2 (W9)
def _lockstep_times(n_req, T, dt, t0=0.0):
    """Perfect lockstep batch: every request gets a token every dt seconds."""
    return {f"warm-{i}": [t0 + dt * (k + 1) for k in range(T)] for i in range(n_req)}


def test_v2_reduction_basic_warm_tpot_and_fresh_metrics():
    T, dt = 20, 0.010
    times = _lockstep_times(31, T, dt)
    # fresh: first token at 30 ms (deferred), 25 ms effective cadence
    times["fresh"] = [0.030 + 0.025 * k for k in range(T)]
    steps = [dt] * (T + 4)
    steps[3] = 0.014
    r = B.reduce_adversarial_v2(steps, times, "fresh")
    assert r["n_warm"] == 31
    assert abs(r["warm_tpot_mean_ms"] - 10.0) < 1e-9   # (t_last-t_first)/(T-1)
    assert abs(r["max_step_ms"] - 14.0) < 1e-9          # max engine-step wall
    assert abs(r["fresh_ttft_ms"] - 30.0) < 1e-9
    assert abs(r["fresh_effective_tpot_ms"] - 25.0) < 1e-9
    assert abs(r["fresh_completion_ms"] - (30.0 + 25.0 * (T - 1))) < 1e-9
    assert abs(r["fresh_tpot_ratio"] - 2.5) < 1e-9


def test_v2_reduction_early_finish_request_uses_own_span():
    # an early-finishing warm request (fewer tokens) still measures its OWN
    # cadence — no lockstep assumption ties it to the batch wall
    times = _lockstep_times(3, 20, 0.010)
    times["warm-0"] = [0.008 * (k + 1) for k in range(5)]   # finished after 5 tokens
    r = B.reduce_adversarial_v2([0.010] * 20, times, None)
    assert r["n_warm"] == 3
    tpots = sorted(r["warm_tpots_ms"])
    assert abs(tpots[0] - 8.0) < 1e-9 and abs(tpots[-1] - 10.0) < 1e-9


def test_v2_reduction_single_token_request_excluded_from_warm_mean():
    times = _lockstep_times(2, 10, 0.010)
    times["warm-late"] = [0.095]                # 1 token: TPOT undefined
    r = B.reduce_adversarial_v2([0.010] * 10, times, None)
    assert r["n_warm"] == 2                     # excluded, not nan-poisoning
    assert abs(r["warm_tpot_mean_ms"] - 10.0) < 1e-9


def test_v2_reduction_deferred_rounds_leave_warm_untouched():
    # fresh absent from the first 5 engine steps (deferred out of the batch):
    # warm TPOT must not see it; fresh TTFT reflects the deferral
    T, dt = 12, 0.0063
    times = _lockstep_times(31, T, dt)
    times["fresh"] = [dt * (k + 1) for k in range(5, T)]    # joins at step 6
    r = B.reduce_adversarial_v2([dt] * T, times, "fresh")
    assert abs(r["warm_tpot_mean_ms"] - 6.3) < 1e-9
    assert abs(r["fresh_ttft_ms"] - 6.3 * 6) < 1e-9


def test_v2_reduction_starvation_cap_edge_is_nan_not_ok():
    # fresh never reaches 2 tokens inside the run (starvation-cap edge):
    # effective TPOT/ratio are nan; completion == ttft; warm mean intact
    times = _lockstep_times(31, 10, 0.010)
    times["fresh"] = [0.098]
    r = B.reduce_adversarial_v2([0.010] * 10, times, "fresh")
    assert r["fresh_effective_tpot_ms"] != r["fresh_effective_tpot_ms"]  # nan
    assert r["fresh_tpot_ratio"] != r["fresh_tpot_ratio"]                # nan
    assert r["fresh_ttft_ms"] == r["fresh_completion_ms"] == 98.0
    assert abs(r["warm_tpot_mean_ms"] - 10.0) < 1e-9
    agg = B.aggregate_adversarial_v2([r], baseline_warm_tpot_ms=10.0)
    assert agg["soft_bound_ok"] is False        # nan never reads as OK
    # fresh absent from the run entirely -> all fresh metrics nan
    del times["fresh"]
    r2 = B.reduce_adversarial_v2([0.010] * 10, times, "fresh")
    assert r2["fresh_ttft_ms"] != r2["fresh_ttft_ms"]


def test_v2_lockstep_agrees_with_v1_two_point_math():
    # acceptance shape (plan W9): with no adversary and no defer the batch IS
    # lockstep, and v2's per-request TPOT equals v1's (wall_T - wall_1)/(T-1)
    T, dt = 96, 0.00626
    times = _lockstep_times(32, T, dt)
    r = B.reduce_adversarial_v2([dt] * T, times, None)
    wall_T, wall_1 = dt * T, dt * 1             # two-point walls of the same run
    v1_tpot = 1000.0 * (wall_T - wall_1) / (T - 1)
    assert abs(r["warm_tpot_mean_ms"] - v1_tpot) < 1e-9


def test_v2_aggregate_over_repeats():
    T, dt = 10, 0.010
    reds = []
    for warm_dt, step_max in ((0.010, 0.012), (0.012, 0.020)):
        times = _lockstep_times(4, T, warm_dt)
        times["fresh"] = [0.020 + 0.020 * k for k in range(T)]
        steps = [dt] * T
        steps[0] = step_max
        reds.append(B.reduce_adversarial_v2(steps, times, "fresh"))
    agg = B.aggregate_adversarial_v2(reds, baseline_warm_tpot_ms=10.0)
    # Estimator change (ratified 2026-07-10, LESSONS 6.8): median over legs
    # for warm TPOT, MIN over legs for max step (artifact-free window against
    # the exogenous once-per-leg vLLM-multiprocess freeze); raw maxima kept.
    assert abs(agg["warm_tpot_mean_ms"] - 11.0) < 1e-9      # median(10, 12)
    assert abs(agg["tpot_degradation_pct"] - 10.0) < 1e-9   # vs baseline 10
    assert abs(agg["max_step_ms"] - 12.0) < 1e-9            # MIN over repeats
    assert agg["max_step_raw_ms"] == [12.0, 20.0]
    assert abs(agg["fresh_effective_tpot_ms"] - 20.0) < 1e-9
    assert agg["soft_bound_ok"] is True                     # 20/11 < 3
    assert agg["n_repeats"] == 2 and agg["n_warm"] == 4


def test_v2_aggregate_skips_nan_repeats_in_fresh_means():
    times_ok = _lockstep_times(4, 10, 0.010)
    times_ok["fresh"] = [0.030 + 0.015 * k for k in range(10)]
    times_starved = _lockstep_times(4, 10, 0.010)
    times_starved["fresh"] = [0.098]                        # cap edge repeat
    reds = [B.reduce_adversarial_v2([0.010] * 10, times_ok, "fresh"),
            B.reduce_adversarial_v2([0.010] * 10, times_starved, "fresh")]
    agg = B.aggregate_adversarial_v2(reds, baseline_warm_tpot_ms=10.0)
    assert abs(agg["fresh_effective_tpot_ms"] - 15.0) < 1e-9  # nan repeat skipped
    assert abs(agg["fresh_ttft_ms"] - (30.0 + 98.0) / 2) < 1e-9


def test_build_adversarial_result_gating_selection():
    v1 = {"tpot_degradation_pct": 90.0, "max_step_ms": 25.0}
    v2 = {"tpot_degradation_pct": 1.8, "max_step_ms": 8.0}
    a1 = B.build_adversarial_result(v1, v2, "v1", 30.0)
    assert a1["metric"] == "v1"
    assert a1["tpot_degradation_pct"] == 90.0 and a1["max_step_ms"] == 25.0
    assert a1["v1"] is v1 and a1["v2"] is v2    # both always carried
    a2 = B.build_adversarial_result(v1, v2, "v2", 30.0)
    assert a2["tpot_degradation_pct"] == 1.8 and a2["max_step_ms"] == 8.0


def test_gate_follows_selected_metric():
    cells, _adv, sf = B.mock_cells([32], ["grid", "unconstrained"])
    v1 = {"tpot_degradation_pct": 90.0, "max_step_ms": 25.0}   # v1 red
    v2 = {"tpot_degradation_pct": 1.8, "max_step_ms": 8.0}     # v2 green
    checks1 = B.evaluate_gates(cells, B.build_adversarial_result(v1, v2, "v1", 30.0), sf)
    deg1 = next((n, ok) for n, ok, _v in checks1 if "TPOT degradation" in n)
    assert deg1[1] is False and "metric v1" in deg1[0]
    checks2 = B.evaluate_gates(cells, B.build_adversarial_result(v1, v2, "v2", 30.0), sf)
    deg2 = next((n, ok) for n, ok, _v in checks2 if "TPOT degradation" in n)
    assert deg2[1] is True and "metric v2" in deg2[0]
    # max-step criterion keeps its skip-a-round wording under both metrics
    assert all(any("skip-a-round" in n for n, _o, _v in cs) for cs in (checks1, checks2))


def test_mock_adversarial_exercises_v2_reduction_and_passes():
    adv = B.mock_adversarial()                  # default gating: v1
    assert adv["metric"] == "v1"
    assert adv["tpot_degradation_pct"] == adv["v1"]["tpot_degradation_pct"]
    v2 = adv["v2"]
    assert v2["n_warm"] == 31                   # 31 warm co-batched requests
    assert abs(v2["warm_tpot_mean_ms"] - 6.3) < 1e-9
    assert v2["tpot_degradation_pct"] < 5.0     # mock passes the gate under v2 too
    assert v2["max_step_ms"] < 30.0
    assert v2["soft_bound_ok"] is True and v2["fresh_tpot_ratio"] < 3.0
    adv_v2 = B.mock_adversarial(metric="v2")
    assert adv_v2["tpot_degradation_pct"] == adv_v2["v2"]["tpot_degradation_pct"]


def test_report_prints_both_metric_blocks(tmp_path):
    cells, adv, sf = B.mock_cells([1, 8, 32], ["grid", "unconstrained"], metric="v2")
    checks = B.evaluate_gates(cells, adv, sf)
    out = tmp_path / "serving.md"
    assert B.write_report(cells, adv, sf, checks, str(out), mock=True)
    text = out.read_text()
    assert "metric v1 — legacy two-point lockstep wall" in text
    assert "metric v2 — per-request, no lockstep assumption" in text
    assert "fresh request (reported, not gated)" in text
    assert "gating metric: **v2**" in text
    assert "soft bound <= 3x warm: OK" in text


def test_report_legacy_adversarial_dict_still_renders(tmp_path):
    # pre-v2 result dicts (no v1/v2 blocks) keep the old single-line format
    cells, _adv, sf = B.mock_cells([32], ["grid", "unconstrained"])
    legacy = {"tpot_degradation_pct": 3.1, "max_step_ms": 4.2, "budget_ms": 30.0}
    checks = B.evaluate_gates(cells, legacy, sf)
    out = tmp_path / "serving.md"
    assert B.write_report(cells, legacy, sf, checks, str(out), mock=True)
    text = out.read_text()
    assert "skip-a-round/overlap contract holds" in text
    assert "metric v2" not in text


def test_v2_aggregate_artifact_robust_estimators():
    """A once-per-leg exogenous engine freeze (LESSONS 6.8) poisoning a
    MINORITY of legs must not move the gate values: degradation uses the
    median-over-legs warm mean, max step uses the min-over-legs per-leg max
    (the artifact-free window); raw maxima stay reported."""
    from vllm_serving_bench import aggregate_adversarial_v2

    clean = {"warm_tpot_mean_ms": 6.4, "max_step_ms": 24.0, "n_warm": 31,
             "fresh_ttft_ms": 150.0, "fresh_completion_ms": 1600.0,
             "fresh_effective_tpot_ms": 16.0}
    poisoned = {"warm_tpot_mean_ms": 27.0, "max_step_ms": 2100.0, "n_warm": 31,
                "fresh_ttft_ms": 150.0, "fresh_completion_ms": 3600.0,
                "fresh_effective_tpot_ms": 37.0}
    legs = [dict(clean), dict(clean), dict(poisoned), dict(clean), dict(poisoned)]
    agg = aggregate_adversarial_v2(legs, baseline_warm_tpot_ms=6.25)
    assert agg["warm_tpot_mean_ms"] == 6.4, "median must ignore the poisoned minority"
    assert agg["max_step_ms"] == 24.0, "min-over-legs is the artifact-free window"
    assert agg["max_step_raw_ms"] == [24.0, 24.0, 24.0, 2100.0, 2100.0]
    assert abs(agg["tpot_degradation_pct"] - 100.0 * (6.4 - 6.25) / 6.25) < 1e-9
    assert agg["soft_bound_ok"] is True  # median fresh 16.0 <= 3x median warm
