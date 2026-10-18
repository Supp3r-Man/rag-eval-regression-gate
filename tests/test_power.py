import pytest

from raggate.power import PowerCell, critical_false_alarm, min_detectable_drop, power_table, simulate_power

FAST = dict(trials=60, boot=150, family=3)


def test_critical_false_alarm_matches_hand_computation():
    assert critical_false_alarm(1, 0.1) == pytest.approx(0.1)           # one question: only 'worse' counts
    # two questions, per-question: worse .1, better .1, same .8
    # net >= 1: (worse,same)x2 + (worse,worse) + (worse... ) enumerate: P(net>=1)
    p = 0.1 * 0.8 * 2 + 0.1 * 0.1                                        # one worse+one same (2 orders), both worse
    assert critical_false_alarm(2, 0.1) == pytest.approx(p)
    assert critical_false_alarm(10, 0.0) == 0.0


def test_critical_false_alarm_grows_with_slice_size_and_churn_and_with_a_laxer_threshold():
    assert critical_false_alarm(10, 0.05) < critical_false_alarm(40, 0.05)
    assert critical_false_alarm(20, 0.02) < critical_false_alarm(20, 0.05)
    assert critical_false_alarm(20, 0.05, min_net_drop=3) < critical_false_alarm(20, 0.05, min_net_drop=1)


def test_simulation_is_deterministic_for_a_seed():
    assert simulate_power(100, 0.1, 0.04, **FAST) == simulate_power(100, 0.1, 0.04, **FAST)


def test_a_bootstrap_too_coarse_to_ever_reach_the_family_threshold_is_refused():
    with pytest.raises(ValueError, match="too coarse"):
        simulate_power(100, 0.1, 0.04, trials=2, boot=100)


def test_no_change_and_no_churn_never_fires():
    c = simulate_power(100, 0.0, 0.0, **FAST)
    assert (c.single, c.family) == (0.0, 0.0)


def test_a_large_real_drop_is_almost_always_caught_and_a_tiny_one_almost_never():
    big = simulate_power(200, 0.20, 0.04, trials=40, boot=400)          # default family=15 needs boot >= 300
    tiny = simulate_power(200, 0.005, 0.04, **FAST)
    assert big.single >= 0.95 and big.family >= 0.9
    assert tiny.single < 0.3


def test_the_family_threshold_is_never_more_powerful_than_a_lone_test():
    cells = power_table(ns=(100,), drops=(0.05, 0.10), churn=0.04, **FAST)
    assert all(c.family <= c.single for c in cells)


def test_power_rises_with_sample_size_for_a_fixed_drop():
    small = simulate_power(40, 0.10, 0.02, trials=120, boot=150, family=3)
    large = simulate_power(400, 0.10, 0.02, trials=120, boot=150, family=3)
    assert large.single > small.single


def test_false_alarm_rate_with_no_real_change_is_low():
    c = simulate_power(150, 0.0, 0.04, trials=120, boot=400)
    assert c.single < 0.15 and c.family < 0.05


def test_min_detectable_drop_picks_the_smallest_grid_drop_with_enough_power():
    cells = [PowerCell(100, 0.02, .04, 0.1, 0.0), PowerCell(100, 0.05, .04, 0.9, 0.85),
             PowerCell(100, 0.10, .04, 1.0, 1.0), PowerCell(50, 0.10, .04, 0.5, 0.2)]
    assert min_detectable_drop(cells, 100) == 0.05
    assert min_detectable_drop(cells, 50) is None
    assert min_detectable_drop(cells, 100, family=False) == 0.05
