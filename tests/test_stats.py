import pytest

from raggate.stats import (cohens_kappa, holm, kappa_interval, make_resamples, paired_bootstrap)


def test_holm_matches_a_hand_computed_example():
    # sorted: 0.01*4=0.04, 0.02*3=0.06, 0.03*2=0.06->max(0.06,0.06), 0.04*1 -> max(0.06,0.04)=0.06
    assert holm([0.04, 0.01, 0.03, 0.02]) == pytest.approx([0.06, 0.04, 0.06, 0.06])


def test_holm_is_never_smaller_than_the_raw_p_and_capped_at_one():
    ps = [0.5, 0.02, 0.9, 0.001]
    adj = holm(ps)
    assert all(a >= p for a, p in zip(adj, ps)) and max(adj) <= 1.0
    assert holm([0.6, 0.7]) == [1.0, 1.0]
    assert holm([0.01]) == [0.01]


def test_holm_is_monotone_in_the_sorted_order():
    ps = [0.001, 0.2, 0.011, 0.04, 0.03]
    adj = holm(ps)
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    assert [adj[i] for i in order] == sorted(adj[i] for i in order)


def test_bootstrap_of_no_change_is_not_significant():
    res = paired_bootstrap([0.0] * 50, make_resamples(50, 300, 1))
    assert res.diff == 0 and res.p_worse == 1.0 and res.lo == res.hi == 0


def test_bootstrap_of_a_clear_drop_is_significant_and_the_interval_excludes_zero():
    diffs = [-1.0] * 12 + [0.0] * 88
    res = paired_bootstrap(diffs, make_resamples(100, 1000, 1))
    assert res.diff == pytest.approx(-0.12)
    assert res.hi < 0 and res.p_worse < 0.01


def test_a_single_failure_among_many_is_not_significant():
    res = paired_bootstrap([-1.0] + [0.0] * 99, make_resamples(100, 1000, 1))
    assert res.p_worse > 0.2


def test_bootstrap_is_paired_so_symmetric_churn_cancels():
    # 10 questions got worse and 10 got better: the mean change is exactly zero
    res = paired_bootstrap([-1.0] * 10 + [1.0] * 10 + [0.0] * 80, make_resamples(100, 1000, 1))
    assert res.diff == 0 and res.lo < 0 < res.hi and res.p_worse > 0.3


def test_undefined_entries_are_excluded_from_the_mean_and_n():
    diffs = [None] * 50 + [-1.0] * 10 + [0.0] * 40
    res = paired_bootstrap(diffs, make_resamples(100, 500, 1))
    assert res.n == 50 and res.diff == pytest.approx(-0.2)
    assert paired_bootstrap([None] * 10, make_resamples(10, 50, 1)) is None


def test_bootstrap_is_reproducible_for_a_seed_and_needs_two_questions():
    d = [-1.0, 0.0, 0.0, 1.0, -1.0, 0.0] * 10
    a = paired_bootstrap(d, make_resamples(60, 200, 7))
    b = paired_bootstrap(d, make_resamples(60, 200, 7))
    assert a == b
    with pytest.raises(ValueError):
        make_resamples(1, 10, 1)


def test_kappa_matches_the_textbook_example():
    # 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no  -> po=0.7, pe=0.5 -> kappa=0.4
    a = [True] * 25 + [False] * 25
    b = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 15
    assert cohens_kappa(a, b) == pytest.approx(0.4)


def test_kappa_is_zero_for_a_judge_that_always_says_pass_whatever_the_accuracy():
    truth = [True] * 80 + [False] * 20
    always = [True] * 100
    assert sum(t == s for t, s in zip(truth, always)) / 100 == 0.8        # accuracy looks fine
    assert cohens_kappa(truth, always) == pytest.approx(0.0)


def test_kappa_edge_cases():
    assert cohens_kappa([True, False] * 5, [True, False] * 5) == 1.0
    with pytest.raises(ValueError):
        cohens_kappa([True] * 5, [True] * 5)
    with pytest.raises(ValueError):
        cohens_kappa([True], [True, False])


def test_kappa_interval_brackets_the_estimate():
    a = [True] * 25 + [False] * 25
    b = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 15
    lo, hi = kappa_interval(a, b)
    assert lo < cohens_kappa(a, b) < hi
