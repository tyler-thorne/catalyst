"""Statistical property tests for the bootstrap CI engine.

Synthetic populations with a known true effect, shaped like real telemetry:
Glean-style exponential buckets, a heavy right tail, and clients that differ
in both speed and activity (a few heavy clients contribute many observations).
"""

import numpy as np
import pandas as pd
import pytest

from lib.analysis import (
    calc_bootstrap_ci,
    calc_cdf_band,
    calc_histogram_percentile,
    compute_client_cluster_bootstrap,
)


def glean_bucket(values):
    """Lower edge of a timing_distribution-like bucket (8 buckets per octave)."""
    exponent = np.floor(8 * np.log2(np.maximum(values, 1))) / 8
    return np.floor(2**exponent).astype(int)


def make_clients(rng, n_clients, effect=1.0, obs_per_client=None, branch="control"):
    """One row per client in the shape compute_client_cluster_bootstrap expects.

    `effect` multiplies every observation, so the true percentile lift is
    effect - 1. `obs_per_client` fixes activity; by default it's heavy-tailed.
    """
    speed = rng.normal(6.0, 0.6, n_clients)
    if obs_per_client is None:
        n_obs = np.minimum(rng.lognormal(2.5, 1.0, n_clients).astype(int) + 1, 2000)
    else:
        n_obs = np.full(n_clients, obs_per_client)
    rows = []
    for client in range(n_clients):
        values = glean_bucket(effect * rng.lognormal(speed[client], 0.5, n_obs[client]))
        buckets, counts = np.unique(values, return_counts=True)
        rows.append(
            {
                "segment": "Windows",
                "branch": branch,
                "client_id": f"c{client}",
                "bucket_counts": [
                    {"bucket": int(b), "counts": int(c)}
                    for b, c in zip(buckets, counts)
                ],
            }
        )
    return pd.DataFrame(rows)


def lift_ci(control_df, treatment_df, n_bootstrap=500, slug="test"):
    """Run the engine end to end for one control/treatment pair."""
    replicates = {}
    for name, df in (("control", control_df), ("treatment", treatment_df)):
        replicates[name] = compute_client_cluster_bootstrap(
            df.assign(branch=name), n_bootstrap, slug, "metric"
        )
    bins = sorted(
        set(replicates["control"].bucket) | set(replicates["treatment"].bucket)
    )

    def aligned(rep):
        rep = rep.set_index("bucket").reindex(bins)
        return [
            r if isinstance(r, np.ndarray) else np.zeros(n_bootstrap)
            for r in rep.bootstrap_replicates
        ]

    return calc_bootstrap_ci(
        bins, aligned(replicates["control"]), bins, aligned(replicates["treatment"])
    )


def width(ci, q=0.5):
    return ci[q]["upper"] - ci[q]["lower"]


# Engine mechanics


def test_replicates_average_to_observed_counts():
    df = make_clients(np.random.default_rng(1), 500)
    out = compute_client_cluster_bootstrap(df, 1000, "test", "metric")
    observed = out.counts.sum()
    resampled = np.mean(np.sum(np.stack(out.bootstrap_replicates), axis=0))
    assert resampled == pytest.approx(observed, rel=0.03)


def test_each_client_gets_one_weight_across_its_buckets():
    df = pd.DataFrame(
        {
            "segment": ["Windows"],
            "branch": ["control"],
            "client_id": ["only"],
            "bucket_counts": [
                [{"bucket": 10, "counts": 2}, {"bucket": 20, "counts": 6}]
            ],
        }
    )
    out = compute_client_cluster_bootstrap(df, 200, "test", "metric").set_index(
        "bucket"
    )
    np.testing.assert_allclose(
        out.loc[20, "bootstrap_replicates"], 3 * out.loc[10, "bootstrap_replicates"]
    )


def test_segments_and_branches_draw_independent_weights():
    base = make_clients(np.random.default_rng(2), 50)
    df = pd.concat(
        [
            base.assign(segment="Windows", branch="control"),
            base.assign(segment="Mac", branch="control"),
            base.assign(segment="Windows", branch="treatment"),
        ]
    )
    out = compute_client_cluster_bootstrap(df, 200, "test", "metric")

    def total(segment, branch):
        rows = out[(out.segment == segment) & (out.branch == branch)]
        return np.sum(np.stack(rows.bootstrap_replicates), axis=0)

    windows = total("Windows", "control")
    assert np.corrcoef(windows, total("Mac", "control"))[0, 1] < 0.3
    assert np.corrcoef(windows, total("Windows", "treatment"))[0, 1] < 0.3


# Known answers


def test_no_effect_ci_contains_zero():
    rng = np.random.default_rng(3)
    ci = lift_ci(make_clients(rng, 2000), make_clients(rng, 2000))
    for q in (0.5, 0.75, 0.95):
        assert ci[q]["lower"] <= 0 <= ci[q]["upper"]


def test_known_slowdown_is_detected():
    rng = np.random.default_rng(4)
    ci = lift_ci(make_clients(rng, 3000), make_clients(rng, 3000, effect=1.2))
    for q in (0.5, 0.75):
        assert ci[q]["lower"] > 0
        assert ci[q]["lower"] <= 0.2 <= ci[q]["upper"]


def test_no_ci_without_cluster_replicates():
    reps = [np.ones(10), np.ones(10)]
    for control, treatment in ((None, reps), (reps, None), ([], reps)):
        ci = calc_bootstrap_ci([10, 20], control, [10, 20], treatment)
        assert all(ci[q]["lower"] is None and ci[q]["upper"] is None for q in ci)
    band = calc_cdf_band([10, 20], [10, 15, 20], None)
    assert band == {"lower": [None] * 3, "upper": [None] * 3}


# Width behaves like precision should


def test_more_clients_narrow_the_ci():
    rng = np.random.default_rng(5)
    small = width(lift_ci(make_clients(rng, 1000), make_clients(rng, 1000)))
    large = width(lift_ci(make_clients(rng, 4000), make_clients(rng, 4000)))
    # 4x the clients should roughly halve the width (1/sqrt(4)).
    assert 0.35 < large / small < 0.7


def test_more_iterations_do_not_narrow_the_ci():
    rng = np.random.default_rng(6)
    control, treatment = make_clients(rng, 2000), make_clients(rng, 2000)
    few = width(lift_ci(control, treatment, n_bootstrap=200))
    many = width(lift_ci(control, treatment, n_bootstrap=1000))
    assert 0.8 < many / few < 1.25


def test_observations_concentrated_in_fewer_clients_widen_the_ci():
    rng = np.random.default_rng(7)
    # Same 10,000 observations per branch, from 100 clients vs. 2,000 clients.
    concentrated = width(
        lift_ci(
            make_clients(rng, 100, obs_per_client=100),
            make_clients(rng, 100, obs_per_client=100),
        )
    )
    spread = width(
        lift_ci(
            make_clients(rng, 2000, obs_per_client=5),
            make_clients(rng, 2000, obs_per_client=5),
        )
    )
    assert concentrated > 2 * spread


def test_aa_coverage_is_near_nominal():
    """In simulated A/A tests, the 95% CI should contain zero ~95% of the time."""
    rng = np.random.default_rng(8)
    covered = []
    for i in range(200):
        ci = lift_ci(make_clients(rng, 300), make_clients(rng, 300), slug=f"aa{i}")
        covered.append(ci[0.5]["lower"] <= 0 <= ci[0.5]["upper"])
    assert 0.90 <= np.mean(covered) <= 0.99


# CDF band


def test_cdf_band_brackets_observed_cdf():
    df = make_clients(np.random.default_rng(9), 1000)
    out = compute_client_cluster_bootstrap(df, 500, "test", "metric").sort_values(
        "bucket"
    )
    bins, counts = list(out.bucket), list(out.counts)
    x = np.linspace(bins[0], bins[-1], 50)
    band = calc_cdf_band(bins, x, list(out.bootstrap_replicates))
    observed = np.interp(x, bins, np.cumsum(counts) / np.sum(counts))
    lower, upper = np.array(band["lower"]), np.array(band["upper"])
    assert np.all(lower <= observed + 1e-9) and np.all(observed <= upper + 1e-9)
    assert np.all((lower >= -1e-9) & (upper <= 1 + 1e-9))
    assert np.all(np.diff(lower) >= -1e-9) and np.all(np.diff(upper) >= -1e-9)


# Regressions


def test_percentile_keeps_fractional_counts():
    # Bootstrap replicate counts are fractional; truncating them to int would
    # empty this histogram entirely.
    assert calc_histogram_percentile([0, 10, 20], [0.6, 0.6, 0.8], 0.5) == (
        pytest.approx(10 + 10 * 0.4 / 0.6)
    )
