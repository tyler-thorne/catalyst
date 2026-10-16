"""enable_bootstrap_ci gates bootstrap CIs, the sample-size floor, and summary colors."""

import pandas as pd
import pytest

from lib.analysis import DataAnalyzer
from lib.generate import resultsMatchBootstrapFlag, transformTelemetryDataByType
from lib.report import get_ci_significance, get_summary_color
from lib.telemetry import TelemetryClient


@pytest.mark.parametrize("enabled", [False, True, None])
def test_bootstrap_ci_follows_flag(enabled):
    config = {
        "slug": "flag-test",
        "branches": ["control", "treatment"],
        "segments": ["Windows"],
        "histograms": {"latency": {"kind": "numerical"}},
    }
    if enabled is not None:
        config["enable_bootstrap_ci"] = enabled
    telemetry = {
        branch: {
            "Windows": {
                "histograms": {
                    "latency": {
                        "bins": [10, 20, 40, 80],
                        "counts": [4000, 3000, 2000, 1000],
                    }
                }
            }
        }
        for branch in config["branches"]
    }

    transformed = transformTelemetryDataByType(telemetry, config)
    results = DataAnalyzer(config).processTelemetryData(transformed)

    latency = results["treatment"]["Windows"]["numerical"]["latency"]
    assert "mwu" in latency["tests"]  # significance tests run either way
    for key in ("median_lift_ci", "p75_lift_ci", "p95_lift_ci"):
        assert (latency[key]["lower"] is not None) == bool(enabled)


@pytest.mark.parametrize("enabled", [False, True])
def test_sample_size_floor_follows_flag(enabled, tmp_path):
    config = {
        "branches": [{"name": "control"}, {"name": "treatment"}],
        "segments": ["Windows"],
        "histograms": {
            "latency": {"distribution_type": "timing_distribution", "kind": "numerical"}
        },
        "enable_bootstrap_ci": enabled,
    }
    df = pd.DataFrame(
        {
            "segment": ["Windows"] * 2,
            "branch": ["control", "treatment"],
            "counts": [10, 10],
        }
    )
    client = TelemetryClient(str(tmp_path), config)
    assert bool(client._insufficientSampleSegments(df, "latency")) == enabled


def test_cache_files_differ_by_flag(tmp_path):
    def client(enabled):
        config = {
            "slug": "s",
            "histograms": {
                "metrics.timing_distribution.latency": {
                    "kind": "numerical",
                    "glean": True,
                },
                "metrics.labeled_counter.process": {
                    "kind": "categorical",
                    "glean": True,
                },
            },
            "enable_bootstrap_ci": enabled,
        }
        return TelemetryClient(str(tmp_path), config)

    off, on = client(False), client(True)
    numerical = "metrics.timing_distribution.latency"
    categorical = "metrics.labeled_counter.process"
    assert off._histogramCacheFile(numerical) != on._histogramCacheFile(numerical)
    assert off._histogramCacheFile(categorical) == on._histogramCacheFile(categorical)
    assert off._pageloadEventCacheFile("lcp_time") != on._pageloadEventCacheFile(
        "lcp_time"
    )


def test_saved_results_require_matching_flag():
    saved = {"input": {"enable_bootstrap_ci": True}}
    assert resultsMatchBootstrapFlag(saved, {"enable_bootstrap_ci": True})
    assert not resultsMatchBootstrapFlag(saved, {"enable_bootstrap_ci": False})
    assert not resultsMatchBootstrapFlag(saved, {})
    assert resultsMatchBootstrapFlag({"input": {}}, {})


NORMAL = "font-weight: normal"
GREEN = "font-weight: bold; color: green"
RED = "font-weight: bold; color: red"


@pytest.mark.parametrize(
    "uplift_pct,lift_ci,higher_is_better,expected",
    [
        # CI excludes zero and lift clears 1.5%: colored by direction.
        (-3.0, {"lower": -0.05, "upper": -0.01}, False, GREEN),
        (3.0, {"lower": 0.01, "upper": 0.05}, False, RED),
        (3.0, {"lower": 0.01, "upper": 0.05}, True, GREEN),
        # CI spans zero: not significant, even with a large lift.
        (5.0, {"lower": -0.01, "upper": 0.11}, False, NORMAL),
        # Significant but below the 1.5% practical threshold.
        (1.0, {"lower": 0.002, "upper": 0.018}, False, NORMAL),
        # No CI available: nothing to base significance on.
        (5.0, None, False, NORMAL),
        (5.0, {"lower": None, "upper": None}, False, NORMAL),
    ],
)
def test_summary_color_uses_lift_ci_when_flag_on(
    uplift_pct, lift_ci, higher_is_better, expected
):
    # effect_meaning is the MWU verdict; flag on, it must not affect color.
    for effect_meaning in ("None", "Large"):
        assert (
            get_summary_color(
                uplift_pct, lift_ci, effect_meaning, higher_is_better, True
            )
            == expected
        )


@pytest.mark.parametrize(
    "uplift_pct,effect_meaning,expected",
    [
        (3.0, "Large", RED),
        (-3.0, "Medium", GREEN),
        (3.0, "Small", NORMAL),
        (3.0, "None", NORMAL),
        (1.0, "Large", NORMAL),
    ],
)
def test_summary_color_uses_mwu_when_flag_off(uplift_pct, effect_meaning, expected):
    # Flag off ignores the lift CI, matching main.
    ci = {"lower": -0.01, "upper": 0.05}
    assert get_summary_color(uplift_pct, ci, effect_meaning, False, False) == expected


def test_ci_significance_lists_percentiles_excluding_zero():
    excludes = {"lower": 0.01, "upper": 0.05}
    spans = {"lower": -0.01, "upper": 0.05}
    assert (
        get_ci_significance(
            {"median_lift_ci": excludes, "p75_lift_ci": spans, "p95_lift_ci": excludes}
        )
        == "median, p95"
    )
    assert get_ci_significance({"median_lift_ci": spans}) == "None"
    assert get_ci_significance({"median_lift_ci": {"lower": None}}) == "No CI"
    assert get_ci_significance({}) == "No CI"
