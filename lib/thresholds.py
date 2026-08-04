# Minimum sample size (observations per branch/segment) required before results
# are considered usable. Values are based on the statistical properties of each
# distribution type — heavier tails and smaller expected effect sizes require
# more data for stable percentile estimates.
#
# Review cadence: quarterly, or when a new distribution type is added.
# Last reviewed: 2026-Q2

# Bootstrap iteration counts by mode. Configurable via `bootstrap_mode` in
# the experiment YAML config. Default is "production".
#
# fast:           quick iteration / debugging
# production:     default; stable CIs for most metrics
# high_precision: tighter CIs for publication-quality results
BOOTSTRAP_ITERATIONS = {
    "fast": 200,
    "production": 1000,
    "high_precision": 5000,
}
DEFAULT_BOOTSTRAP_MODE = "production"

MIN_SAMPLE_SIZE = {
    "timing_distribution": 50000,
    "memory_distribution": 25000,
    "labeled_percentiles": 10000,  # per label
    "quantity_percentiles": 10000,
    "categorical": 5000,
    "scalar": 1000,
    "default": 1000,
}


def get_min_sample_size(distribution_type: str, kind: str) -> int:
    if kind in ("scalar",):
        return MIN_SAMPLE_SIZE["scalar"]
    if kind == "categorical":
        return MIN_SAMPLE_SIZE["categorical"]
    if kind == "labeled_percentiles":
        return MIN_SAMPLE_SIZE["labeled_percentiles"]
    if kind == "quantity_percentiles":
        return MIN_SAMPLE_SIZE["quantity_percentiles"]
    if distribution_type in MIN_SAMPLE_SIZE:
        return MIN_SAMPLE_SIZE[distribution_type]
    return MIN_SAMPLE_SIZE["default"]
