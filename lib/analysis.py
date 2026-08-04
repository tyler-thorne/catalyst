from scipy import stats, sparse
import numpy as np
import pandas as pd
import hashlib
from lib.thresholds import BOOTSTRAP_ITERATIONS, DEFAULT_BOOTSTRAP_MODE


# Expand the histogram into an array of values
def flatten_histogram(bins, counts):
    # Convert counts to int in case they're decimal.Decimal from BigQuery
    counts = [int(c) for c in counts]
    array = []
    for i in range(len(bins)):
        for j in range(1, counts[i] // 2):
            array.append(bins[i])
    return array


# effect size calculation for t-test
def calc_cohen_d(x1, x2, s1, s2, n1, n2):
    # Calculate pooled standard deviation
    pooled_variance = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_variance)

    # Handle edge cases where pooled standard deviation is very small
    if pooled_std < 1e-10:  # Essentially zero variance
        # If means are identical, effect size is 0; otherwise return a large but finite value
        if abs(x1 - x2) < 1e-10:
            return 0.0
        else:
            return np.sign(x1 - x2) * 10.0  # Large effect size, capped at 10

    effect_size = (x1 - x2) / pooled_std

    # Cap effect size to reasonable bounds to avoid overflow issues
    return np.clip(effect_size, -10.0, 10.0)


# effect size calculation for mwu
def rank_biserial_correlation(n1, n2, U):
    return 1 - 2 * U / (n1 * n2)


# Calculate two-way t-test with unequal sample size and unequal variances.
# Return the t-value, p-value, and effect size via cohen's d.
def calc_t_test(x1, x2, s1, s2, n1, n2):
    s_prime = np.sqrt((s1**2 / n1) + (s2**2) / n2)
    if s_prime == 0:
        raise ZeroDivisionError("t-test is undefined with zero variance")

    t_value = (x1 - x2) / s_prime

    df = (s1**2 / n1 + s2**2 / n2) ** 2 / (
        (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
    )
    p_value = 2 * (1 - (stats.t.cdf(abs(t_value), df)))
    effect_size = calc_cohen_d(x1, x2, s1, s2, n1, n2)
    return [t_value, p_value, effect_size]


def create_subsample(bins, counts, sample_size=100000):
    total_counts = sum(counts)
    if total_counts <= sample_size:
        return flatten_histogram(bins, counts)

    ratio = total_counts / sample_size
    subsample = []
    for i in range(len(bins)):
        subsample.extend(np.repeat(bins[i], counts[i] / ratio))
    return subsample


def calc_cdf_from_density(density, vals):
    cdf = []
    sum = 0
    for i in range(0, len(density) - 2):
        width = vals[i + 1] - vals[i]
        cdf_val = sum + density[i] * width
        sum = cdf_val
        cdf.append(cdf_val)

    width = vals[-1] - vals[-2]
    cdf_val = sum + density[-1] * width
    sum = cdf_val
    cdf.append(cdf_val)
    return cdf


# TODO: Interpolate the quantiles.
def calc_histogram_quantiles(bins, density):
    vals = []
    quantiles = []
    q = 0
    for i in range(len(bins)):
        q = q + density[i]
        vals.append(bins[i])
        quantiles.append(q)

    return [quantiles, vals]


def calc_histogram_density(counts, n):
    # Convert counts to float in case they're decimal.Decimal from BigQuery
    counts = [float(c) for c in counts]
    n = float(n)
    density = []
    cdf = []
    cum = 0
    for i in range(len(counts)):
        density.append(counts[i] / n)
        cum = cum + counts[i]
        cdf.append(cum)
    cdf = [x / cum for x in cdf]
    return [density, cdf]


def calc_histogram_mean_var(bins, counts):
    mean = 0
    n = 0
    for i in range(len(bins)):
        bucket = float(bins[i])
        count = float(counts[i])
        n = n + count
        mean = mean + bucket * count
    mean = float(mean) / float(n)

    var = 0
    for i in range(len(bins)):
        bucket = float(bins[i])
        count = float(counts[i])
        var = var + count * (bucket - mean) ** 2
    var = float(var) / float(n)
    std = np.sqrt(var)

    return [mean, var, std, n]


def calc_histogram_percentile(bins, counts, q):
    """Calculate the q-th percentile (0 < q < 1) from histogram bins and counts."""
    # float, not int: bootstrap replicate counts are fractional weighted sums,
    # and BigQuery may return decimal.Decimal
    counts = [float(c) for c in counts]
    total_count = sum(counts)
    if total_count == 0:
        return 0.0

    # Find the bin containing the q-th percentile
    target = total_count * q
    cumulative = 0

    for i, count in enumerate(counts):
        cumulative += count
        if cumulative >= target:
            # Linear interpolation within the bin
            if count == 0:
                return bins[i]

            # How far into the bin is the target?
            remaining = target - (cumulative - count)
            fraction = remaining / count

            # If we're at the last bin, just return the bin value
            if i == len(bins) - 1:
                return bins[i]

            # Interpolate between this bin and the next
            bin_width = bins[i + 1] - bins[i] if i < len(bins) - 1 else 0
            return bins[i] + fraction * bin_width

    # Fallback - should not reach here
    return bins[-1] if bins else 0.0


def calculate_histogram_stats(bins, counts, data):
    # Calculate mean, std, and var
    [mean, var, std, n] = calc_histogram_mean_var(bins, counts)
    data["mean"] = mean
    data["std"] = std
    data["var"] = var
    data["n"] = n

    # Calculate median, p75, p95
    data["median"] = calc_histogram_percentile(bins, counts, 0.5)
    data["p75"] = calc_histogram_percentile(bins, counts, 0.75)
    data["p95"] = calc_histogram_percentile(bins, counts, 0.95)

    # Calculate densities
    [density, cdf] = calc_histogram_density(counts, n)
    data["pdf"]["cdf"] = cdf
    data["pdf"]["density"] = density
    data["pdf"]["values"] = bins

    # Calculate quantiles
    [quantiles, vals] = calc_histogram_quantiles(bins, density)
    data["quantiles"] = quantiles
    data["quantile_vals"] = vals

    # Check for overflow bucket issues
    # If >90% of data is in the last bucket, the histogram is likely saturated
    if len(counts) > 0 and n > 0:
        last_bucket_ratio = float(counts[-1]) / float(n)
        if last_bucket_ratio > 0.90:
            data["overflow_warning"] = {
                "last_bucket_ratio": last_bucket_ratio,
                "last_bucket_value": bins[-1],
                "message": f"WARNING: {last_bucket_ratio*100:.1f}% of data is in the overflow bucket at {bins[-1]}. "
                f"This histogram's maximum bucket is too small for the measured values, "
                f"making statistical analysis unreliable.",
            }


def calculate_histogram_tests_subsampling(control_data, branch_data, result):
    bins_control = control_data["bins"]
    counts_control = control_data["counts"]
    control_sample = create_subsample(bins_control, counts_control)

    bins_branch = branch_data["bins"]
    counts_branch = branch_data["counts"]
    branch_sample = create_subsample(bins_branch, counts_branch)

    # Calculate t-test and effect
    x1 = np.mean(control_sample)
    s1 = np.std(control_sample)
    n1 = len(control_sample)
    x2 = np.mean(branch_sample)
    s2 = np.std(branch_sample)
    n2 = len(branch_sample)
    effect = calc_cohen_d(x1, x2, s1, s2, n1, n2)
    [t, p] = stats.ttest_ind(control_sample, branch_sample)
    result["tests"]["ttest"] = {}
    result["tests"]["ttest"]["score"] = t
    result["tests"]["ttest"]["p-value"] = p
    result["tests"]["ttest"]["effect"] = effect

    # Calculate mwu-test
    [U, p] = stats.mannwhitneyu(control_sample, branch_sample)
    r = rank_biserial_correlation(n1, n2, U)
    result["tests"]["mwu"] = {}
    result["tests"]["mwu"]["score"] = U
    result["tests"]["mwu"]["p-value"] = p
    result["tests"]["mwu"]["effect"] = r

    # Calculate ks-test
    [D, p] = stats.ks_2samp(control_sample, branch_sample)
    result["tests"]["ks"] = {}
    result["tests"]["ks"]["score"] = D
    result["tests"]["ks"]["p-value"] = p
    result["tests"]["ks"]["effect"] = D


def calculate_histogram_ttest(bins, counts, data, control):
    mean_control = control["mean"]
    std_control = control["std"]
    n_control = control["n"]

    mean = data["mean"]
    std = data["std"]
    n = data["n"]

    # Calculate t-test
    [t_value, p_value, effect] = calc_t_test(
        mean, mean_control, std, std_control, n, n_control
    )
    data["tests"]["ttest"] = {}
    data["tests"]["ttest"]["score"] = t_value
    data["tests"]["ttest"]["p-value"] = p_value
    data["tests"]["ttest"]["effect"] = effect


def calc_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return [m, se, m - h, m + h]


def _bootstrap_seed(
    slug: str, metric_name: str, branch_name: str, segment: str = None
) -> int:
    """Deterministic seed from experiment context so CI bounds don't jitter between runs.

    Pass `segment` when replicates from several segments may later be summed
    (the "All" segment), so each segment draws independent weights."""
    key = f"{slug}:{metric_name}:{branch_name}"
    if segment is not None:
        key += f":{segment}"
    key = key.encode()
    return int(hashlib.md5(key).hexdigest()[:8], 16)


def compute_client_cluster_bootstrap(
    client_df, n_bootstrap, slug, metric_name, unit_col="client_id"
):
    """Compute cluster (Bayesian) bootstrap replicates in Python from a
    per-unit nested-array pull, instead of the old in-SQL unrolled-array
    approach.

    The resampling unit is whatever `unit_col` names -- `client_id` for
    metrics-ping histograms (one row per client), or `document_id` for
    pageload-ping events (one row per ping submission, since that ping
    deliberately never carries a client identifier). The matrix math below
    doesn't know or care which: it just clusters observations under
    whatever value the caller supplies for `unit_col`, which is why the
    exact same function serves both data sources.

    `client_df` is expected to have one row per (unit, segment, branch)
    with a `bucket_counts` column of {"bucket": ..., "counts": ...} entries
    (BigQuery's ARRAY_AGG(STRUCT(bucket, counts)) output). For each
    (segment, branch) group, builds a *sparse* [units x buckets] matrix and
    a dense [units x n_bootstrap] weight matrix, then computes replicates
    via one matrix multiply.

    The matrix is sparse, not dense, because bucket cardinality varies wildly
    by source: metrics-ping histograms have ~30-40 exponential buckets
    (Glean pre-bins them), but pageload-ping events carry raw millisecond
    values as their "bucket" -- tens of thousands of distinct values across
    a real experiment. A dense [units x buckets] array at pageload-ping
    scale (millions of documents) would be tens of gigabytes and practically
    never finish; each unit only ever touches a handful of buckets, so a
    sparse matrix keeps the cost proportional to actual data volume
    (nnz * n_bootstrap) instead of units * buckets * n_bootstrap.

    Uses the same mean-1 Exponential(1) weighting the retired in-SQL
    FARM_FINGERPRINT-hash approach used -- that hash was only ever a
    determinism/reproducibility mechanism, not a statistical requirement, so
    this draws weights from a numpy RNG seeded via _bootstrap_seed instead.

    Returns a long DataFrame with columns (segment, branch, bucket, counts,
    bootstrap_replicates) -- the exact shape the in-SQL approach produced,
    so downstream code (collectResultsFromQuery_OS_segments,
    processNumericalMetrics) needs no changes.
    """
    rows = []
    for (segment, branch), group in client_df.groupby(["segment", "branch"]):
        # Canonical unit order so a fixed seed reproduces the same result.
        group = group.sort_values(unit_col).reset_index(drop=True)
        n_units = len(group)

        buckets = sorted(
            {
                entry["bucket"]
                for bucket_counts in group["bucket_counts"]
                for entry in bucket_counts
            }
        )
        bucket_index = {b: i for i, b in enumerate(buckets)}

        row_idx, col_idx, values = [], [], []
        for row_i, bucket_counts in enumerate(group["bucket_counts"]):
            for entry in bucket_counts:
                row_idx.append(row_i)
                col_idx.append(bucket_index[entry["bucket"]])
                values.append(float(entry["counts"]))
        data_matrix = sparse.csr_matrix(
            (values, (row_idx, col_idx)), shape=(n_units, len(buckets))
        )

        seed = _bootstrap_seed(slug, metric_name, branch, segment)
        rng = np.random.default_rng(seed)
        weights = rng.exponential(scale=1.0, size=(n_units, n_bootstrap))

        counts = np.asarray(data_matrix.sum(axis=0)).ravel()
        # (buckets x units) sparse @ (units x n_bootstrap) dense = dense
        # (buckets x n_bootstrap) -- keep the sparse matrix on the left so
        # scipy uses its sparse-dense matmul instead of densifying first.
        replicates = (data_matrix.T @ weights).T  # shape: (n_bootstrap, n_buckets)

        for bi, bucket in enumerate(buckets):
            rows.append(
                {
                    "segment": segment,
                    "branch": branch,
                    "bucket": bucket,
                    "counts": counts[bi],
                    "bootstrap_replicates": replicates[:, bi],
                }
            )

    return pd.DataFrame(rows)


def _replicate_histograms(replicates):
    """Transpose a list of per-bucket replicate arrays into a list of
    per-iteration histograms (each aligned to the same bucket order)."""
    matrix = np.array(replicates)  # shape: (n_buckets, n_bootstrap)
    return matrix.T  # shape: (n_bootstrap, n_buckets)


def calc_bootstrap_ci(
    control_bins,
    control_counts,
    treatment_bins,
    treatment_counts,
    quantiles=(0.5, 0.75, 0.95),
    n_bootstrap=1000,
    confidence=0.95,
    seed=None,
    control_replicates=None,
    treatment_replicates=None,
):
    """
    Paired bootstrap CIs for relative lift and absolute per-branch percentiles.

    When control_replicates/treatment_replicates are provided, each is a list
    of per-bucket arrays (one value per bootstrap iteration) computed in-SQL
    as client-weighted aggregates — i.e. a genuine client-cluster bootstrap,
    where the client (not the raw event) is the resampling unit. In that case
    we just read off each iteration's already-resampled histogram.

    Otherwise, falls back to resampling both control and treatment
    independently via multinomial draw on pooled bin probabilities — this is
    the only option for data sources (e.g. the pageload ping) that don't
    carry a client identifier to cluster on, and treats every pooled
    observation as independent.

    Returns dict keyed by quantile, each with:
        control_lower, control_upper  — absolute CI for control branch
        treatment_lower, treatment_upper — absolute CI for treatment branch
        lift_lower, lift_upper        — 95% CI on relative lift (treatment/control - 1)
    """
    ctrl_counts = np.array([float(c) for c in control_counts])
    trt_counts = np.array([float(c) for c in treatment_counts])
    ctrl_total = ctrl_counts.sum()
    trt_total = trt_counts.sum()

    if ctrl_total == 0 or trt_total == 0:
        return {
            q: {
                "control_lower": None,
                "control_upper": None,
                "treatment_lower": None,
                "treatment_upper": None,
                "lift_lower": None,
                "lift_upper": None,
            }
            for q in quantiles
        }

    boot = {q: {"ctrl": [], "trt": [], "lift": []} for q in quantiles}

    if control_replicates is not None and treatment_replicates is not None:
        ctrl_iters = _replicate_histograms(control_replicates)
        trt_iters = _replicate_histograms(treatment_replicates)
        for ctrl_resample, trt_resample in zip(ctrl_iters, trt_iters):
            for q in quantiles:
                c = calc_histogram_percentile(control_bins, ctrl_resample, q)
                t = calc_histogram_percentile(treatment_bins, trt_resample, q)
                boot[q]["ctrl"].append(c)
                boot[q]["trt"].append(t)
                boot[q]["lift"].append(t / c - 1 if c > 0 else 0.0)
    else:
        rng = np.random.default_rng(seed)
        ctrl_probs = ctrl_counts / ctrl_total
        trt_probs = trt_counts / trt_total
        ctrl_n = min(int(ctrl_total), 10000)
        trt_n = min(int(trt_total), 10000)

        for _ in range(n_bootstrap):
            ctrl_resample = rng.multinomial(ctrl_n, ctrl_probs)
            trt_resample = rng.multinomial(trt_n, trt_probs)
            for q in quantiles:
                c = calc_histogram_percentile(control_bins, ctrl_resample, q)
                t = calc_histogram_percentile(treatment_bins, trt_resample, q)
                boot[q]["ctrl"].append(c)
                boot[q]["trt"].append(t)
                boot[q]["lift"].append(t / c - 1 if c > 0 else 0.0)

    alpha = (1 - confidence) / 2
    lo, hi = alpha * 100, (1 - alpha) * 100
    return {
        q: {
            "control_lower": float(np.percentile(boot[q]["ctrl"], lo)),
            "control_upper": float(np.percentile(boot[q]["ctrl"], hi)),
            "treatment_lower": float(np.percentile(boot[q]["trt"], lo)),
            "treatment_upper": float(np.percentile(boot[q]["trt"], hi)),
            "lift_lower": float(np.percentile(boot[q]["lift"], lo)),
            "lift_upper": float(np.percentile(boot[q]["lift"], hi)),
        }
        for q in quantiles
    }


def calc_cdf_band(
    bins,
    counts,
    x_values,
    n_bootstrap=1000,
    confidence=0.95,
    seed=None,
    replicates=None,
):
    """Bootstrap confidence envelope for a CDF at specified x-values.

    When `replicates` is provided (a list of per-bucket arrays, one value per
    bootstrap iteration, computed in-SQL as client-weighted aggregates), each
    iteration's already-resampled histogram is used directly -- a client-
    cluster bootstrap. Otherwise falls back to a multinomial resample over
    the pooled histogram, for sources with no client identifier to cluster on.
    """
    counts_arr = np.array([float(c) for c in counts])
    total = counts_arr.sum()
    if total == 0:
        return {"lower": [None] * len(x_values), "upper": [None] * len(x_values)}

    bins_arr = np.array([float(b) for b in bins])
    boot_cdfs = []

    if replicates is not None:
        for resampled in _replicate_histograms(replicates):
            resampled_total = resampled.sum()
            if resampled_total == 0:
                continue
            cum = np.cumsum(resampled).astype(float) / resampled_total
            boot_cdfs.append(np.interp(x_values, bins_arr, cum))
    else:
        rng = np.random.default_rng(seed)
        probs = counts_arr / total
        n = min(int(total), 10000)
        for _ in range(n_bootstrap):
            resampled = rng.multinomial(n, probs)
            cum = np.cumsum(resampled).astype(float) / n
            boot_cdfs.append(np.interp(x_values, bins_arr, cum))

    boot_cdfs = np.array(boot_cdfs)
    alpha = (1 - confidence) / 2
    lo, hi = alpha * 100, (1 - alpha) * 100
    return {
        "lower": np.percentile(boot_cdfs, lo, axis=0).tolist(),
        "upper": np.percentile(boot_cdfs, hi, axis=0).tolist(),
    }


def createNumericalTemplate():
    ci_empty = {"lower": None, "upper": None}
    template = {
        "desc": "",
        "mean": 0,
        "confidence": {"min": 0, "max": 0},
        "se": 0,
        "var": 0,
        "std": 0,
        "n": 0,
        "pdf": {"values": [], "density": [], "cdf": []},
        "quantiles": [],
        "quantile_vals": [],
        "tests": {},
        "median_ci": dict(ci_empty),
        "p75_ci": dict(ci_empty),
        "p95_ci": dict(ci_empty),
        "median_lift_ci": dict(ci_empty),
        "p75_lift_ci": dict(ci_empty),
        "p95_lift_ci": dict(ci_empty),
    }
    return template


def createCategoricalTemplate():
    template = {"desc": "", "labels": [], "counts": [], "sample_counts": [], "sum": 0}
    return template


def createScalarTemplate():
    template = {"desc": "", "count": 0, "n": 1}
    return template


def createResultsTemplate(config):
    template = {}
    for branch in config["branches"]:
        template[branch] = {}
        for segment in config["segments"]:
            template[branch][segment] = {
                "numerical": {},
                "categorical": {},
                "scalar": {},
                "labeled_percentiles": {},
                "quantity_percentiles": {},
            }
    return template


class DataAnalyzer:
    def __init__(self, config):
        self.config = config
        self.event_controldf = None
        self.control = self.config["branches"][0]
        self.results = createResultsTemplate(config)

    def processTelemetryData(self, transformedData):
        """Process transformed telemetry data organized by data types.

        Args:
            transformedData: Data organized by types: {
                "numerical": [...],
                "categorical": [...],
                "scalar": [...],
                "labeled_percentiles": [...]
            }
        """
        print("Processing unified data types...")

        # Process each data type with unified algorithms
        self.processNumericalMetrics(transformedData["numerical"])
        self.processCategoricalMetrics(transformedData["categorical"])
        self.processScalarMetrics(transformedData["scalar"])
        self.processLabeledPercentilesMetrics(
            transformedData.get("labeled_percentiles", [])
        )
        self.processQuantityPercentilesMetrics(
            transformedData.get("quantity_percentiles", [])
        )

        return self.results

    def processNumericalMetrics(self, numerical_metrics):
        """Process all numerical metrics with unified algorithms."""
        print(f"Processing {len(numerical_metrics)} numerical metrics")

        for metric_data in numerical_metrics:
            branch = metric_data["branch"]
            segment = metric_data["segment"]
            metric_name = metric_data["metric_name"]
            source_section = metric_data["source_section"]
            config = metric_data["config"]
            data = metric_data["data"]

            print(f"  {branch}/{segment}: {source_section}.{metric_name}")

            # Store in results structure by data type
            if metric_name not in self.results[branch][segment]["numerical"]:
                self.results[branch][segment]["numerical"][
                    metric_name
                ] = createNumericalTemplate()
            result_location = self.results[branch][segment]["numerical"][metric_name]
            result_location["desc"] = config.get("desc", f"{metric_name}")

            # Apply unified numerical analysis (works for both histograms and pageload events)
            bins = data["bins"]
            counts = data["counts"]
            calculate_histogram_stats(bins, counts, result_location)
            result_location["bins"] = bins
            result_location["counts"] = counts
            if "bootstrap_replicates" in data:
                result_location["bootstrap_replicates"] = data["bootstrap_replicates"]

            # Print overflow warning if detected
            if "overflow_warning" in result_location:
                warning = result_location["overflow_warning"]
                print(
                    f"  ⚠️  {branch}/{segment}/{metric_name}: {warning['last_bucket_ratio']*100:.1f}% "
                    f"of data in overflow bucket (value={warning['last_bucket_value']})"
                )

            # Calculate statistical tests for non-control branches
            if branch != self.control:
                # Find corresponding control data
                control_data = None
                for control_metric in numerical_metrics:
                    if (
                        control_metric["branch"] == self.control
                        and control_metric["segment"] == segment
                        and control_metric["metric_name"] == metric_name
                        and control_metric["source_section"] == source_section
                    ):
                        control_data = control_metric["data"]
                        break

                if control_data:
                    calculate_histogram_tests_subsampling(
                        control_data, data, result_location
                    )

                    bootstrap_mode = self.config.get(
                        "bootstrap_mode", DEFAULT_BOOTSTRAP_MODE
                    )
                    n_bootstrap = BOOTSTRAP_ITERATIONS.get(
                        bootstrap_mode, BOOTSTRAP_ITERATIONS[DEFAULT_BOOTSTRAP_MODE]
                    )
                    result_location["bootstrap_iterations"] = n_bootstrap
                    seed = _bootstrap_seed(
                        self.config.get("slug", ""),
                        metric_name,
                        branch,
                    )
                    ci = calc_bootstrap_ci(
                        control_bins=control_data["bins"],
                        control_counts=control_data["counts"],
                        treatment_bins=data["bins"],
                        treatment_counts=data["counts"],
                        quantiles=(0.5, 0.75, 0.95),
                        n_bootstrap=n_bootstrap,
                        seed=seed,
                        control_replicates=control_data.get("bootstrap_replicates"),
                        treatment_replicates=data.get("bootstrap_replicates"),
                    )
                    result_location["median_ci"] = {
                        "lower": ci[0.5]["treatment_lower"],
                        "upper": ci[0.5]["treatment_upper"],
                    }
                    result_location["p75_ci"] = {
                        "lower": ci[0.75]["treatment_lower"],
                        "upper": ci[0.75]["treatment_upper"],
                    }
                    result_location["p95_ci"] = {
                        "lower": ci[0.95]["treatment_lower"],
                        "upper": ci[0.95]["treatment_upper"],
                    }
                    result_location["median_lift_ci"] = {
                        "lower": ci[0.5]["lift_lower"],
                        "upper": ci[0.5]["lift_upper"],
                    }
                    result_location["p75_lift_ci"] = {
                        "lower": ci[0.75]["lift_lower"],
                        "upper": ci[0.75]["lift_upper"],
                    }
                    result_location["p95_lift_ci"] = {
                        "lower": ci[0.95]["lift_lower"],
                        "upper": ci[0.95]["lift_upper"],
                    }

    def processCategoricalMetrics(self, categorical_metrics):
        """Process all categorical metrics with unified algorithms."""
        print(f"Processing {len(categorical_metrics)} categorical metrics")

        for metric_data in categorical_metrics:
            branch = metric_data["branch"]
            segment = metric_data["segment"]
            metric_name = metric_data["metric_name"]
            source_section = metric_data["source_section"]
            config = metric_data["config"]
            data = metric_data["data"]

            print(f"  {branch}/{segment}: {source_section}.{metric_name}")

            # Store in results structure by data type
            if metric_name not in self.results[branch][segment]["categorical"]:
                self.results[branch][segment]["categorical"][
                    metric_name
                ] = createCategoricalTemplate()
            result_location = self.results[branch][segment]["categorical"][metric_name]
            result_location["desc"] = config.get("desc", f"{metric_name}")

            # Apply unified categorical analysis
            labels = data["bins"]
            counts = data["counts"]
            sample_counts = data.get("sample_counts", [])
            result_location["labels"] = labels
            result_location["counts"] = counts
            result_location["sample_counts"] = sample_counts
            total = sum(counts)
            result_location["sum"] = total

            # Calculate uplift for non-control branches (percentage change in raw counts)
            if branch != self.control:
                if metric_name in self.results[self.control][segment]["categorical"]:
                    control_result = self.results[self.control][segment]["categorical"][
                        metric_name
                    ]
                    if "counts" in control_result:
                        control_counts = control_result["counts"]
                        uplift = []
                        for i in range(len(counts)):
                            if control_counts[i] > 0:
                                uplift.append(
                                    (
                                        (counts[i] - control_counts[i])
                                        / control_counts[i]
                                    )
                                    * 100
                                )
                            else:
                                uplift.append(0)
                        result_location["uplift"] = uplift

    def processScalarMetrics(self, scalar_metrics):
        """Process all scalar metrics with unified algorithms."""
        print(f"Processing {len(scalar_metrics)} scalar metrics")

        for metric_data in scalar_metrics:
            branch = metric_data["branch"]
            segment = metric_data["segment"]
            metric_name = metric_data["metric_name"]
            source_section = metric_data["source_section"]
            config = metric_data["config"]
            data = metric_data["data"]

            print(f"  {branch}/{segment}: {source_section}.{metric_name}")

            # Store in results structure by data type
            if metric_name not in self.results[branch][segment]["scalar"]:
                self.results[branch][segment]["scalar"][
                    metric_name
                ] = createScalarTemplate()
            result_location = self.results[branch][segment]["scalar"][metric_name]
            result_location["desc"] = config.get("desc", f"Total {metric_name}")

            # Apply unified scalar analysis
            # Handle both dictionary format and raw scalar values
            if isinstance(data, dict):
                scalar_value = data.get("crash_count", data.get("count", 0))
            else:
                # For crash events, data is stored as a raw integer
                scalar_value = data
            result_location["count"] = scalar_value
            result_location["n"] = 1

            # Calculate comparative statistics for non-control branches
            if branch != self.control:
                # Find corresponding control data
                if metric_name in self.results[self.control][segment]["scalar"]:
                    control_result = self.results[self.control][segment]["scalar"][
                        metric_name
                    ]
                    if "count" in control_result:
                        control_value = control_result["count"]
                        result_location["control_count"] = control_value
                        result_location["treatment_count"] = scalar_value

                        # Calculate relative change
                        if control_value > 0:
                            relative_change = (
                                (scalar_value - control_value) / control_value
                            ) * 100
                        else:
                            relative_change = scalar_value * 100

                        result_location["relative_change"] = relative_change

                        # Calculate absolute difference
                        absolute_diff = scalar_value - control_value
                        result_location["absolute_difference"] = absolute_diff

    def processLabeledPercentilesMetrics(self, labeled_percentiles_metrics):
        """Process all labeled_percentiles metrics - median, p75, p95 per label."""
        print(
            f"Processing {len(labeled_percentiles_metrics)} labeled_percentiles metrics"
        )

        for metric_data in labeled_percentiles_metrics:
            branch = metric_data["branch"]
            segment = metric_data["segment"]
            metric_name = metric_data["metric_name"]
            source_section = metric_data["source_section"]
            config = metric_data["config"]
            data = metric_data["data"]

            print(f"  {branch}/{segment}: {source_section}.{metric_name}")

            # Store in results structure by data type
            if metric_name not in self.results[branch][segment]["labeled_percentiles"]:
                self.results[branch][segment]["labeled_percentiles"][metric_name] = {
                    "desc": config.get("desc", f"{metric_name}"),
                    "labels": [],
                    "medians": [],
                    "p75s": [],
                    "p95s": [],
                    "counts": [],
                }
            result_location = self.results[branch][segment]["labeled_percentiles"][
                metric_name
            ]

            # Store the data
            result_location["labels"] = data["labels"]
            result_location["medians"] = data["medians"]
            result_location["p75s"] = data["p75s"]
            result_location["p95s"] = data["p95s"]
            result_location["counts"] = data["counts"]

            # Calculate uplift for non-control branches
            if branch != self.control:
                if (
                    metric_name
                    in self.results[self.control][segment]["labeled_percentiles"]
                ):
                    control_result = self.results[self.control][segment][
                        "labeled_percentiles"
                    ][metric_name]
                    control_labels = control_result["labels"]
                    control_medians = control_result["medians"]
                    control_p75s = control_result["p75s"]
                    control_p95s = control_result["p95s"]

                    # Create label->percentile mappings for control
                    control_median_map = dict(zip(control_labels, control_medians))
                    control_p75_map = dict(zip(control_labels, control_p75s))
                    control_p95_map = dict(zip(control_labels, control_p95s))

                    # Calculate uplift for each label and percentile
                    median_uplifts = []
                    p75_uplifts = []
                    p95_uplifts = []

                    for i, label in enumerate(data["labels"]):
                        # Median uplift
                        if (
                            label in control_median_map
                            and control_median_map[label] > 0
                        ):
                            pct_change = (
                                (data["medians"][i] - control_median_map[label])
                                / control_median_map[label]
                            ) * 100
                            median_uplifts.append(pct_change)
                        else:
                            median_uplifts.append(0.0)

                        # p75 uplift
                        if label in control_p75_map and control_p75_map[label] > 0:
                            pct_change = (
                                (data["p75s"][i] - control_p75_map[label])
                                / control_p75_map[label]
                            ) * 100
                            p75_uplifts.append(pct_change)
                        else:
                            p75_uplifts.append(0.0)

                        # p95 uplift
                        if label in control_p95_map and control_p95_map[label] > 0:
                            pct_change = (
                                (data["p95s"][i] - control_p95_map[label])
                                / control_p95_map[label]
                            ) * 100
                            p95_uplifts.append(pct_change)
                        else:
                            p95_uplifts.append(0.0)

                    result_location["median_uplifts"] = median_uplifts
                    result_location["p75_uplifts"] = p75_uplifts
                    result_location["p95_uplifts"] = p95_uplifts

    def processQuantityPercentilesMetrics(self, quantity_percentiles_metrics):
        """Process all quantity_percentiles metrics - median, p75, p95 for scalar quantities."""
        print(
            f"Processing {len(quantity_percentiles_metrics)} quantity_percentiles metrics"
        )

        for metric_data in quantity_percentiles_metrics:
            branch = metric_data["branch"]
            segment = metric_data["segment"]
            metric_name = metric_data["metric_name"]
            source_section = metric_data["source_section"]
            config = metric_data["config"]
            data = metric_data["data"]

            print(f"  {branch}/{segment}: {source_section}.{metric_name}")

            # Store in results structure
            if metric_name not in self.results[branch][segment]["quantity_percentiles"]:
                self.results[branch][segment]["quantity_percentiles"][metric_name] = {
                    "desc": config.get("desc", f"{metric_name}"),
                    "median": 0,
                    "p75": 0,
                    "p95": 0,
                    "sum": 0,
                    "count": 0,
                }
            result_location = self.results[branch][segment]["quantity_percentiles"][
                metric_name
            ]

            # Store the data
            result_location["median"] = data["median"]
            result_location["p75"] = data["p75"]
            result_location["p95"] = data["p95"]
            result_location["sum"] = data["sum"]
            result_location["count"] = data["count"]

            # Calculate uplift for non-control branches
            if branch != self.control:
                if (
                    metric_name
                    in self.results[self.control][segment]["quantity_percentiles"]
                ):
                    control_result = self.results[self.control][segment][
                        "quantity_percentiles"
                    ][metric_name]
                    control_median = control_result["median"]
                    control_p75 = control_result["p75"]
                    control_p95 = control_result["p95"]
                    control_sum = control_result["sum"]

                    # Calculate uplifts for each percentile
                    if control_median > 0:
                        result_location["median_uplift"] = (
                            (data["median"] - control_median) / control_median
                        ) * 100
                    if control_p75 > 0:
                        result_location["p75_uplift"] = (
                            (data["p75"] - control_p75) / control_p75
                        ) * 100
                    if control_p95 > 0:
                        result_location["p95_uplift"] = (
                            (data["p95"] - control_p95) / control_p95
                        ) * 100
                    if control_sum > 0:
                        result_location["sum_uplift"] = (
                            (data["sum"] - control_sum) / control_sum
                        ) * 100
