import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_preprocessing import config, utils
from novel_preprocessing import n03_observation_process

TAU_MIN, TAU_MAX = 1, 72
TAU_MAX_PATIENTS = 5000
MIN_PAIRS = 30

# every fit call is recorded here, so the leakage test can check what each fit saw
FIT_LOG = []
_SPLITS = None


def value_variables(measured=None):
    measured = config.MEASURED if measured is None else measured
    return [v for v in measured if v not in n03_observation_process.NO_VALUE]


def p2_columns(measured=None):
    """Columns turned into within-hospital percentiles: P1 last values, hours since, rates."""
    measured = config.MEASURED if measured is None else measured
    cols = [f"{v}_last" for v in value_variables(measured)]
    cols += [f"{v}_hsince" for v in measured]
    cols += [f"{v}_rate" for v in measured]
    return cols


def load_splits():
    global _SPLITS
    if _SPLITS is None:
        _SPLITS = utils.load(config.SPLITS_PARQUET)
    return _SPLITS


def train_ids(key):
    strategy, seed, fold = key
    s = load_splits()
    s = s[(s["strategy"] == strategy) & (s["seed"] == seed) & (s["fold"] == fold)]
    return set(s.loc[s["role"] == "train", "patient_id"])


def assert_train_only(patient_ids, key, what):
    ids = set(pd.unique(np.asarray(patient_ids)))
    allowed = train_ids(key)
    outside = ids - allowed
    FIT_LOG.append({"fit": what, "key": key, "n_patients": len(ids), "n_outside": len(outside)})
    assert not outside, f"{what} got {len(outside)} patients that are not training patients of {key}"


# ---- BASE: median imputation and scaling ----

def fit_medians(X, patient_ids, key):
    """Column medians of the training rows (X is a float array)."""
    assert_train_only(patient_ids, key, "medians")
    with warnings.catch_warnings():
        # an all-NaN column gives a warning and a NaN median, which becomes 0 below
        warnings.simplefilter("ignore", RuntimeWarning)
        # column by column: nanmedian on the whole array would copy it once more
        medians = np.array([np.nanmedian(X[:, j]) for j in range(X.shape[1])])
    return np.nan_to_num(medians, nan=0.0)


def apply_medians(X, medians):
    """Fill NaN with the medians, in place (X may be a column slice of a bigger array)."""
    for j in range(X.shape[1]):
        column = X[:, j]
        np.copyto(column, medians[j], where=np.isnan(column))
    return X


def fit_scaler(X, patient_ids, key):
    assert_train_only(patient_ids, key, "scaler")
    scaler = StandardScaler()
    # row chunks: a single fit makes a float64 copy of the whole block
    for start in range(0, len(X), 100_000):
        scaler.partial_fit(X[start:start + 100_000])
    return scaler


def apply_scaler(X, scaler):
    """Standardise in place with the fitted mean and scale."""
    X -= scaler.mean_.astype(X.dtype)
    X /= scaler.scale_.astype(X.dtype)
    return X


# ---- P6: time-decay imputation of the P1 last values ----

def autocorrelation_curve(values, patient_codes, hours, rng):
    """Pooled within-patient autocorrelation and number of pairs for lags 1..TAU_MAX."""
    codes = np.unique(patient_codes)
    if len(codes) > TAU_MAX_PATIENTS:
        codes = rng.choice(codes, TAU_MAX_PATIENTS, replace=False)
    keep = np.isin(patient_codes, codes)
    rows = np.searchsorted(np.sort(codes), patient_codes[keep])
    grid = np.full((len(codes), int(hours.max()) + 1), np.nan)
    grid[rows, hours[keep].astype(int)] = values[keep]
    lags, corr, pairs = [], [], []
    for lag in range(1, TAU_MAX + 1):
        if lag >= grid.shape[1]:
            break
        a, b = grid[:, :-lag], grid[:, lag:]
        both = ~np.isnan(a) & ~np.isnan(b)
        if both.sum() < MIN_PAIRS:
            continue
        x, y = a[both], b[both]
        if x.std() == 0 or y.std() == 0:
            continue
        lags.append(lag)
        corr.append(np.corrcoef(x, y)[0, 1])
        pairs.append(both.sum())
    return np.array(lags), np.array(corr), np.array(pairs)


def tau_first_crossing(lags, corr):
    """v1: first lag at which the autocorrelation drops below 1/e."""
    below = np.nonzero(corr < 1 / np.e)[0]
    return lags[below[0]] if len(below) else TAU_MAX


def tau_exponential_fit(lags, corr, pairs):
    """v2: tau of exp(-lag / tau) fitted to the whole curve.

    Weighted least squares of log(corr) on lag through the origin, weights = number
    of pairs; lags with corr <= 0 are left out. One early dip no longer decides tau.
    """
    keep = corr > 0
    if keep.sum() < 2:
        return TAU_MIN
    k, r, w = lags[keep], corr[keep], pairs[keep]
    slope = (w * k * np.log(r)).sum() / (w * k * k).sum()
    return TAU_MAX if slope >= 0 else -1 / slope


def autocorrelation_tau(values, patient_codes, hours, rng, method="first"):
    if len(values) == 0:
        # never measured in the training patients (e.g. iCalcium when training on A)
        return TAU_MAX
    lags, corr, pairs = autocorrelation_curve(values, patient_codes, hours, rng)
    if len(lags) == 0:
        return TAU_MAX
    if method == "exp":
        return tau_exponential_fit(lags, corr, pairs)
    return tau_first_crossing(lags, corr)


def fit_decay(p1, patient_ids, hours, key, measured=None, tau_method="first"):
    """tau_v and the training median of the observed values, per variable."""
    assert_train_only(patient_ids, key, "decay")
    rng = np.random.default_rng(config.RANDOM_STATE)
    codes = pd.factorize(np.asarray(patient_ids))[0]
    hours = np.asarray(hours)
    tau, median = {}, {}
    for v in value_variables(measured):
        observed = (p1[f"{v}_hsince"].to_numpy() == 0) & (p1[f"{v}_ever"].to_numpy() == 1)
        values = p1[f"{v}_last"].to_numpy(dtype="float64")
        median[v] = float(np.median(values[observed])) if observed.any() else 0.0
        tau[v] = autocorrelation_tau(values[observed], codes[observed], hours[observed], rng, tau_method)
    tau = pd.Series(tau).clip(TAU_MIN, TAU_MAX)
    return {"tau": tau, "median": pd.Series(median)}


def apply_decay(p1, decay):
    """last value pulled towards the training median as it gets older; median if never seen."""
    out = {}
    for v in decay["tau"].index:
        last = p1[f"{v}_last"].to_numpy(dtype="float64")
        weight = np.exp(-p1[f"{v}_hsince"].to_numpy(dtype="float64") / decay["tau"][v])
        value = weight * last + (1 - weight) * decay["median"][v]
        never = p1[f"{v}_ever"].to_numpy() == 0
        out[f"{v}_last"] = np.where(never | np.isnan(last), decay["median"][v], value)
    return pd.DataFrame(out, index=p1.index).astype("float32")


# ---- P2: hospital-relative percentile encoding ----

def _reference(X, rng):
    if len(X) > config.P2_REFERENCE_ROWS:
        X = X.iloc[rng.choice(len(X), config.P2_REFERENCE_ROWS, replace=False)]
    ref = {}
    for col in X.columns:
        values = X[col].to_numpy(dtype="float64")
        ref[col] = np.sort(values[~np.isnan(values)])
    return ref


def fit_p2(X, patient_ids, hospitals, key):
    """Sorted reference sample per hospital, from training rows of that hospital."""
    assert_train_only(patient_ids, key, "p2")
    rng = np.random.default_rng(config.RANDOM_STATE)
    hospitals = np.asarray(hospitals)
    return {h: _reference(X[hospitals == h], rng) for h in np.unique(hospitals)}


def fit_p2_unlabeled(X, hospital):
    """Reference for a new hospital from its own feature rows (transductive P2).

    Only the feature values of the target hospital are used, never its labels, so
    this is unsupervised domain adaptation: like recalibrating to a new site from
    its routine data before any outcomes are known.
    """
    rng = np.random.default_rng(config.RANDOM_STATE)
    return {hospital: _reference(X, rng)}


def row_uniform(row_keys, column):
    """Deterministic uniform [0, 1) number per (row, column), from a hash of the row key.

    The draw depends only on the row itself, never on other rows, so it stays causal
    and gives the same value in every split.
    """
    with np.errstate(over="ignore"):
        x = np.asarray(row_keys, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(column + 1)
        # splitmix64 finaliser
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype("float64") / float(1 << 53)


def row_keys(patient_ids, hours):
    """Numeric key per row: patient number and ICULOS (p000123 at hour 7 -> 123007)."""
    numbers = pd.Series(np.asarray(patient_ids)).str.slice(1).astype("int64").to_numpy()
    return numbers * 1000 + np.asarray(hours, dtype="int64")


def apply_p2(X, hospitals, refs, keys):
    """Percentile in [0, 1] of every value within its row's hospital reference.

    A value tied with reference values gets a random position inside the tie, drawn
    from row_uniform. With mid-ranks every tied value (hours since = 0, an integer
    heart rate) became one number per hospital, so the encoded value told the model
    which hospital the row came from.
    """
    hospitals = np.asarray(hospitals)
    keys = np.asarray(keys)
    out = np.full(X.shape, np.nan, dtype="float32")
    for h in np.unique(hospitals):
        rows = hospitals == h
        for j, col in enumerate(X.columns):
            ref = refs[h][col]
            values = X[col].to_numpy(dtype="float64")[rows]
            if len(ref) == 0:
                continue
            low = np.searchsorted(ref, values, "left")
            high = np.searchsorted(ref, values, "right")
            pct = (low + row_uniform(keys[rows], j) * (high - low)) / len(ref)
            out[rows, j] = np.where(np.isnan(values), np.nan, pct)
    return pd.DataFrame(out, index=X.index, columns=X.columns)


def run(force=False):
    """Fit P6 on the S1 seed-42 training patients and show the tau values."""
    from ablation import a01_make_splits
    a01_make_splits.run()
    labels = utils.load(config.LABELS_PARQUET)
    p1 = utils.load(config.P1_PARQUET)
    key = ("S1", config.SPLIT_SEEDS[0], 0)
    train = labels["patient_id"].isin(train_ids(key)).to_numpy()
    decay = fit_decay(p1[train], labels.loc[train, "patient_id"], labels.loc[train, "ICULOS"], key)
    table = pd.DataFrame({"tau_hours": decay["tau"], "train_median": decay["median"]})
    table.to_csv(config.TAU_CSV)
    print("time-decay constants tau_v (hours) fitted on S1 seed 42 training patients:")
    utils.show(table.round(3))
    return table


if __name__ == "__main__":
    run()
