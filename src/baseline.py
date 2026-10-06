"""Personal baseline deviations (PRD §10) and temporal context (PRD §11).

Baselines are leave-one-out: each window is compared against the mean/std of the same
user's *other* windows. No labels are used, so a test user's baseline needs only their
unlabelled typing, which is what a deployed system would have after a calibration period.
"""
import numpy as np
import pandas as pd

from . import config


def z_col(f):
    return f"{f}_z"


def change_col(f):
    return f"{f}_change"


def add_baseline_deviation(windows: pd.DataFrame, features=config.CORE_FEATURES) -> pd.DataFrame:
    windows = windows.copy()
    g = windows.groupby("user_id")
    n = g[features[0]].transform("size")
    for f in features:
        x = windows[f]
        s1 = g[f].transform("sum") - x
        s2 = g[f].transform(lambda v: (v ** 2).sum()) - x ** 2
        mean = s1 / (n - 1)
        var = (s2 - (n - 1) * mean ** 2) / (n - 2)
        std = np.sqrt(var.clip(lower=0))
        windows[z_col(f)] = ((x - mean) / std.replace(0, np.nan)).fillna(0.0)
    return windows


def add_temporal(windows: pd.DataFrame, features=config.CORE_FEATURES, k=config.TREND_WINDOWS) -> pd.DataFrame:
    """Current value minus the mean of the user's previous k windows (0 when there is no history)."""
    windows = windows.sort_values(["user_id", "window_start"]).copy()
    g = windows.groupby("user_id")
    for f in features:
        prev = g[f].transform(lambda v: v.shift(1).rolling(k, min_periods=1).mean())
        windows[change_col(f)] = (windows[f] - prev).fillna(0.0)
    return windows.reset_index(drop=True)


def user_baseline(windows: pd.DataFrame, user_id, features=config.CORE_FEATURES) -> pd.DataFrame:
    """Full-history baseline for display in the dashboard."""
    u = windows.loc[windows["user_id"] == user_id, features]
    return pd.DataFrame({"baseline_mean": u.mean(), "baseline_std": u.std()})


def trend_label(change: float, user_std: float, threshold=0.5) -> str:
    if user_std == 0 or np.isnan(user_std):
        return "stable"
    r = change / user_std
    return "increasing" if r > threshold else "decreasing" if r < -threshold else "stable"
