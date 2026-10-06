"""Window aggregation and core keystroke features (PRD §9)."""
import numpy as np
import pandas as pd

from . import config


def assign_windows(events: pd.DataFrame, window_s=config.WINDOW_S) -> pd.DataFrame:
    """Split every session into fixed time windows; windows never cross session boundaries."""
    events = events.copy()
    start = events.groupby(["user_id", "session_id"])["timestamp"].transform("min")
    events["window"] = ((events["timestamp"] - start).dt.total_seconds() // window_s).astype(int)
    prev = events.groupby(["user_id", "session_id"])["key_category"].shift()
    events["char_after_char"] = (events["key_category"] == "char") & (prev == "char")
    return events


def _burst_lengths(is_pause: pd.Series) -> np.ndarray:
    """Lengths of runs of keystrokes not interrupted by a pause."""
    burst_id = is_pause.to_numpy().cumsum()
    return np.bincount(burst_id)


def _entropy(categories: pd.Series) -> float:
    p = categories.value_counts(normalize=True).to_numpy()
    return float(-(p * np.log2(p)).sum())


def _window_stats(w: pd.DataFrame) -> pd.Series:
    flight = w["flight_time_ms"]
    typing_flight = flight[~w["is_pause"]]
    pauses = flight[w["is_pause"]]
    mood_share = w["mood_state"].value_counts(normalize=True)
    return pd.Series({
        "window_start": w["timestamp"].min(),
        "n_events": len(w),
        # keys per second over the time actually spent in the window (including pauses)
        "typing_speed": len(w) / (flight.sum() / 1000),
        "mean_dwell": w["dwell_time_ms"].mean(),
        "std_dwell": w["dwell_time_ms"].std(),
        "mean_flight": typing_flight.mean(),
        "std_flight": typing_flight.std(),
        "mean_pause": pauses.mean() if len(pauses) else 0.0,
        "pause_count": len(pauses),
        "long_pause_count": int((flight > config.LONG_PAUSE_MS).sum()),
        "correction_rate": w["is_correction"].mean(),
        # optional extensions (config.EXTENDED_FEATURES)
        "median_dwell": w["dwell_time_ms"].median(),
        "dwell_cv": w["dwell_time_ms"].std() / w["dwell_time_ms"].mean(),
        "flight_cv": typing_flight.std() / typing_flight.mean(),
        "char_char_latency": typing_flight[w.loc[~w["is_pause"], "char_after_char"]].mean(),
        "mean_burst_length": _burst_lengths(w["is_pause"]).mean(),
        "key_entropy": _entropy(w["key_category"]),
        # labels
        "mood_state": mood_share.idxmax(),
        "label_purity": mood_share.max(),
        "valence": w["valence"].mean(),
        "arousal": w["arousal"].mean(),
    })


def build_windows(events: pd.DataFrame, window_s=config.WINDOW_S) -> pd.DataFrame:
    events = assign_windows(events, window_s)
    windows = (events.groupby(["user_id", "session_id", "window"])
               .apply(_window_stats, include_groups=False)
               .reset_index())
    windows = windows[windows["n_events"] >= config.MIN_EVENTS_PER_WINDOW]
    windows = windows.sort_values(["user_id", "window_start"]).reset_index(drop=True)
    feats = config.CORE_FEATURES + config.EXTENDED_FEATURES
    num = feats + ["n_events", "label_purity", "valence", "arousal"]
    windows[num] = windows[num].astype(float)
    windows[feats] = windows[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return windows
