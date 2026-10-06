"""Data inspection and cleaning (PRD §7).

Every rule here is backed by a check in `audit()` and in notebooks/01_data_cleaning.ipynb.
The cleaned output never contains the typed character: `key_event` is reduced to an
abstract category before anything is written to disk (R02).
"""
import numpy as np
import pandas as pd

from . import config

SPECIAL_KEYS = {"space", "backspace", "enter", "shift"}


def load_raw(path=config.RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def key_category(key: pd.Series) -> pd.Series:
    """Map any concrete key to an abstract category so no typed content survives."""
    k = key.astype(str).str.lower()
    return k.where(k.isin(SPECIAL_KEYS), "char")


def audit(df: pd.DataFrame) -> dict:
    """Data-quality report for the raw file (PRD §7 tasks 1-9)."""
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    sess = df.assign(ts=ts).groupby(["user_id", "session_id"])
    labels_per_session = sess["mood_state"].nunique()
    run_id = (df["mood_state"] != sess["mood_state"].shift()).groupby([df["user_id"], df["session_id"]]).cumsum()
    run_len = df.groupby([df["user_id"], df["session_id"], run_id]).size()
    return {
        "rows": len(df),
        "users": df["user_id"].nunique(),
        "sessions_per_user": sess.ngroups / df["user_id"].nunique(),
        "events_per_session": df.groupby(["user_id", "session_id"]).size().describe().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing": df.isna().sum()[lambda s: s > 0].to_dict(),
        "negative_dwell": int((df["dwell_time_ms"] < 0).sum()),
        "negative_flight": int((df["flight_time_ms"] < 0).sum()),
        "dwell_over_max": int((df["dwell_time_ms"] > config.MAX_DWELL_MS).sum()),
        "flight_over_long_pause": int((df["flight_time_ms"] > config.LONG_PAUSE_MS).sum()),
        "unparseable_timestamps": int(ts.isna().sum()),
        "sessions_not_time_ordered": int((~sess["ts"].apply(lambda s: s.is_monotonic_increasing)).sum()),
        "session_duration_s": (sess["ts"].max() - sess["ts"].min()).dt.total_seconds().describe().to_dict(),
        "raw_key_values": int(df["key_event"].nunique()),
        "raw_keys_are_characters": bool((~df["key_event"].str.lower().isin(SPECIAL_KEYS | {"char"})).any()),
        "backspace_not_flagged_correction": int(((df["key_event"] == "backspace") & (df["is_correction"] == 0)).sum()),
        "correction_not_backspace": int(((df["key_event"] != "backspace") & (df["is_correction"] == 1)).sum()),
        "mood_distribution": df["mood_state"].value_counts(normalize=True).round(4).to_dict(),
        "moods_per_session_mean": float(labels_per_session.mean()),
        "mood_run_length_mean": float(run_len.mean()),
    }


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return the cleaned event table and a human-readable log of every change."""
    log = []
    df = df.copy()

    n = len(df)
    df = df.drop_duplicates()
    log.append(f"Removed {n - len(df)} exact duplicate rows.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    bad_ts = df["timestamp"].isna().sum()
    df = df.dropna(subset=["timestamp"])
    log.append(f"Dropped {bad_ts} rows with unparseable timestamps.")

    df = df.sort_values(["user_id", "session_id", "timestamp"]).reset_index(drop=True)
    df["event_in_session"] = df.groupby(["user_id", "session_id"]).cumcount() + 1
    resets = (df["event_index"] == df["event_in_session"]).all()
    log.append("Sorted by user/session/time and re-indexed events per session"
               + ("." if resets else " (raw event_index does not restart at each session)."))

    # Privacy: replace the concrete key with a category, then drop the original column.
    df["key_category"] = key_category(df["key_event"])
    df = df.drop(columns=["key_event", "event_index"])
    log.append("Replaced key_event with key_category {char, space, backspace, enter, shift}; "
               "the raw characters are not carried forward.")

    # Every correction in the raw data is a backspace, but some backspaces are not flagged.
    unflagged = int(((df["key_category"] == "backspace") & (df["is_correction"] == 0)).sum())
    df["is_correction"] = (df["key_category"] == "backspace").astype(int)
    log.append(f"Set is_correction=1 for {unflagged} backspace events that were not flagged.")

    invalid_dwell = (df["dwell_time_ms"] < 0) | (df["dwell_time_ms"] > config.MAX_DWELL_MS)
    df.loc[invalid_dwell, "dwell_time_ms"] = np.nan
    invalid_flight = df["flight_time_ms"] < 0
    df.loc[invalid_flight, "flight_time_ms"] = np.nan
    log.append(f"Marked {int(invalid_dwell.sum())} invalid dwell values (<0 or >{config.MAX_DWELL_MS} ms) "
               f"and {int(invalid_flight.sum())} negative flight values as missing.")

    # flight_time_ms equals the gap between consecutive timestamps, so most gaps can be recovered exactly.
    df["flight_imputed"] = df["flight_time_ms"].isna()
    gap_ms = df.groupby(["user_id", "session_id"])["timestamp"].diff().dt.total_seconds() * 1000
    recovered = df["flight_time_ms"].isna() & gap_ms.notna()
    df.loc[recovered, "flight_time_ms"] = gap_ms[recovered]
    log.append(f"Recovered {int(recovered.sum())} missing flight values from timestamp differences.")

    df["dwell_imputed"] = df["dwell_time_ms"].isna()
    for col in ["dwell_time_ms", "flight_time_ms"]:
        missing = df[col].isna().sum()
        df[col] = df[col].fillna(df.groupby("user_id")[col].transform("median"))
        log.append(f"Imputed {missing} remaining missing {col} values with the user's median.")

    df["is_pause"] = df["flight_time_ms"] > config.PAUSE_MS
    log.append(f"Kept long flights as pauses (not outliers): {int(df['is_pause'].sum())} events "
               f"> {config.PAUSE_MS} ms form a separate mode in the flight distribution.")

    return df, log


def run(save=True):
    raw = load_raw()
    report = audit(raw)
    cleaned, log = clean(raw)
    if save:
        config.CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(config.CLEAN_PATH, index=False)
    return cleaned, report, log


if __name__ == "__main__":
    _, report, log = run()
    for k, v in report.items():
        print(f"{k}: {v}")
    print("\n".join(log))
