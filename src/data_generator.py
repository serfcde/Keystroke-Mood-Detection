"""Synthetic keystroke generator (PRD §5).

    python -m src.data_generator            # writes data/raw/generated_keystroke_raw_100k.csv

Design (all assumptions, not human ground truth):
- 50 users x 20 sessions x 100 events. Every user has their own typing baseline.
- One mood per session. A user's mood persists between consecutive sessions with
  probability STAY_PROB (a Markov chain), which gives the temporal features something to find.
- Moods scale the user's baseline (MOOD_EFFECTS) with a per-session intensity, so moods overlap
  and are not trivially separable.
- key_event holds only an abstract category; no characters are ever generated.
- Data-quality problems are injected on purpose so the cleaning stage has real work to do.
"""
import numpy as np
import pandas as pd

from . import config

N_USERS, N_SESSIONS, N_EVENTS = 50, 20, 100
STAY_PROB = 0.6
MOOD_PRIOR = {"calm": 0.30, "positive": 0.25, "stressed": 0.25, "low": 0.20}

# multipliers on the user's baseline: dwell, flight, timing variability, pause rate, correction rate
MOOD_EFFECTS = {
    "calm":     dict(dwell=1.00, flight=1.00, var=1.00, pause=1.0, corr=1.0),
    "positive": dict(dwell=0.97, flight=0.92, var=0.95, pause=0.8, corr=0.9),
    "stressed": dict(dwell=0.93, flight=0.86, var=1.35, pause=1.4, corr=1.8),
    "low":      dict(dwell=1.10, flight=1.18, var=1.10, pause=1.8, corr=1.2),
}
# centre of each mood in valence/arousal space
VA_CENTRE = {"calm": (0.4, -0.4), "positive": (0.6, 0.5), "stressed": (-0.5, 0.6), "low": (-0.5, -0.5)}

KEY_PROBS = {"char": 0.80, "space": 0.16, "shift": 0.03, "enter": 0.01}

# injected data-quality problems (fractions of events)
P_MISSING_DWELL, P_MISSING_FLIGHT = 0.0015, 0.0015
P_NEGATIVE_DWELL, P_STUCK_KEY = 0.001, 0.0003
P_OUTLIER_PAUSE = 0.001
N_DUPLICATES = 50


def _user_params(rng):
    return dict(
        dwell=rng.normal(105, 14),
        flight=rng.normal(135, 25),
        dwell_cv=rng.uniform(0.15, 0.30),
        flight_cv=rng.uniform(0.25, 0.45),
        pause_rate=rng.uniform(0.02, 0.05),
        corr_rate=rng.uniform(0.02, 0.06),
        start=pd.Timestamp("2026-01-05 08:00") + pd.Timedelta(minutes=int(rng.integers(0, 600))),
    )


def _mood_sequence(rng, n):
    moods, p = list(MOOD_PRIOR), list(MOOD_PRIOR.values())
    seq = [rng.choice(moods, p=p)]
    for _ in range(n - 1):
        seq.append(seq[-1] if rng.random() < STAY_PROB else rng.choice(moods, p=p))
    return seq


def _session(rng, user_id, session_id, u, mood, t0):
    e = MOOD_EFFECTS[mood]
    k = rng.uniform(0.5, 1.2)           # how strongly this session expresses the mood
    m = {key: 1 + k * (val - 1) for key, val in e.items()}

    dwell = rng.normal(u["dwell"] * m["dwell"], u["dwell"] * u["dwell_cv"] * m["var"], N_EVENTS).clip(25)
    mu_f = u["flight"] * m["flight"]
    sigma = np.sqrt(np.log(1 + (u["flight_cv"] * m["var"]) ** 2))
    flight = rng.lognormal(np.log(mu_f) - sigma ** 2 / 2, sigma, N_EVENTS)
    pause = rng.random(N_EVENTS) < u["pause_rate"] * m["pause"]
    flight[pause] = rng.uniform(600, 3000, pause.sum())

    keys = rng.choice(list(KEY_PROBS), size=N_EVENTS, p=list(KEY_PROBS.values()))
    corr = rng.random(N_EVENTS) < u["corr_rate"] * m["corr"]
    keys[corr] = "backspace"

    v0, a0 = VA_CENTRE[mood]
    v_s, a_s = rng.normal(v0, 0.15), rng.normal(a0, 0.15)
    ts = t0 + pd.to_timedelta(np.cumsum(flight), unit="ms")
    return pd.DataFrame({
        "user_id": user_id, "session_id": session_id, "event_index": np.arange(1, N_EVENTS + 1),
        "timestamp": ts, "key_event": keys, "dwell_time_ms": dwell, "flight_time_ms": flight,
        "is_correction": corr.astype(int),
        "valence": np.clip(v_s + rng.normal(0, 0.05, N_EVENTS), -1, 1),
        "arousal": np.clip(a_s + rng.normal(0, 0.05, N_EVENTS), -1, 1),
        "mood_state": mood,
    })


def _inject_problems(df, rng):
    n = len(df)
    pick = lambda p: rng.random(n) < p  # noqa: E731
    df.loc[pick(P_MISSING_DWELL), "dwell_time_ms"] = np.nan
    df.loc[pick(P_MISSING_FLIGHT), "flight_time_ms"] = np.nan
    neg = pick(P_NEGATIVE_DWELL)
    df.loc[neg, "dwell_time_ms"] = -df.loc[neg, "dwell_time_ms"]
    df.loc[pick(P_STUCK_KEY), "dwell_time_ms"] = rng.uniform(1500, 5000)
    # very long gaps: shift the rest of the session so timestamps stay consistent with flight
    for i in np.flatnonzero(pick(P_OUTLIER_PAUSE)):
        extra = rng.uniform(5000, 15000)
        same = (df["user_id"] == df.at[i, "user_id"]) & (df["session_id"] == df.at[i, "session_id"])
        later = same & (df.index >= i)
        df.loc[later, "timestamp"] += pd.Timedelta(milliseconds=extra)
        if pd.notna(df.at[i, "flight_time_ms"]):
            df.at[i, "flight_time_ms"] += extra
    dups = df.sample(N_DUPLICATES, random_state=int(rng.integers(1 << 31)))
    return pd.concat([df, dups]).sort_values(["user_id", "session_id", "timestamp"], kind="stable").reset_index(drop=True)


def generate(seed=config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for ui in range(1, N_USERS + 1):
        u = _user_params(rng)
        t = u["start"]
        for si, mood in enumerate(_mood_sequence(rng, N_SESSIONS), start=1):
            s = _session(rng, f"G{ui:03d}", si, u, mood, t)
            parts.append(s)
            t = s["timestamp"].iloc[-1] + pd.Timedelta(minutes=float(rng.uniform(2, 90)))
    return _inject_problems(pd.concat(parts, ignore_index=True), rng)


if __name__ == "__main__":
    out = config.RAW_FILES["generated"]
    df = generate()
    df.to_csv(out, index=False)
    print(f"wrote {out.name}: {len(df)} rows ({N_DUPLICATES} injected duplicates)")
    print(df.groupby(["user_id", "session_id"]).mood_state.nunique().max(), "mood(s) per session")
    print(df.mood_state.value_counts(normalize=True).round(3).to_dict())
