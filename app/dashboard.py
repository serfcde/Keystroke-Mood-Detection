"""Streamlit dashboard (PRD §18).   Run:  streamlit run app/dashboard.py"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import baseline, config, llm  # noqa: E402

st.set_page_config(page_title="Keystroke Mood Prototype", layout="wide")


@st.cache_resource
def artifacts():
    return llm.load_artifacts()


bundle, windows, results = artifacts()
test = windows[windows["user_id"].isin(bundle["test_users"])]

st.title("Keystroke-dynamics affective-state prototype")
st.caption("Synthetic benchmark. Predictions are not real emotion detection and not a diagnosis.")

with st.sidebar:
    st.header("Select a window")
    st.write("Only users held out from training are listed.")
    user = st.selectbox("User", sorted(test["user_id"].unique()))
    user_windows = test[test["user_id"] == user].reset_index()
    pos = st.slider("Window (chronological)", 1, len(user_windows), len(user_windows) // 2)
    idx = int(user_windows.loc[pos - 1, "index"])
    use_llm = st.toggle(f"Ask the LLM ({config.LLM_PROVIDER})", value=False,
                        help="Provider and key are set in .env (LLM_PROVIDER, GROQ_API_KEY). Off = deterministic template.")
    st.divider()
    st.success("Typed content: NOT COLLECTED\n\nOnly timing and key category (char / space / backspace / enter / shift).")

payload = llm.payload_for(windows, bundle, results, idx)
row = windows.loc[idx]
rel = payload["model_reliability"]

c1, c2, c3 = st.columns(3)
c1.metric("Predicted state", payload["ml_prediction"]["state"].upper())
c2.metric("Model confidence", f"{payload['ml_prediction']['confidence']:.0%}")
c3.metric("Window's majority synthetic label", row["mood_state"], help=f"label purity {row['label_purity']:.0%}")
st.warning(f"Benchmark reliability: cross-validated balanced accuracy {rel['cv_balanced_accuracy']:.2f} "
           f"vs {rel['chance_balanced_accuracy']:.2f} chance on unseen users. Treat individual predictions as weak.")

left, right = st.columns(2)
with left:
    st.subheader("Current features vs personal baseline")
    table = pd.DataFrame({
        "current": payload["current"],
        "deviation (SD)": payload["baseline_deviation_z"],
        "trend": payload["recent_trend_vs_previous_windows"],
    })
    table.index = [llm.READABLE[f] for f in table.index]
    table["deviation (SD)"] = table["deviation (SD)"].map(lambda z: f"{'↑' if z > 0 else '↓'} {abs(z):.1f}")
    st.dataframe(table, width="stretch")
    st.subheader("Class probabilities")
    st.bar_chart(pd.Series(payload["ml_prediction"]["class_probabilities"]))

with right:
    st.subheader("Recent windows")
    feat = st.selectbox("Feature", config.CORE_FEATURES, format_func=llm.READABLE.get)
    hist = windows[(windows["user_id"] == user) & (windows["window_start"] <= row["window_start"])].tail(8)
    base = baseline.user_baseline(windows, user).loc[feat, "baseline_mean"]
    st.line_chart(pd.DataFrame({feat: hist[feat].to_numpy(), "baseline": base},
                               index=range(-len(hist) + 1, 1)))
    st.caption("x = windows before the selected one; 0 = selected window.")

    st.subheader("Explanation")
    exp, source = llm.explain(payload, use_llm=use_llm)
    st.write(exp.explanation)
    st.write("**Key features:** " + "; ".join(exp.important_features))
    st.caption(f"Confidence: {exp.confidence} · Agrees with ML: {exp.agrees_with_ml} · Source: {source}")
    st.caption(exp.uncertainty_note)

with st.expander("Structured payload sent to the LLM (no typed text)"):
    st.json(payload)
with st.expander("Global model explanation: XGBoost feature importance"):
    st.bar_chart(pd.Series(results["holdout"]["xgboost_importance"]).head(12))
