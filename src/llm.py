"""LLM explanation layer (PRD §15-17).

The LLM sees only a structured summary of timing features, baseline deviations, trends and the
ML prediction. It never receives typed text (the cleaned data no longer contains any).

Provider is chosen by LLM_PROVIDER in .env:
    groq       (default) free tier, needs GROQ_API_KEY; model from GROQ_MODEL
    anthropic  Claude via the Anthropic SDK, needs ANTHROPIC_API_KEY with API credits
    template   no LLM call
If the call fails for any reason, a deterministic template is used and the reason is recorded, so
the pipeline and dashboard always work offline.

    python -m src.llm --n 5        # explain 5 held-out windows, write reports/llm_explanations.json
"""
import argparse
import json
import os
import pickle
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from . import baseline, config
from .model import BalancedXGB  # noqa: F401  (needed to unpickle the saved model)

CLAUDE_MODEL = "claude-opus-5"

READABLE = {
    "typing_speed": "typing speed", "mean_dwell": "key hold (dwell) time", "std_dwell": "dwell variability",
    "mean_flight": "time between keys (flight)", "std_flight": "flight variability", "mean_pause": "pause length",
    "pause_count": "number of pauses", "long_pause_count": "number of long pauses",
    "correction_rate": "correction (backspace) rate",
}

SYSTEM_PROMPT = """You interpret keystroke-timing summaries for a research prototype that estimates affective state.

Rules:
1. Use only the numbers and labels in the supplied JSON. Do not invent facts about the person.
2. You never see what was typed; do not speculate about content.
3. The ML prediction is one input, not ground truth. Check whether the feature evidence supports it. If it does not, say so and set agrees_with_ml to false.
4. Name the specific features behind your reasoning, quoting their z-scores or trends.
5. State uncertainty plainly. Calibrate your confidence to model_reliability: compare the model's cross-validated balanced accuracy with chance (0.25), and remember the labels are synthetic, not self-reported.
6. Do not give medical or psychological diagnoses or advice.
Keep the explanation to 2-4 sentences."""


class Explanation(BaseModel):
    predicted_state: Literal["calm", "positive", "stressed", "low"]
    agrees_with_ml: bool
    confidence: Literal["low", "moderate", "high"]
    explanation: str
    important_features: list[str] = Field(description="Short phrases, each naming one supplied feature")
    uncertainty_note: str


def build_payload(window: pd.Series, history: pd.DataFrame, proba: np.ndarray,
                  user_std: pd.Series, model_reliability: dict) -> dict:
    """Structured, content-free representation of one window (PRD §15)."""
    top = int(np.argmax(proba))
    return {
        "current": {f: round(float(window[f]), 3) for f in config.CORE_FEATURES},
        "baseline_deviation_z": {f: round(float(window[baseline.z_col(f)]), 2) for f in config.CORE_FEATURES},
        "recent_trend_vs_previous_windows": {
            f: baseline.trend_label(window[baseline.change_col(f)], user_std[f]) for f in config.CORE_FEATURES},
        "windows_of_history": len(history),
        "ml_prediction": {"state": config.MOODS[top], "confidence": round(float(proba[top]), 2),
                          "class_probabilities": {m: round(float(p), 2) for m, p in zip(config.MOODS, proba)}},
        "model_reliability": model_reliability,
        "units": {"typing_speed": "keys/s", "dwell/flight/pause": "ms", "correction_rate": "fraction of keys"},
    }


def explain_with_groq(payload: dict) -> Explanation:
    """Groq's free tier. JSON mode works on every Groq model; the schema is enforced by validation."""
    from groq import Groq

    schema = json.dumps(Explanation.model_json_schema())
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nReply with only a JSON object that matches this JSON Schema:\n{schema}"},
        {"role": "user", "content": json.dumps(payload, indent=1)},
    ]
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is empty: add your free key to .env (https://console.groq.com/keys)")
    client = Groq()
    for _ in range(2):  # one retry if the reply does not match the schema
        response = client.chat.completions.create(
            model=config.GROQ_MODEL, messages=messages, temperature=0.2, max_tokens=2000,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content
        try:
            return Explanation.model_validate_json(text)
        except ValidationError as e:
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"That JSON does not match the schema ({e.errors()[:3]}). "
                                                     "Reply again with only the corrected JSON object."}]
    raise RuntimeError("Groq reply did not match the explanation schema after a retry")


def explain_with_claude(payload: dict) -> Explanation:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(payload, indent=1)}],
        output_format=Explanation,
    )
    if response.stop_reason != "end_turn" or response.parsed_output is None:
        raise RuntimeError(f"Claude did not return a complete explanation (stop_reason={response.stop_reason})")
    return response.parsed_output


def explain_with_template(payload: dict) -> Explanation:
    """Deterministic fallback: names the three largest baseline deviations."""
    z = pd.Series(payload["baseline_deviation_z"])
    biggest = z.abs().sort_values(ascending=False).head(3).index
    phrases = [f"{READABLE[f]} {'above' if z[f] > 0 else 'below'} baseline ({z[f]:+.1f} SD)" for f in biggest]
    pred = payload["ml_prediction"]
    return Explanation(
        predicted_state=pred["state"], agrees_with_ml=True, confidence="low",
        explanation=(f"The classifier suggests '{pred['state']}' (p={pred['confidence']:.2f}). "
                     f"The largest departures from this user's usual typing are: {'; '.join(phrases)}."),
        important_features=phrases,
        uncertainty_note=(f"Template explanation (no LLM call). The model's cross-validated balanced accuracy on "
                          f"unseen users is {payload['model_reliability']['cv_balanced_accuracy']:.2f} vs 0.25 chance, "
                          f"on synthetic labels."),
    )


PROVIDERS = {
    "groq": (explain_with_groq, lambda: f"groq:{config.GROQ_MODEL}"),
    "anthropic": (explain_with_claude, lambda: f"claude:{CLAUDE_MODEL}"),
}


def explain(payload: dict, use_llm=True) -> tuple[Explanation, str]:
    """Returns (explanation, source): source is e.g. 'groq:<model>', or 'template: <reason>' on failure."""
    if use_llm and config.LLM_PROVIDER in PROVIDERS:
        fn, label = PROVIDERS[config.LLM_PROVIDER]
        try:
            return fn(payload), label()
        except Exception as e:  # missing key, network, rate limit, refusal: fall back, but say why
            reason = getattr(e, "body", None) or str(e)
            if isinstance(reason, dict):
                reason = reason.get("error", {}).get("message", str(e))
            return explain_with_template(payload), f"template: {type(e).__name__}: {reason}"[:300]
    return explain_with_template(payload), "template"


def grounding_check(payload: dict, exp: Explanation) -> dict:
    """PRD §19.4: does the explanation reference real features and agree with the model?"""
    text = (exp.explanation + " " + " ".join(exp.important_features)).lower()
    keywords = {f: [f.replace("_", " "), READABLE[f].split(" (")[0]] for f in config.CORE_FEATURES}
    keywords["mean_dwell"] += ["dwell"]; keywords["mean_flight"] += ["flight"]; keywords["correction_rate"] += ["correction", "backspace"]
    keywords["pause_count"] += ["pause"]; keywords["typing_speed"] += ["speed", "slower", "faster"]
    referenced = [f for f, kws in keywords.items() if any(k in text for k in kws)]
    return {
        "features_referenced": referenced,
        "references_any_feature": bool(referenced),
        "matches_ml_state": exp.predicted_state == payload["ml_prediction"]["state"],
        "claims_agreement": exp.agrees_with_ml,
        "confidence_label": exp.confidence,
        "mentions_uncertainty": bool(exp.uncertainty_note.strip()),
        "diagnostic_language": any(w in text for w in ["diagnos", "disorder", "depression", "anxiety"]),
    }


# ---------- helpers shared with the dashboard ----------

def load_artifacts():
    with open(config.MODEL_DIR / "xgboost_model.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    windows = pd.read_csv(config.WINDOW_PATH, parse_dates=["window_start"])
    results = json.loads((config.REPORT_DIR / "results.json").read_text())
    return bundle, windows, results


def reliability_summary(results: dict) -> dict:
    cv = {(r["feature_set"], r["model"]): r for r in results["cross_validation"]}
    best = cv[("C_+baseline+temporal", "xgboost")]
    return {"benchmark": "synthetic", "cv_balanced_accuracy": round(best["balanced_accuracy_mean"], 3),
            "chance_balanced_accuracy": 0.25,
            "window_label_purity": round(results["label_purity_mean"], 2)}


def payload_for(windows: pd.DataFrame, bundle: dict, results: dict, idx: int) -> dict:
    row = windows.loc[idx]
    user = windows[windows["user_id"] == row["user_id"]]
    history = user[user["window_start"] < row["window_start"]].tail(config.TREND_WINDOWS)
    proba = bundle["model"].predict_proba(row[bundle["features"]].to_frame().T.astype(float))[0]
    return build_payload(row, history, proba, user[config.CORE_FEATURES].std(), reliability_summary(results))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    bundle, windows, results = load_artifacts()
    test = windows[windows["user_id"].isin(bundle["test_users"])]
    picks = test.groupby("mood_state").head(2).index[: args.n]
    out = []
    for idx in picks:
        payload = payload_for(windows, bundle, results, idx)
        exp, source = explain(payload, use_llm=not args.no_llm)
        out.append({"window": windows.loc[idx, ["user_id", "session_id", "window"]].astype(str).to_dict(),
                    "true_window_mood": windows.loc[idx, "mood_state"], "payload": payload,
                    "explanation": exp.model_dump(), "source": source, "grounding": grounding_check(payload, exp)})
        if source.startswith("template: "):
            print(f"LLM call failed, used template instead. Reason: {source[len('template: '):]}")
        print(f"[{source.split(':')[0]}] {exp.predicted_state}: {exp.explanation}\n")
    (config.REPORT_DIR / "llm_explanations.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
