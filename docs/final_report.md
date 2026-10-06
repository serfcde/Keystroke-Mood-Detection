# Final Report — LLM-Assisted Mood Detection from Keystroke Dynamics

**Research question.** Can privacy-preserving keystroke behavioural features, personalised against a user's normal typing baseline and combined with simple temporal context, support affective-state prediction, and can an LLM provide feature-grounded explanations without accessing typed content?

**Short answer (synthetic benchmark only).** Partly. Personalisation is the component that matters: on coherently labelled data it lifts balanced accuracy from about 0.55 to about 0.77. Simple temporal trend features add nothing. On the provided course dataset every model sits just above chance, because that dataset's labels change on almost every keystroke. The LLM layer is implemented and produces grounded explanations through its fallback path; a live LLM run (Groq free tier by default) is still pending.

---

## 1. Data

### 1.1 Provided dataset
`data/raw/synthetic_keystroke_raw_100k.csv`: 100,000 events, 50 users × 20 sessions × 100 events, with the PRD §5.2 schema.

The audit (`notebooks/01_data_cleaning.ipynb`) found:

| Check | Finding |
|---|---|
| Duplicates, bad timestamps, out-of-order sessions | none |
| Missing values | 150 dwell, 150 flight |
| Invalid values | 100 negative dwell times |
| Long flights | 695 events > 2 s (max 3.5 s) |
| `key_event` | **real letters a–z** plus space/backspace/enter/shift, which breaks R02 |
| `is_correction` | 2,348 backspaces not flagged; every flagged correction is a backspace |
| `event_index` | counts per user (1–2000), not per session |
| Session length | 8–26 s (mean 14 s), shorter than one 30-s window |
| **Label structure** | **every session contains all 4 moods; a mood lasts 1.3 keystrokes on average** |

### 1.2 Generated dataset
`src/data_generator.py` (seed 42) follows PRD §5.3:
- 50 users with individual dwell, flight, variability, pause and correction baselines.
- One mood per session, persisting to the next session with probability 0.6.
- Mood effects scale each user's baseline with a random per-session intensity (0.5–1.2), so moods overlap.
- Valence/arousal are drawn around a mood-specific centre.
- Only abstract key categories are generated.
- Problems are injected on purpose: 0.15% missing dwell and flight, 0.1% negative dwell, stuck keys over 1 s, 0.1% very long gaps (5–15 s), and 50 duplicate rows.

**Caveat.** The effect sizes are our own assumptions. High accuracy on this data shows the pipeline works, not that keystrokes reveal mood.

## 2. Methodology

### 2.1 Cleaning (`src/cleaning.py`)
Each rule is backed by an audit finding:
1. Drop exact duplicates.
2. Sort events by user, session and time; re-index events within each session.
3. Map `key_event` to `{char, space, backspace, enter, shift}` and drop the raw column (privacy).
4. Recompute `is_correction` as "key is backspace".
5. Mark dwell values below 0 or above 1 s as missing.
6. Recover missing flight times from timestamp differences: flight equals the gap to under 1 µs, which recovered all 150.
7. Impute the remaining gaps with the user's median, since users differ.
8. **Keep** flights over 500 ms as pauses. They form a second mode in the log-flight histogram, so they are behaviour, not noise.

### 2.2 Features (`src/features.py`)
- 30-s windows that never cross session boundaries. Because sessions are shorter than 30 s, each window equals one session.
- **Core features (PRD §9):**
  - `typing_speed`: keys per second, counting pauses.
  - Mean and std of dwell.
  - Mean and std of flight, excluding pauses.
  - `mean_pause`, `pause_count` (over 500 ms) and `long_pause_count` (over 2 s).
  - `correction_rate`.
- **Extended features (optional):**
  - Median dwell.
  - Dwell and flight coefficient of variation.
  - char→char latency: a content-free stand-in for digraph latency, since real digraphs would need the characters we removed.
  - Mean burst length between pauses.
  - Key-category entropy.
- **Labels:**
  - The window's majority mood, plus `label_purity` (the share of keystrokes carrying that mood).
  - Mean valence and mean arousal.

### 2.3 Personal baseline and temporal context (`src/baseline.py`)
- **Baseline.** z = (x − μ₋ᵢ) / σ₋ᵢ, where μ₋ᵢ and σ₋ᵢ are the mean and standard deviation of the same user's *other* windows. It is leave-one-out and uses no labels, so a test user's baseline needs only their unlabelled typing, as a deployed system would after a calibration period.
- **Temporal.** `<feature>_change` is the current value minus the mean of the user's previous 3 windows (0 for the first window). The PRD's "increasing / decreasing / stable" trend labels (used in the LLM payload) come from this change divided by the user's std, with a ±0.5 threshold.

### 2.4 Models and evaluation (`src/model.py`)
- **Models:**
  - Random Forest (300 trees, balanced class weights).
  - XGBoost (300 trees, depth 3, balanced sample weights).
  - RBF-SVM (standardised features, balanced class weights).
  - A majority-class baseline.
- **Feature sets:**
  - A: core.
  - B: A + baseline z-scores.
  - C: B + temporal change.
  - E: C + extended.
  - (D in the PRD is the LLM layer, which doesn't change accuracy.)
- **Evaluation, always grouped by user:**
  - 5-fold GroupKFold, so each fold trains on 40 users and tests on 10.
  - A fixed 40/10 hold-out used for the confusion matrix and the saved models.
  - Leave-one-user-out.
  - A 200-run label-permutation test.
- **Metrics:**
  - Headline: balanced accuracy and macro-F1, because 63% of provided windows are calm and accuracy alone is misleading.
  - Also reported: accuracy, per-class precision and recall, and the confusion matrix.
  - Option B regression: XGBoost on window-mean valence and arousal (MAE, RMSE, R²).
- **ML explanation:** XGBoost gain importance and mean |SHAP| on the hold-out users.

### 2.5 LLM layer (`src/llm.py`)
- **Payload (JSON, no text):**
  - The current features.
  - Baseline z-scores.
  - Trend labels.
  - The ML prediction with class probabilities.
  - A `model_reliability` block: the cross-validated balanced accuracy and chance level.
- **System prompt (the PRD §16 rules):**
  - Use only the supplied data.
  - Do not speculate about content.
  - Treat the ML prediction as an input, not ground truth.
  - Cite features with their z-scores.
  - Calibrate confidence to `model_reliability`.
  - Give no diagnosis.
- **Output:** a structured Pydantic object (`predicted_state`, `agrees_with_ml`, `confidence`, `explanation`, `important_features`, `uncertainty_note`) produced by the provider set in `.env`: Groq's free tier by default (`llama-3.3-70b-versatile`, JSON mode, validated against the schema with one retry), or Claude (`claude-opus-5`, structured outputs). The Claude path was tried, but the account had no API credits.
- **Fallback:** a deterministic template that names the three largest baseline deviations.
- **`grounding_check()`:** automates the PRD §19.4 questions (feature references, agreement with the model, uncertainty, diagnostic language).

## 3. Results

### 3.1 Ablation (balanced accuracy, mean of 5 user-grouped folds; chance 0.25)

| Set | Provided RF | XGB | SVM | Generated RF | XGB | SVM |
|---|---|---|---|---|---|---|
| A | 0.269 | 0.252 | 0.294 | 0.542 | 0.532 | 0.560 |
| B | 0.284 | 0.291 | 0.313 | **0.777** | 0.726 | 0.757 |
| C | 0.288 | 0.278 | 0.302 | 0.773 | 0.757 | 0.755 |
| E | 0.290 | 0.279 | 0.302 | 0.772 | 0.771 | 0.746 |

On both datasets the fold standard deviation is 0.01–0.08 (`reports/*/figures/ablation.png`).

### 3.2 Significance and other evaluations

| | Provided | Generated |
|---|---|---|
| Permutation test (XGB, C): real vs permuted mean | 0.278 vs 0.249, **p = 0.020** | 0.757 vs 0.267, **p = 0.005** |
| Leave-one-user-out balanced accuracy, RF / XGB / SVM (C) | 0.296 / 0.279 / 0.283 | 0.769 / 0.746 / 0.752 |
| Hold-out (10 users), XGB: accuracy / balanced accuracy / macro-F1 | 0.48 / 0.26 / 0.26 (majority class: 0.62 accuracy) | 0.80 / 0.80 / 0.81 |
| Valence regression R², XGB vs mean predictor | −0.12 vs 0.00 | **0.54** vs 0.00 |
| Arousal regression R² | 0.03 | **0.50** |

**Generated hold-out confusion matrix (XGB; rows = true, columns = predicted; order calm, positive, stressed, low):**

```
calm      40 18  1  2
positive   8 39  1  0
stressed   2  3 45  0
low        8  0  0 45
```

Calm and positive are confused most often. That matches the generator, whose positive effects are the mildest. On the provided data, the model mostly predicts calm and almost never predicts low, which makes up only 2.7% of windows.

### 3.3 What the models use
- **Generated data:** the largest mean |SHAP| values are `mean_flight_z`, `std_dwell_z` and `mean_dwell_z`. Deviations from a person's own baseline dominate, which is the mechanism behind H2.
- **Provided data:** importance is spread thinly across many features, which is typical of a model fitting noise.

### 3.4 Hypotheses

| | Verdict |
|---|---|
| **H1** | Supported. Significant on both datasets; very weak on the provided one. |
| **H2** | Supported on generated data (+0.19 to +0.24 balanced accuracy, A→B). On provided data +0.02 to +0.04, within noise. |
| **H3** | Not supported (B→C changes between −0.01 and +0.03). A likely reason is that a "change" score is near zero whenever the mood stays the same between sessions, so it carries little information about *which* mood it is. Rolling-mean features of the previous windows' raw values would be a better test. |
| **H4** | Template path: 16/16 sampled explanations reference real payload features, match the ML state, carry an uncertainty note and contain no diagnostic language. **A live LLM run is still pending**: add a free `GROQ_API_KEY` to `.env`, run `python -m src.llm --n 8`, and inspect `reports/llm_explanations.json`. |

## 4. Limitations
1. **Synthetic labels.** No result here measures human emotion. The generated data's separability reflects effect sizes we chose.
2. **Provided-data label structure.** Per-keystroke mood re-sampling (label purity 0.32) caps any window-level classifier near chance. We report that honestly instead of tuning the setup until the numbers look good.
3. **Window = session.** With 30-s windows and sessions of 14–20 s, temporal context is measured across sessions, not within them.
4. **Leave-one-out baseline.** It uses all of a user's other windows, including later ones. A strictly causal calibration period would be more realistic.
5. **Small evaluation.** 50 users and about 1,000 windows.
6. **LLM evaluation.** So far only the template fallback has been tested, and the grounding check matches keywords rather than meaning.
7. **Not a diagnostic or monitoring tool.**

## 5. Future work
1. A human-participant study: collect timing only, with periodic self-reported valence/arousal.
2. Causal calibration-period baselines.
3. Rolling-mean temporal features.
4. Within-session windows of 5–10 s once sessions are longer.
5. Run the LLM explanations and compare them against SHAP.
6. Fairness analysis across typing-speed groups.

## 6. Viva notes
- **Why synthetic data?** No participant data was available. Synthetic data lets us validate cleaning, features, personalisation, evaluation and explanation end to end. We do not claim real-world accuracy.
- **Why two datasets?** The provided one's labels change every keystroke, so its near-chance result is a property of the data, not proof the pipeline is broken. The generated dataset, with one mood per session, shows the same code recovers a signal when one exists.
- **How is privacy enforced?** Characters are mapped to five categories during cleaning, and the raw column is dropped before anything is saved. The LLM receives only aggregate numbers.
- **Why balanced accuracy?** On the provided data, always guessing calm gets 63% accuracy. Balanced accuracy gives 0.25 to any constant guess.
- **Why group by user?** A random row split would put the same person in both training and test, which inflates results. All our splits keep test users unseen.

## Reproduce
See the README Quick start. Every number above comes from `reports/results.json` and `reports/generated/results.json`.
