# LLM-Assisted Mood Detection from Keystroke Dynamics

A privacy-preserving pipeline that turns keystroke **timing** (never typed content) into an affective-state estimate, personalises it against each user's own typing baseline, and explains it with an LLM. College ML project; see [PRD.md](PRD.md) for the brief and [docs/final_report.md](docs/final_report.md) for the full write-up.

> All results are **performance on a synthetic benchmark**. They are not real-world emotion-recognition accuracy, and nothing here is a medical or psychological diagnosis.

## Project status

**Done:** everything in PRD §23 except a live LLM run. That covers the cleaning, EDA, features, personal baseline, temporal features, RF/XGBoost/SVM, user-grouped evaluation, ablation, SHAP, dashboard, notebooks, report and slides.

**Open:** a live LLM run. Add a free Groq key, run `python -m src.llm --n 8`, then update H4 in `docs/final_report.md`. All explanations so far come from the offline template.

**Changes from the PRD, and why:**

| PRD said | We did | Why |
|---|---|---|
| `key_event` = abstract category | Provided CSV had real letters, so cleaning maps them to 5 categories | R02: no typed content may be stored |
| One dataset | Provided CSV + a second dataset from `src/data_generator.py` | Provided labels change every keystroke, which makes window-level results near chance; the generated data checks that the pipeline itself works |
| OpenAI / any LLM | Groq free tier (default) or Claude, switched in `.env` | Free to run; Claude needs paid API credits |
| Ablation A/B/C/D | A/B/C plus E (optional extended features); D is the explanation layer | PRD §20 says D is not an accuracy step |
| 30-s windows | Kept, but sessions last 14–20 s, so each window covers one whole session | Session length is fixed by the data |

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # macOS also needs: brew install libomp  (for XGBoost)
cp .env.example .env                     # then paste a free GROQ_API_KEY into .env

# models and results are already in the repo, so the dashboard works straight away;
# re-run the pipeline only if you change the code
python -m src.model                      # clean → features → baseline → CV/ablation → save models (~5 min)
python -m src.llm --n 8                  # feature-grounded explanations for held-out windows
streamlit run app/dashboard.py           # dashboard

# second, pipeline-validation run on generated data
python -m src.data_generator
MOOD_RUN=generated python -m src.model
MOOD_RUN=generated streamlit run app/dashboard.py
```

The LLM layer uses **Groq's free tier** by default (`llama-3.3-70b-versatile`). Copy `.env.example` to `.env` (git-ignored) and paste a free key from https://console.groq.com/keys as `GROQ_API_KEY=...`. Set `LLM_PROVIDER=anthropic` to use Claude instead (needs API credits), or `template` for no LLM. If the call fails for any reason it falls back to a deterministic template and records why in `source`, so everything still runs offline. Turn on "Ask the LLM" in the dashboard sidebar to use it.

## Two datasets

| Run | File | What it is |
|---|---|---|
| `provided` (default) | `data/raw/synthetic_keystroke_raw_100k.csv` | The course dataset: 50 users × 20 sessions × 100 events. **Mood is re-drawn almost every keystroke** (average run 1.3 keys; every session contains all 4 moods). |
| `generated` | `data/raw/generated_keystroke_raw_100k.csv` | Made by `src/data_generator.py` (seed 42): one mood per session, moods persist across sessions (Markov, p_stay = 0.6), user-specific baselines, injected missing/invalid/duplicate/outlier values. |

The generated run is a **pipeline-validation check**: it shows the pipeline can recover a signal when labels are coherent. Outputs are kept separate (`data/processed/generated/`, `models/generated/`, `reports/generated/`).

## Pipeline

```
raw events ─► cleaning ─► 30-s windows ─► 9 core features ─┬─► personal baseline z-scores (leave-one-out, label-free)
 (src/cleaning.py)        (src/features.py)                 └─► temporal change vs previous 3 windows   (src/baseline.py)
                                                                         │
                         RF / XGBoost / SVM, user-grouped CV ◄───────────┘   (src/model.py)
                                     │
             structured JSON (no text) ─► LLM (Groq / Claude) ─► explanation + grounding check   (src/llm.py)
                                     │
                              Streamlit dashboard   (app/dashboard.py)
```

- **Privacy (R02).** The provided raw file stores real letters in `key_event`. Cleaning maps them to `{char, space, backspace, enter, shift}` and drops the original column before anything is saved. Nothing downstream, including the LLM, ever sees a character.
- **Evaluation.** Every split is grouped by `user_id`, so test users are never seen in training: 5-fold GroupKFold (40/10 users), a fixed 40/10 hold-out, and leave-one-user-out. Majority-class baseline, 200-run label-permutation test, and balanced accuracy / macro-F1 as headline metrics (63% of provided windows are "calm").

## Results (balanced accuracy, 5-fold user-grouped CV; chance = 0.25)

| Feature set | Provided: RF / XGB / SVM | Generated: RF / XGB / SVM |
|---|---|---|
| A raw window features | 0.27 / 0.25 / 0.29 | 0.54 / 0.53 / 0.56 |
| B + personal baseline | 0.28 / 0.29 / 0.31 | **0.78** / 0.73 / 0.76 |
| C + temporal trends | 0.29 / 0.28 / 0.30 | 0.77 / 0.76 / 0.76 |
| E C + extended features | 0.29 / 0.28 / 0.30 | 0.77 / 0.77 / 0.75 |
| majority class | 0.25 (accuracy 0.63) | 0.25 (accuracy 0.31) |
| permutation test (XGB, C) | p = 0.020 | p = 0.005 |
| leave-one-user-out (RF, C) | 0.30 | 0.77 |

| Hypothesis | Verdict |
|---|---|
| H1 features carry mood information | Supported. Weak but significant on the provided data, clear on generated data. |
| H2 personal baseline helps | Supported on generated data (+0.19 to +0.24); within noise on the provided data. |
| H3 temporal trends help | Not supported on either dataset. |
| H4 LLM gives grounded explanations | Template path verified: 16/16 sampled explanations reference real features, none use diagnostic language. **A live LLM run is still pending**: add `GROQ_API_KEY` and run `python -m src.llm --n 8`. |

**Why the provided data scores near chance:** keystroke-level timing does differ by mood (stressed flight ≈ 98 ms vs low ≈ 147 ms). But because the label changes every keystroke, a 30-s window is a mix of all four moods: mean label purity is 0.32, and window-level feature differences vanish. See `notebooks/02_eda.ipynb` and `03_feature_engineering.ipynb`.

Figures: `reports/figures/` and `reports/generated/figures/`. Raw numbers: `reports/*/results.json`.

## Repository layout

```
data/raw/                 provided + generated raw CSVs
data/processed/           cleaned events, window features   (generated/ for the second run)
notebooks/01–04           cleaning, EDA, features, modelling (built by notebooks/build_notebooks.py)
src/config.py             paths, constants, MOOD_RUN switch
src/data_generator.py     synthetic generator
src/cleaning.py           audit + cleaning rules
src/features.py           30-s windowing, core + extended features
src/baseline.py           leave-one-out personal baseline, temporal change
src/model.py              RF/XGBoost/SVM, ablation, permutation test, LOSO, SHAP, regression
src/llm.py                payload serializer, prompt, Groq/Claude call, template fallback, grounding check
app/dashboard.py          Streamlit dashboard
models/                   xgboost_model.pkl, random_forest_model.pkl, svm_model.pkl
reports/                  results.json, figures, llm_explanations.json
docs/final_report.md      methodology, results, limitations, viva notes
docs/presentation/        slide deck source (14 slides, published as a claude.ai artifact)
```

## Limitations

- Both datasets are synthetic, and their mood labels are generator assumptions, not self-reports. The generated dataset's effect sizes were chosen by us, so its high accuracy partly reflects those choices.
- The provided dataset's per-keystroke labels do not suit window-level classification; its results should be read as "near chance".
- 30-s windows are longer than a session (≈ 14 s provided, ≈ 20 s generated), so each window is effectively one session.
- The LLM explanation layer has been evaluated only through the template fallback.
- Not a diagnostic tool.
