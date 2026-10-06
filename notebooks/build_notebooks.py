"""Regenerates the four notebooks from source.  Run from repo root:
    python notebooks/build_notebooks.py && jupyter nbconvert --execute --inplace notebooks/0*.ipynb
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
SETUP = """import sys, json
sys.path.insert(0, '..')
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from src import config, cleaning, features, baseline
pd.set_option('display.width', 160); pd.set_option('display.max_columns', 40)
sns.set_theme(style='whitegrid')
config.FIG_DIR.mkdir(parents=True, exist_ok=True)
def save(name): plt.tight_layout(); plt.savefig(config.FIG_DIR / f'{name}.png', dpi=130)"""


def nb(cells):
    book = nbf.v4.new_notebook()
    book.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c) for c in cells]
    book.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return book


notebooks = {}

notebooks["01_data_cleaning"] = [
    "md:# 01 — Data inspection and cleaning\nPRD §7. Every cleaning rule below is justified by a check on the raw file.",
    SETUP,
    "raw = cleaning.load_raw()\nraw.head()",
    "md:## Audit of the raw file",
    "report = cleaning.audit(raw)\nfor k, v in report.items():\n    print(f'{k:35s} {v}')",
    "md:### Privacy check: `key_event` stores real characters\nR02 requires that typed content is never stored. The raw file contains the actual letters, so they are mapped to categories during cleaning and never written back out.",
    "raw['key_event'].value_counts().head(10)",
    "md:### Label structure: mood changes almost every keystroke\nThis is the most important property of the dataset and limits everything downstream (see notebook 03).",
    "raw.loc[raw.user_id == 'U001'].head(15)[['session_id', 'event_index', 'key_event', 'mood_state', 'valence', 'arousal']]",
    "md:### Invalid and missing timing values",
    "print(raw[['dwell_time_ms', 'flight_time_ms']].describe().round(1))\nprint('negative dwell:', (raw.dwell_time_ms < 0).sum())",
    "md:`flight_time_ms` is the difference between consecutive timestamps, so a missing flight can be recovered exactly instead of imputed:",
    "ts = pd.to_datetime(raw.timestamp)\ngap = ts.groupby([raw.user_id, raw.session_id]).diff().dt.total_seconds() * 1000\nprint('max |gap - flight| (ms):', (gap - raw.flight_time_ms).abs().quantile([.5, .99]).round(6).to_dict())",
    "md:### Correction flag\nEvery `is_correction=1` row is a backspace, but some backspaces are not flagged. A backspace is a correction by definition, so the flag is recomputed from the key category.",
    "pd.crosstab(raw.key_event.eq('backspace'), raw.is_correction, rownames=['is backspace'])",
    "md:## Apply cleaning",
    "clean, log = cleaning.clean(raw)\nprint('\\n'.join(log))\nclean.to_csv(config.CLEAN_PATH, index=False)\nclean.head()",
    "assert 'key_event' not in clean.columns and set(clean.key_category) <= {'char','space','backspace','enter','shift'}\nassert clean[['dwell_time_ms','flight_time_ms']].isna().sum().sum() == 0 and (clean.dwell_time_ms > 0).all()\nprint(clean.shape, clean.groupby('user_id').size().describe()[['min','max']].to_dict())",
    "md:## Summary of cleaning decisions\n| Problem found | Decision | Why |\n|---|---|---|\n| Real characters in `key_event` | Map to 5 categories, drop original | R02: typed content must not be stored |\n| 2,348 unflagged backspaces | `is_correction = (key == backspace)` | Every flagged correction is a backspace |\n| 100 negative dwell times | Set missing, impute with user median | A key cannot be held for negative time |\n| 150 missing flight times | Recover from timestamp differences, else user median | Flight equals the timestamp gap to < 1 µs |\n| 150 missing dwell times | User median | Users differ in dwell (between-user CV ≈ 8%), so a global median would bias them |\n| Flights > 500 ms (~1%) | **Kept** as pauses | They form a separate mode (see 02_eda plot 2); they are behaviour, not noise |\n| No duplicates, unparseable timestamps or out-of-order sessions | Nothing to do | Checked anyway |"
]

notebooks["02_eda"] = [
    "md:# 02 — Exploratory data analysis\nPRD §8: the ten required plots and the six questions. Event-level plots use the cleaned keystrokes; window-level plots use 30-second windows (notebook 03).",
    SETUP,
    "ev = pd.read_csv(config.CLEAN_PATH, parse_dates=['timestamp'])\nwin = features.build_windows(ev)\nprint(ev.shape, win.shape)",
    "md:### 1. Dwell-time distribution",
    "fig, ax = plt.subplots(figsize=(7,3.5)); sns.histplot(ev.dwell_time_ms, bins=80, ax=ax); ax.set(title='Dwell time', xlabel='ms'); save('eda_01_dwell')\nprint('skew:', round(ev.dwell_time_ms.skew(), 2))",
    "md:### 2. Flight-time distribution (log scale: pauses form a second mode)",
    "fig, ax = plt.subplots(figsize=(7,3.5)); sns.histplot(ev.flight_time_ms, bins=100, log_scale=True, ax=ax)\nax.axvline(config.PAUSE_MS, c='r', ls='--', label=f'pause threshold {config.PAUSE_MS} ms'); ax.legend(); ax.set(title='Flight time', xlabel='ms (log)'); save('eda_02_flight')\nprint('skew:', round(ev.flight_time_ms.skew(), 2), '| share > pause threshold:', round(ev.is_pause.mean(), 4))",
    "md:### 3. Typing-speed distribution (per window)",
    "fig, ax = plt.subplots(figsize=(7,3.5)); sns.histplot(win.typing_speed, bins=40, ax=ax); ax.set(title='Typing speed per window', xlabel='keys / s'); save('eda_03_speed')",
    "md:### 4. Pause-duration distribution",
    "fig, ax = plt.subplots(figsize=(7,3.5)); sns.histplot(ev.loc[ev.is_pause, 'flight_time_ms'], bins=50, ax=ax); ax.set(title='Pause duration (flight > 500 ms)', xlabel='ms'); save('eda_04_pause')",
    "md:### 5. Correction-rate distribution (per window)",
    "fig, ax = plt.subplots(figsize=(7,3.5)); sns.histplot(win.correction_rate, bins=30, ax=ax); ax.set(title='Correction rate per window', xlabel='fraction of keys'); save('eda_05_correction')",
    "md:### 6. Mood-state distribution: events vs window majority labels",
    "fig, axes = plt.subplots(1, 2, figsize=(10,3.5), sharey=True)\nev.mood_state.value_counts(normalize=True).reindex(config.MOODS).plot.bar(ax=axes[0], title='Per keystroke')\nwin.mood_state.value_counts(normalize=True).reindex(config.MOODS).plot.bar(ax=axes[1], title='Per 30-s window (majority)')\nsave('eda_06_mood')",
    "md:### 7. User-level typing-speed comparison",
    "order = win.groupby('user_id').typing_speed.median().sort_values().index\nfig, ax = plt.subplots(figsize=(14,4)); sns.boxplot(win, x='user_id', y='typing_speed', order=order, ax=ax)\nax.tick_params(axis='x', rotation=90); ax.set(title='Typing speed by user', xlabel=''); save('eda_07_users')\nu = win.groupby('user_id')[config.CORE_FEATURES].mean()\nprint('between-user CV of mean feature:'); print((u.std()/u.mean()).round(3))",
    "md:### 8. Correlation heatmap (window features and labels)",
    "fig, ax = plt.subplots(figsize=(9,7)); sns.heatmap(win[config.CORE_FEATURES + ['valence','arousal']].corr(), annot=True, fmt='.2f', cmap='vlag', center=0, ax=ax); save('eda_08_corr')",
    "md:### 9. Feature distributions by mood\nShown at keystroke level (where the label is exact) and at window level (where the label is a majority vote).",
    "fig, axes = plt.subplots(1, 3, figsize=(14,3.5))\nfor ax, col in zip(axes, ['dwell_time_ms', 'flight_time_ms', 'is_correction']):\n    if col == 'is_correction':\n        ev.groupby('mood_state')[col].mean().reindex(config.MOODS).plot.bar(ax=ax, title='correction rate (event)')\n    else:\n        sns.boxplot(ev[~ev.is_pause], x='mood_state', y=col, order=config.MOODS, showfliers=False, ax=ax); ax.set_title(f'{col} (event, pauses excluded)')\nsave('eda_09a_by_mood_events')\nfig, axes = plt.subplots(1, 3, figsize=(14,3.5))\nfor ax, col in zip(axes, ['mean_dwell', 'mean_flight', 'typing_speed']):\n    sns.boxplot(win, x='mood_state', y=col, order=config.MOODS, showfliers=False, ax=ax); ax.set_title(f'{col} (window)')\nsave('eda_09b_by_mood_windows')",
    "effect = ev[~ev.is_pause].groupby('mood_state')[['dwell_time_ms','flight_time_ms','is_correction']].mean().reindex(config.MOODS)\nprint('event-level means by mood'); print(effect.round(3))\nprint('\\nwindow-level means by majority mood'); print(win.groupby('mood_state')[config.CORE_FEATURES].mean().reindex(config.MOODS).round(3))",
    "md:### 10. Valence vs arousal",
    "s = ev.sample(5000, random_state=0)\nfig, ax = plt.subplots(figsize=(6,5)); sns.scatterplot(s, x='valence', y='arousal', hue='mood_state', hue_order=config.MOODS, s=8, alpha=.5, ax=ax); save('eda_10_va')\nprint('valence-arousal correlation (events):', round(ev.valence.corr(ev.arousal), 3))\nprint('valence-arousal correlation (windows):', round(win.valence.corr(win.arousal), 3))",
    "md:## Answers to the PRD §8 questions\n**Are timing distributions skewed?** Yes. Dwell is mildly right-skewed (skew ≈ 1.0). Flight is heavily right-skewed (skew ≈ 9.4) because of a second mode of pauses above 500 ms (about 1% of keystrokes), which is why pauses get their own features and are excluded from `mean_flight`.\n\n**Do users have different baselines?** Yes. Between-user coefficient of variation is 15% for typing speed, 17% for flight time and 8% for dwell time (plot 7). That is the motivation for personal baselines.\n\n**Which features vary most between states?** Per keystroke: flight time (stressed ≈ 98 ms vs low ≈ 147 ms) and dwell time (stressed ≈ 97 ms vs low ≈ 124 ms). Per 30-second window those gaps nearly vanish (window mean flight 118–121 ms for every mood) because each window mixes all four moods.\n\n**Are valence and arousal correlated?** Weakly: r ≈ 0.21 per keystroke, 0.17 per window.\n\n**Are there outliers?** 100 negative dwell values (invalid, fixed in 01) and long flights up to ~3.5 s (kept as pauses).\n\n**Does correction behaviour differ by state?** Slightly: stressed has the highest correction rate (≈ 7.2% of keys vs ≈ 5.6–6.6% for the others).\n\n**Key EDA finding:** the dataset contains a real per-keystroke mood signal, but mood is re-drawn almost every keystroke (average run 1.3 keys), so 30-second windows average it away. This predicts the weak window-level results in notebook 04."
]

notebooks["03_feature_engineering"] = [
    "md:# 03 — Windowing, features, personal baseline and temporal context\nPRD §9–11.",
    SETUP,
    "ev = pd.read_csv(config.CLEAN_PATH, parse_dates=['timestamp'])\ndur = ev.groupby(['user_id','session_id']).timestamp.agg(lambda t: (t.max()-t.min()).total_seconds())\nprint('session duration (s):'); print(dur.describe().round(1))",
    "md:Sessions last about 14 s, shorter than the 30-s window, so each 30-s window covers one whole session (1,000 windows). The window length is `config.WINDOW_S` if you want to experiment.",
    "win = features.build_windows(ev)\nprint(win.shape); win[['user_id','session_id','window','n_events'] + config.CORE_FEATURES + ['mood_state','label_purity']].head()",
    "md:### Optional extended features (PRD §9)\nMedian dwell, dwell/flight coefficient of variation, char→char latency (the content-free stand-in for digraph latency; per-letter digraphs would need the characters we removed), mean burst length between pauses, and entropy of the key-category mix.",
    "win[config.EXTENDED_FEATURES].describe().round(3)",
    "md:## How clean is a window label?\nBecause the raw mood changes almost every keystroke, the window label is a majority vote over a mix of all four moods. `label_purity` is the share of the window's keystrokes that carry the majority label.",
    "fig, ax = plt.subplots(figsize=(7,3.5)); sns.histplot(win.label_purity, bins=25, ax=ax); ax.set(title='Window label purity', xlabel='share of keystrokes with the majority mood'); save('fe_label_purity')\nprint(win.label_purity.describe().round(3))\nprint(pd.crosstab(win.mood_state, 'n'))",
    "md:## Personal baseline (leave-one-out z-scores)\nFor each window: z = (value − mean of the user's other windows) / std of the user's other windows. No labels are used.",
    "win = baseline.add_baseline_deviation(win)\nu = win.user_id.iloc[0]\nprint(baseline.user_baseline(win, u).round(3))\nwin.loc[win.user_id == u, ['typing_speed', 'typing_speed_z', 'correction_rate', 'correction_rate_z']].head()",
    "md:## Temporal context\nChange = current value − mean of the previous 3 windows of the same user (sessions are back to back in time).",
    "win = baseline.add_temporal(win)\nwin.loc[win.user_id == u, ['window_start', 'typing_speed', 'typing_speed_change', 'pause_count', 'pause_count_change']].head(6)",
    "win.to_csv(config.WINDOW_PATH, index=False)\nprint('saved', config.WINDOW_PATH.name, win.shape)",
    "md:**Takeaway.** Mean label purity is ≈ 0.32: in a typical window only about a third of keystrokes carry the label the window is given, barely above the 0.25 of a random mix. Any classifier trained on these labels has a low ceiling, so results in notebook 04 must be read against chance, not against 100%."
]

notebooks["04_model_training"] = [
    "md:# 04 — Model training, user-independent evaluation and ablation\nPRD §12–14, §17, §19–20. All splits are grouped by `user_id`: test users never appear in training.\n\nThis notebook reads the results written by `python -m src.model` (about 5 minutes) instead of retraining everything, so it opens and re-runs quickly.",
    SETUP + "\nfrom src import model\nfrom IPython.display import Image, display\nres = json.loads((config.REPORT_DIR / 'results.json').read_text())\nwin = pd.read_csv(config.WINDOW_PATH, parse_dates=['window_start'])\nprint(res['n_windows'], 'windows,', win.user_id.nunique(), 'users, mean label purity', round(res['label_purity_mean'], 3))\nprint('window labels:', res['window_label_distribution'])",
    "md:## Ablation: A (raw) → B (+personal baseline) → C (+temporal) → E (C + optional extended features)\n5-fold GroupKFold: each fold trains on 40 users and tests on 10 unseen users. Accuracy is misleading because most windows are labelled calm, so **balanced accuracy (chance = 0.25)** and **macro-F1** are the headline metrics.",
    "cv = pd.DataFrame(res['cross_validation'])\ncv[['feature_set','model','accuracy_mean','balanced_accuracy_mean','balanced_accuracy_std','macro_f1_mean']].round(3)",
    "display(Image(config.FIG_DIR / 'ablation.png', width=700))",
    "md:## Is the model better than chance?\nPermutation test: labels are shuffled 200 times and the whole grouped CV is re-run (XGBoost, feature set C).",
    "res['permutation_test']",
    "md:## Leave-one-user-out (PRD §14, advanced)",
    "pd.DataFrame(res['leave_one_user_out']).round(3)",
    "md:## Option B: valence/arousal regression (window means)",
    "pd.DataFrame(res['regression']).T.round(4)",
    "md:## Final models on a 40/10 user split",
    "hold = res['holdout']\nprint('test users:', hold['test_users'])\nfor name in ['random_forest', 'xgboost', 'svm']:\n    print(f'{name:14s}', {k: round(v, 3) for k, v in hold[name].items() if isinstance(v, float)})\nprint('majority-class accuracy on these test users:', round(hold['majority_class_on_test'], 3))\npd.DataFrame(hold['xgboost']['report']).T.round(3)",
    "display(Image(config.FIG_DIR / 'confusion_matrix.png', width=420))",
    "md:## ML explanation layer: feature importance and SHAP (PRD §17)",
    "display(Image(config.FIG_DIR / 'feature_importance.png', width=800))\npd.DataFrame({'gain': hold['xgboost_importance'], 'mean_abs_shap': hold['xgboost_shap']}).sort_values('mean_abs_shap', ascending=False).head(10).round(4)",
    "md:## LLM explanation layer (PRD §15–16, §19.4)\nWritten by `python -m src.llm`. `source` shows whether the LLM (Groq or Claude) or the offline template produced each explanation; `grounding` is the automatic §19.4 check.",
    "p = config.REPORT_DIR / 'llm_explanations.json'\nexps = json.loads(p.read_text()) if p.exists() else []\nfor e in exps:\n    print(f\"[{e['source'][:30]}] true={e['true_window_mood']} -> {e['explanation']['predicted_state']} ({e['explanation']['confidence']})\")\n    print('  ', e['explanation']['explanation'])\npd.DataFrame([e['grounding'] for e in exps])",
    "md:## Pipeline validation on the generated dataset\nThe same code run with `MOOD_RUN=generated` on data from `src/data_generator.py` (one mood per session). This checks that the pipeline *can* recover a signal when the labels are coherent; it says nothing about real people.",
    "gp = config.ROOT / 'reports/generated/results.json'\nif gp.exists():\n    g = json.loads(gp.read_text())\n    both = pd.DataFrame(res['cross_validation']).merge(pd.DataFrame(g['cross_validation']), on=['feature_set','model'], suffixes=('_provided','_generated'))\n    display(both[['feature_set','model','balanced_accuracy_mean_provided','balanced_accuracy_mean_generated','macro_f1_mean_provided','macro_f1_mean_generated']].round(3))\n    print('generated permutation test:', g['permutation_test'])",
    "md:## Conclusions (synthetic benchmark only)\n| Hypothesis | Provided dataset | Generated dataset | Verdict |\n|---|---|---|---|\n| **H1** keystroke features carry mood information | Best balanced accuracy 0.29–0.31 vs 0.25 chance; permutation p ≈ 0.02 | 0.54–0.56 with raw features alone | Supported, but the provided data's signal is very weak at window level |\n| **H2** personal baselines help | +0.01 to +0.04 (within fold noise) | **+0.19 to +0.24** (A → B) | Supported on coherent labels; inconclusive on the provided data |\n| **H3** temporal trends help | No gain (B → C) | No gain (B → C) | Not supported |\n| **H4** LLM gives grounded explanations | All sampled explanations reference real features and avoid diagnosis | same | Template path verified; live LLM run (Groq) pending |\n\nPersonalisation is the component that matters. Temporal *change* features do not help, even on the generated data where mood persists across sessions. A likely reason is that a change score is near zero whenever the mood stays the same, so it carries little information about which mood it is. The optional extended features (E) add nothing on either dataset.\n\nNone of these numbers describe real people: both datasets are synthetic and their labels are generator assumptions."
]

for name, cells in notebooks.items():
    nbf.write(nb(cells), HERE / f"{name}.ipynb")
    print("wrote", name)
