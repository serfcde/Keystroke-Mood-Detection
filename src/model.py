"""Training, user-independent evaluation and ablation (PRD §12-14, §19-20).

Run the whole pipeline with:  python -m src.model
"""
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, LeaveOneGroupOut, permutation_test_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from . import baseline, cleaning, config, features

Z = [baseline.z_col(f) for f in config.CORE_FEATURES]
CHANGE = [baseline.change_col(f) for f in config.CORE_FEATURES]
FEATURE_SETS = {
    "A_raw": config.CORE_FEATURES,
    "B_+baseline": config.CORE_FEATURES + Z,
    "C_+baseline+temporal": config.CORE_FEATURES + Z + CHANGE,
    # PRD §9 optional extensions on top of C ("D" in the PRD ablation is the LLM layer)
    "E_C+extended": config.CORE_FEATURES + Z + CHANGE + config.EXTENDED_FEATURES,
}
FINAL_SET = "C_+baseline+temporal"
MOOD_TO_ID = {m: i for i, m in enumerate(config.MOODS)}


class BalancedXGB(XGBClassifier):
    """XGBoost that reweights classes like RandomForest(class_weight='balanced')."""

    def fit(self, X, y, **kw):
        return super().fit(X, y, sample_weight=compute_sample_weight("balanced", y), **kw)


def make_models():
    return {
        "majority_class": DummyClassifier(strategy="most_frequent"),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                                random_state=config.RANDOM_STATE, n_jobs=-1),
        "xgboost": BalancedXGB(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, eval_metric="mlogloss",
                               random_state=config.RANDOM_STATE, n_jobs=-1),
        "svm": make_pipeline(StandardScaler(), SVC(C=1.0, class_weight="balanced", random_state=config.RANDOM_STATE)),
    }


def build_dataset(save=True) -> pd.DataFrame:
    events, _, _ = cleaning.run(save=save)
    windows = features.build_windows(events)
    windows = baseline.add_baseline_deviation(windows)
    windows = baseline.add_temporal(windows)
    if save:
        windows.to_csv(config.WINDOW_PATH, index=False)
    return windows


def _scores(y, pred):
    return {"accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", labels=list(range(len(config.MOODS))), zero_division=0)}


def cross_validate(windows: pd.DataFrame, n_splits=5) -> pd.DataFrame:
    """GroupKFold by user: every fold tests on users never seen in training (40 train / 10 test users)."""
    y = windows["mood_state"].map(MOOD_TO_ID).to_numpy()
    groups = windows["user_id"].to_numpy()
    rows = []
    for set_name, cols in FEATURE_SETS.items():
        X = windows[cols].to_numpy()
        for model_name, model in make_models().items():
            if model_name == "majority_class" and set_name != "A_raw":
                continue
            fold_scores = []
            for tr, te in GroupKFold(n_splits).split(X, y, groups):
                m = clone(model).fit(X[tr], y[tr])
                fold_scores.append(_scores(y[te], m.predict(X[te])))
            fs = pd.DataFrame(fold_scores)
            row = {"feature_set": "-" if model_name == "majority_class" else set_name, "model": model_name}
            for metric in fs:
                row[f"{metric}_mean"] = fs[metric].mean()
                row[f"{metric}_std"] = fs[metric].std()
            rows.append(row)
    return pd.DataFrame(rows)


def leave_one_user_out(windows, set_name=FINAL_SET) -> pd.DataFrame:
    """PRD §14 advanced evaluation: train on 49 users, test on the 50th, repeat for every user."""
    y = windows["mood_state"].map(MOOD_TO_ID).to_numpy()
    X = windows[FEATURE_SETS[set_name]].to_numpy()
    groups = windows["user_id"].to_numpy()
    rows = []
    for name, model in make_models().items():
        pred = np.empty_like(y)
        for tr, te in LeaveOneGroupOut().split(X, y, groups):
            pred[te] = clone(model).fit(X[tr], y[tr]).predict(X[te])
        rows.append({"model": name, "feature_set": set_name, **_scores(y, pred)})
    return pd.DataFrame(rows)


def shap_importance(model, X: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per feature, averaged over classes."""
    import shap
    values = shap.TreeExplainer(model).shap_values(X)
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[0] == len(config.MOODS):   # (classes, rows, features)
        values = values.transpose(1, 2, 0)
    return pd.Series(np.abs(values).mean(axis=(0, 2)), index=X.columns).sort_values(ascending=False)


def permutation_test(windows, set_name=FINAL_SET, n_permutations=200):
    """Is the model better than chance? Shuffles labels and re-runs grouped CV."""
    y = windows["mood_state"].map(MOOD_TO_ID).to_numpy()
    X = windows[FEATURE_SETS[set_name]].to_numpy()
    score, perm_scores, p = permutation_test_score(
        make_models()["xgboost"], X, y, groups=windows["user_id"], cv=GroupKFold(5),
        scoring="balanced_accuracy", n_permutations=n_permutations, random_state=config.RANDOM_STATE, n_jobs=-1)
    return {"feature_set": set_name, "metric": "balanced_accuracy", "score": float(score),
            "permuted_mean": float(perm_scores.mean()), "permuted_95th": float(np.percentile(perm_scores, 95)),
            "p_value": float(p), "n_permutations": n_permutations}


def regression_cv(windows, set_name=FINAL_SET):
    """PRD §13 option B: predict the window's mean valence/arousal."""
    X = windows[FEATURE_SETS[set_name]].to_numpy()
    groups = windows["user_id"].to_numpy()
    out = {}
    for target in ["valence", "arousal"]:
        y = windows[target].to_numpy()
        for name, model in {"mean_predictor": DummyRegressor(),
                            "xgboost": XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                    subsample=0.8, random_state=config.RANDOM_STATE)}.items():
            pred = np.empty_like(y)
            for tr, te in GroupKFold(5).split(X, y, groups):
                pred[te] = clone(model).fit(X[tr], y[tr]).predict(X[te])
            out[f"{target}/{name}"] = {"mae": mean_absolute_error(y, pred),
                                       "rmse": float(np.sqrt(mean_squared_error(y, pred))),
                                       "r2": r2_score(y, pred)}
    return out


def train_holdout(windows, set_name=FINAL_SET):
    """40 train users / 10 test users; saves the final models and the test split for the dashboard."""
    cols = FEATURE_SETS[set_name]
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_STATE)
    tr, te = next(split.split(windows, groups=windows["user_id"]))
    train, test = windows.iloc[tr], windows.iloc[te]
    y_tr = train["mood_state"].map(MOOD_TO_ID)
    y_te = test["mood_state"].map(MOOD_TO_ID)

    results = {"train_users": sorted(train["user_id"].unique()), "test_users": sorted(test["user_id"].unique())}
    models = make_models()
    for name in ["random_forest", "xgboost", "svm"]:
        m = models[name].fit(train[cols], y_tr)
        pred = m.predict(test[cols])
        results[name] = {**_scores(y_te, pred),
                         "report": classification_report(y_te, pred, labels=range(len(config.MOODS)),
                                                         target_names=config.MOODS, zero_division=0, output_dict=True),
                         "confusion_matrix": confusion_matrix(y_te, pred, labels=range(len(config.MOODS))).tolist()}
        with open(config.MODEL_DIR / f"{name}_model.pkl", "wb") as fh:
            pickle.dump({"model": m, "features": cols, "classes": config.MOODS,
                         "test_users": results["test_users"]}, fh)
        models[name] = m
    importance = pd.Series(models["xgboost"].feature_importances_, index=cols).sort_values(ascending=False)
    results["xgboost_importance"] = importance.round(4).to_dict()
    results["xgboost_shap"] = shap_importance(models["xgboost"], test[cols].astype(float)).round(4).to_dict()
    results["majority_class_on_test"] = float((test["mood_state"] == train["mood_state"].mode()[0]).mean())
    return results


def plot_results(cv: pd.DataFrame, holdout: dict):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)

    cm = np.array(holdout["xgboost"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=config.MOODS, yticklabels=config.MOODS, ax=ax)
    ax.set(xlabel="Predicted", ylabel="True (window majority mood)", title="XGBoost, 10 unseen users")
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "confusion_matrix.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    pd.Series(holdout["xgboost_importance"]).head(12)[::-1].plot.barh(ax=axes[0], color="#4C72B0")
    axes[0].set(title="XGBoost built-in importance", xlabel="gain share")
    pd.Series(holdout["xgboost_shap"]).head(12)[::-1].plot.barh(ax=axes[1], color="#DD8452")
    axes[1].set(title="Mean |SHAP| on test users", xlabel="mean |SHAP| (log-odds)")
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "feature_importance.png", dpi=150); plt.close(fig)

    models = cv[cv["model"] != "majority_class"]
    sets, names = list(FEATURE_SETS), list(models["model"].unique())
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, name in enumerate(names):
        r = models[models["model"] == name].set_index("feature_set").loc[sets]
        ax.bar(np.arange(len(sets)) + i * width, r["balanced_accuracy_mean"], width,
               yerr=r["balanced_accuracy_std"], capsize=3, label=name)
    ax.axhline(0.25, ls="--", c="k", lw=1, label="chance (0.25)")
    ax.set_xticks(np.arange(len(sets)) + width * (len(names) - 1) / 2, sets)
    ax.set(title="Ablation: balanced accuracy, 5-fold user-grouped CV (mean ± std)", ylabel="balanced accuracy")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(config.FIG_DIR / "ablation.png", dpi=150); plt.close(fig)


def main():
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    windows = build_dataset()
    print(f"{len(windows)} windows from {windows['user_id'].nunique()} users")

    cv = cross_validate(windows)
    perm = permutation_test(windows)
    loso = leave_one_user_out(windows)
    reg = regression_cv(windows)
    holdout = train_holdout(windows)
    plot_results(cv, holdout)

    results = {"n_windows": len(windows), "window_s": config.WINDOW_S,
               "label_purity_mean": float(windows["label_purity"].mean()),
               "window_label_distribution": windows["mood_state"].value_counts(normalize=True).round(4).to_dict(),
               "cross_validation": cv.round(4).to_dict(orient="records"),
               "permutation_test": perm, "leave_one_user_out": loso.round(4).to_dict(orient="records"),
               "regression": reg, "holdout": holdout}
    (config.REPORT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=str))

    pd.set_option("display.width", 200)
    print(cv.round(3).to_string(index=False))
    print("permutation test:", perm)
    print(loso.round(3).to_string(index=False))
    print("regression:", json.dumps(reg, indent=1))
    print("holdout xgboost:", {k: round(v, 3) for k, v in holdout["xgboost"].items() if isinstance(v, float)})


if __name__ == "__main__":
    # import via the package so pickled models reference src.model.BalancedXGB, not __main__
    from src.model import main as _main
    _main()
