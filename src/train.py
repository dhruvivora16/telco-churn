"""Step 2: baseline logistic regression vs tuned XGBoost.
Run: python -m src.train
"""
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import MODEL_PATH, REPORTS, SEED
from src.features import build_preprocessor, split_data

THRESHOLD = 0.5  # same decision threshold for both models -> fair comparison


def evaluate(name, pipe, X_te, y_te):
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    m = {
        "model": name,
        "recall": recall_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, proba),
        "pr_auc": average_precision_score(y_te, proba),
    }
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
    return m, proba, pred


def main():
    REPORTS.mkdir(exist_ok=True)
    X_tr, X_te, y_tr, y_te = split_data()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    # ---- Baseline: logistic regression ----
    baseline = Pipeline([
        ("prep", build_preprocessor(scale_numeric=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ]).fit(X_tr, y_tr)
    base_m, base_p, _ = evaluate("Logistic Regression (baseline)", baseline, X_te, y_te)

    # ---- Challenger: XGBoost with randomized search ----
    spw = (y_tr == 0).sum() / (y_tr == 1).sum()  # handle class imbalance
    xgb_pipe = Pipeline([
        ("prep", build_preprocessor(scale_numeric=False)),
        ("clf", XGBClassifier(eval_metric="logloss", scale_pos_weight=spw,
                              random_state=SEED, n_jobs=-1)),
    ])
    search = RandomizedSearchCV(
        xgb_pipe,
        {"clf__n_estimators": randint(150, 500),
         "clf__max_depth": randint(2, 6),
         "clf__learning_rate": uniform(0.01, 0.15),
         "clf__subsample": uniform(0.7, 0.3),
         "clf__colsample_bytree": uniform(0.6, 0.4),
         "clf__min_child_weight": randint(1, 10)},
        n_iter=25, scoring="roc_auc", cv=cv, random_state=SEED, n_jobs=-1,
    ).fit(X_tr, y_tr)
    best = search.best_estimator_
    print("Best params:", search.best_params_, f"(CV AUC {search.best_score_:.3f})")
    xgb_m, xgb_p, xgb_pred = evaluate("XGBoost (tuned)", best, X_te, y_te)

    # ---- Save artifacts ----
    joblib.dump(best, MODEL_PATH)
    results = {"threshold": THRESHOLD, "test_size": int(len(y_te)),
               "best_params": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v))
                               for k, v in search.best_params_.items()},
               "baseline": base_m, "xgboost": xgb_m}
    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2))

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_te, base_p, name="Logistic Regression", ax=ax)
    RocCurveDisplay.from_predictions(y_te, xgb_p, name="XGBoost", ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=.4)
    ax.set_title("ROC curve: baseline vs XGBoost (test set)")
    fig.tight_layout(); fig.savefig(REPORTS / "roc_curve.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_te, xgb_pred, ax=ax, cmap="Blues")
    ax.set_title("XGBoost confusion matrix"); fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=150); plt.close(fig)
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
