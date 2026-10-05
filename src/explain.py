"""Step 3: SHAP global explanations + top-3 churn drivers.
Run: python -m src.explain
"""
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.config import CATEGORICAL, MODEL_PATH, REPORTS
from src.features import clean_name, split_data


def original_feature(encoded: str) -> str:
    base = encoded.split("__", 1)[-1]
    for col in CATEGORICAL:
        if base.startswith(col + "_"):
            return col
    return base


def main():
    pipe = joblib.load(MODEL_PATH)
    _, X_te, _, _ = split_data()
    pre, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    names = list(pre.get_feature_names_out())
    X = pd.DataFrame(pre.transform(X_te), columns=names)

    sv = shap.TreeExplainer(clf)(X)
    sv.feature_names = [clean_name(n) for n in names]

    plt.figure(); shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.savefig(REPORTS / "shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()
    plt.figure(); shap.plots.bar(sv, max_display=15, show=False)
    plt.savefig(REPORTS / "shap_bar.png", dpi=150, bbox_inches="tight"); plt.close()

    # Roll one-hot columns back up to the original business feature
    imp = pd.Series(abs(sv.values).mean(0), index=names)
    by_feature = imp.groupby(original_feature).sum().sort_values(ascending=False)
    print("Top churn drivers (mean |SHAP|, one-hot columns aggregated):")
    print(by_feature.head(10).round(4).to_string())

    (REPORTS / "top_drivers.json").write_text(json.dumps(
        {"top3": by_feature.head(3).round(4).to_dict(),
         "all": by_feature.round(4).to_dict()}, indent=2))


if __name__ == "__main__":
    main()
