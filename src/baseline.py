import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.data import load_raw, clean, feature_matrix


def metrics(y_true, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {"precision": precision_score(y_true, pred), "recall": recall_score(y_true, pred),
            "f1": f1_score(y_true, pred), "pr_auc": average_precision_score(y_true, prob),
            "roc_auc": roc_auc_score(y_true, prob), "fpr": fp / (fp + tn)}


def main(path, out_dir):
    df, _ = clean(load_raw(path))
    X, y = feature_matrix(df), df["phishing"]

    # 70% train / 15% validation / 15% test, stratified. Test stays untouched for now.
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=config.SEED)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=config.SEED)

    models = {
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=1000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                                random_state=config.SEED),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_tr, y_tr)                      # scaler is fit on training data only
        prob = model.predict_proba(X_val)[:, 1]
        rows.append({"model": name, "eval_set": "validation",
                     "split": "random stratified 70/15/15", **metrics(y_val, prob)})

    res = pd.DataFrame(rows)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    res.to_csv(f"{out_dir}/baseline_validation.csv", index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    sample = "--sample" in sys.argv
    main(config.SAMPLE_CSV if sample else config.FULL_CSV,
         "results/sample" if sample else "results/full")