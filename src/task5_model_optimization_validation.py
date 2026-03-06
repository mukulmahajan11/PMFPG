#!/usr/bin/env python3
import argparse, json, os, joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier

DROP_COLS = ["timestamp","linked_event_id","label","label_code"]

def pack_metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    risky_true = np.isin(y_true, [1,2]).astype(int)
    risky_pred = np.isin(y_pred, [1,2]).astype(int)
    tp = int(((risky_true==1)&(risky_pred==1)).sum()); tn = int(((risky_true==0)&(risky_pred==0)).sum())
    fp = int(((risky_true==0)&(risky_pred==1)).sum()); fn = int(((risky_true==1)&(risky_pred==0)).sum())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(p),
        "recall_weighted": float(r),
        "f1_weighted": float(f1),
        "false_positive_rate_risky_vs_normal": float(fp / max(1, fp + tn)),
        "false_negative_rate_risky_vs_normal": float(fn / max(1, fn + tp)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

def main(data_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    X = df[[c for c in df.columns if c not in DROP_COLS]]
    y = df["label_code"]
    num_cols = [c for c in X.columns if X[c].dtype != "object"]
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    rf_pipe = Pipeline([("preprocessor", pre), ("model", RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"))])
    search = RandomizedSearchCV(
        rf_pipe,
        param_distributions={
            "model__n_estimators":[150, 220],
            "model__max_depth":[10, 14, None],
            "model__min_samples_split":[2, 4, 8],
            "model__min_samples_leaf":[1, 2, 4]
        },
        n_iter=6,
        scoring="f1_weighted",
        cv=cv,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    tuned_rf = search.best_estimator_
    tuned_pred = tuned_rf.predict(X_test)

    pre_fit = pre.fit(X_train)
    X_train_pre, X_test_pre = pre_fit.transform(X_train), pre_fit.transform(X_test)
    ensemble = VotingClassifier(estimators=[
        ("rf", RandomForestClassifier(n_estimators=220, max_depth=search.best_params_.get("model__max_depth"), random_state=42, n_jobs=-1, class_weight="balanced")),
        ("et", ExtraTreesClassifier(n_estimators=220, max_depth=14, random_state=42, n_jobs=-1, class_weight="balanced"))
    ], voting="soft")
    ensemble.fit(X_train_pre, y_train)
    ensemble_pred = ensemble.predict(X_test_pre)

    rf_metrics = pack_metrics(y_test, tuned_pred)
    ens_metrics = pack_metrics(y_test, ensemble_pred)
    cv_scores = cross_val_score(tuned_rf, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)
    best_final = "ensemble" if ens_metrics["f1_weighted"] >= rf_metrics["f1_weighted"] else "tuned_random_forest"

    if best_final == "ensemble":
        joblib.dump({"preprocessor": pre_fit, "model": ensemble}, os.path.join(out_dir, "best_optimized_model.joblib"))
    else:
        joblib.dump(tuned_rf, os.path.join(out_dir, "best_optimized_model.joblib"))

    results = {
        "best_final_model": best_final,
        "tuned_random_forest": {"best_params": search.best_params_, "metrics": rf_metrics, "cross_val_f1_mean": float(cv_scores.mean()), "cross_val_f1_std": float(cv_scores.std()), "classification_report": classification_report(y_test, tuned_pred, output_dict=True, zero_division=0)},
        "ensemble": {"metrics": ens_metrics, "classification_report": classification_report(y_test, ensemble_pred, output_dict=True, zero_division=0)},
        "limitations": [
            "Synthetic data usually yields cleaner boundaries than real grid telemetry.",
            "Operational drift, seasonality, and sensor recalibration can reduce production performance.",
            "The 24-hour pre-failure proxy is useful for classification but is not a true survival-analysis time-to-failure model."
        ]
    }
    with open(os.path.join(out_dir, "optimized_model_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({"best_final_model": best_final, "tuned_rf_f1": rf_metrics["f1_weighted"], "ensemble_f1": ens_metrics["f1_weighted"]}, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="data/processed/labeled_asset_features.csv")
    p.add_argument("--out_dir", default="artifacts")
    a = p.parse_args()
    main(a.data_path, a.out_dir)
