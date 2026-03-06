#!/usr/bin/env python3
import argparse, json, os, joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

DROP_COLS = ["timestamp","linked_event_id","label","label_code"]

def metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision_weighted": float(p), "recall_weighted": float(r), "f1_weighted": float(f1)}

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
    models = {
        "random_forest": (RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"), {"model__n_estimators":[120], "model__max_depth":[10, 14]}),
        "extra_trees": (ExtraTreesClassifier(random_state=42, n_jobs=-1, class_weight="balanced"), {"model__n_estimators":[120], "model__max_depth":[10, 14]}),
        "logistic_regression": (LogisticRegression(max_iter=500, class_weight="balanced"), {"model__C":[0.5, 1.0]})
    }
    results, best_name, best_score, best_estimator = {}, None, -1, None
    for name, (model, grid) in models.items():
        pipe = Pipeline([("preprocessor", pre), ("model", model)])
        search = GridSearchCV(pipe, grid, scoring="f1_weighted", cv=3, n_jobs=-1)
        search.fit(X_train, y_train)
        pred = search.predict(X_test)
        m = metrics(y_test, pred)
        results[name] = {"best_params": search.best_params_, "metrics": m, "confusion_matrix": confusion_matrix(y_test, pred).tolist(), "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0)}
        if m["f1_weighted"] > best_score:
            best_name, best_score, best_estimator = name, m["f1_weighted"], search.best_estimator_
    joblib.dump(best_estimator, os.path.join(out_dir, "best_initial_model.joblib"))
    with open(os.path.join(out_dir, "initial_model_metrics.json"), "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)
    print(json.dumps({"best_model": best_name, "f1_weighted": best_score}, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="data/processed/labeled_asset_features.csv")
    p.add_argument("--out_dir", default="artifacts")
    a = p.parse_args()
    main(a.data_path, a.out_dir)
