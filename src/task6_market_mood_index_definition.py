#!/usr/bin/env python3
import argparse, json, os, joblib, numpy as np, pandas as pd

DROP_COLS = ["timestamp","linked_event_id","label","label_code"]

def confidence_from_proba(proba):
    p = np.clip(np.array(proba, dtype=float), 1e-12, 1.0)
    entropy = -np.sum(p * np.log(p))
    return float(np.clip(1 - entropy / np.log(len(p)), 0, 1))

def get_proba(model_bundle, X):
    if isinstance(model_bundle, dict):
        X_pre = model_bundle["preprocessor"].transform(X)
        return model_bundle["model"].predict_proba(X_pre)
    return model_bundle.predict_proba(X)

def regime_logic(df):
    current, enter_bad, exit_good = "Normal", 0, 0
    regimes = []
    for _, r in df.iterrows():
        mood = r["asset_health_index_ewma"]
        conf = r["confidence"]
        p_risk = r["p_prefailure"] + r["p_failure"]
        if r["p_failure"] >= 0.55 and conf >= 0.60:
            current = "Critical"; enter_bad = 0; exit_good = 0; regimes.append(current); continue
        if current == "Normal":
            if mood <= 65:
                enter_bad += 1
                if enter_bad >= 2:
                    current = "Warning"; enter_bad = 0
            else:
                enter_bad = 0
        elif current == "Warning":
            if mood <= 40:
                enter_bad += 1
                if enter_bad >= 2:
                    current = "Critical"; enter_bad = 0; exit_good = 0
            elif mood >= 72:
                exit_good += 1
                if exit_good >= 3:
                    current = "Normal"; exit_good = 0
            else:
                enter_bad, exit_good = 0, 0
            if p_risk >= 0.65 and conf >= 0.70:
                current = "Critical"
        elif current == "Critical":
            if mood >= 48:
                exit_good += 1
                if exit_good >= 3:
                    current = "Warning"; exit_good = 0
            else:
                exit_good = 0
        regimes.append(current)
    df["asset_health_regime"] = regimes
    return df

def main(data_path: str, model_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(data_path, parse_dates=["timestamp"]).sort_values(["asset_id","timestamp"]).reset_index(drop=True)
    X = df[[c for c in df.columns if c not in DROP_COLS]]
    model_bundle = joblib.load(model_path)
    probs = get_proba(model_bundle, X)
    out = df[["timestamp","asset_id","asset_type","region","label","label_code"]].copy()
    out["p_normal"], out["p_prefailure"], out["p_failure"] = probs[:,0], probs[:,1], probs[:,2]
    out["confidence"] = [confidence_from_proba(p) for p in probs]
    out["risk_probability"] = out["p_prefailure"] + out["p_failure"]
    out["asset_health_index_raw"] = 100 * (1 - out["risk_probability"])
    out["asset_health_index"] = out["asset_health_index_raw"] * (0.5 + 0.5 * out["confidence"])
    out["asset_health_index_ewma"] = out.groupby("asset_id")["asset_health_index"].transform(lambda s: s.ewm(alpha=0.25, adjust=False).mean())
    out = out.groupby("asset_id", group_keys=False).apply(regime_logic)
    out.to_csv(os.path.join(out_dir, "asset_health_regime_output.csv"), index=False)
    with open(os.path.join(out_dir, "task6_regime_rules.json"), "w") as f:
        json.dump({
            "Normal": "Index generally above 72 with sustained low failure risk.",
            "Warning": "Index <= 65 for 2 periods or rising pre-failure probability.",
            "Critical": "Index <= 40 for 2 periods or p_failure >= 0.55 with confidence >= 0.60."
        }, f, indent=2)
    print(out.head(8).to_string(index=False))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="data/processed/labeled_asset_features.csv")
    p.add_argument("--model_path", default="artifacts/best_optimized_model.joblib")
    p.add_argument("--out_dir", default="artifacts")
    a = p.parse_args()
    main(a.data_path, a.model_path, a.out_dir)
