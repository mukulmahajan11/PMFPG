#!/usr/bin/env python3
import argparse, json, os
from datetime import timedelta
import pandas as pd

def main(features_path: str, outage_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    feats = pd.read_csv(features_path, parse_dates=["timestamp"])
    outages = pd.read_csv(outage_path, parse_dates=["event_start","event_end"])

    feats["label"] = "normal_operation"
    feats["label_code"] = 0
    feats["linked_event_id"] = None

    for _, e in outages.iterrows():
        pre = (
            (feats["asset_id"] == e["asset_id"]) &
            (feats["timestamp"] >= e["event_start"] - timedelta(hours=24)) &
            (feats["timestamp"] < e["event_start"])
        )
        fail = (
            (feats["asset_id"] == e["asset_id"]) &
            (feats["timestamp"] >= e["event_start"]) &
            (feats["timestamp"] <= e["event_end"])
        )
        feats.loc[pre, ["label","label_code","linked_event_id"]] = ["pre_failure_anomaly", 1, e["event_id"]]
        feats.loc[fail, ["label","label_code","linked_event_id"]] = ["failure_window", 2, e["event_id"]]

    outage_summary = {
        "outage_count": int(len(outages)),
        "failure_type_distribution": outages["failure_type"].value_counts().to_dict(),
        "root_cause_distribution": outages["root_cause"].value_counts().to_dict(),
        "avg_duration_hours": float(((outages["event_end"] - outages["event_start"]).dt.total_seconds() / 3600).mean() if len(outages) else 0.0),
        "label_distribution": feats["label"].value_counts().to_dict(),
    }

    feats.to_csv(os.path.join(out_dir, "labeled_asset_features.csv"), index=False)
    with open(os.path.join(out_dir, "task3_outage_label_report.json"), "w") as f:
        json.dump(outage_summary, f, indent=2, default=str)
    print(json.dumps(outage_summary, indent=2, default=str))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features_path", default="data/processed/asset_features.csv")
    p.add_argument("--outage_path", default="data/interim/clean_outage_logs.csv")
    p.add_argument("--out_dir", default="data/processed")
    a = p.parse_args()
    main(a.features_path, a.outage_path, a.out_dir)
