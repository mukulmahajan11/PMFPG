#!/usr/bin/env python3
import argparse, json, os
import pandas as pd

def main(sensor_path: str, outage_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    sensor = pd.read_csv(sensor_path)
    outage = pd.read_csv(outage_path)
    sensor.columns = [c.strip().lower() for c in sensor.columns]
    outage.columns = [c.strip().lower() for c in outage.columns]

    sensor["timestamp"] = pd.to_datetime(sensor["timestamp"], errors="coerce")
    outage["event_start"] = pd.to_datetime(outage["event_start"], errors="coerce")
    outage["event_end"] = pd.to_datetime(outage["event_end"], errors="coerce")

    num_cols = ["temperature_c","vibration_mm_s","load_pct","voltage_v","current_a"]
    for c in num_cols:
        sensor[c] = pd.to_numeric(sensor[c], errors="coerce")

    sensor = sensor.drop_duplicates(subset=["asset_id","timestamp"]).sort_values(["asset_id","timestamp"])
    outage = outage.drop_duplicates(subset=["event_id"]).sort_values(["asset_id","event_start"])

    sensor[num_cols] = sensor.groupby("asset_id")[num_cols].transform(lambda g: g.ffill().bfill())
    for c in num_cols:
        sensor[c] = sensor[c].fillna(sensor[c].median())

    report = {
        "sensor_rows": int(len(sensor)),
        "outage_rows": int(len(outage)),
        "sensor_missing_after_clean": sensor[num_cols].isna().sum().to_dict(),
        "null_timestamps": int(sensor["timestamp"].isna().sum()),
        "invalid_outage_durations": int((outage["event_end"] < outage["event_start"]).fillna(False).sum()),
    }
    sensor.to_csv(os.path.join(out_dir, "clean_sensor_data.csv"), index=False)
    outage.to_csv(os.path.join(out_dir, "clean_outage_logs.csv"), index=False)
    with open(os.path.join(out_dir, "task1_validation_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sensor_path", default="data/raw/utility_sensor_data.csv")
    p.add_argument("--outage_path", default="data/raw/outage_logs.csv")
    p.add_argument("--out_dir", default="data/interim")
    a = p.parse_args()
    main(a.sensor_path, a.outage_path, a.out_dir)
