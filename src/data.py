#!/usr/bin/env python3
import argparse, os
from datetime import datetime, timedelta
import numpy as np, pandas as pd
 
def generate_data(out_dir: str, seed: int = 42, n_assets: int = 12, n_hours: int = 24 * 90):
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1)
    sensor_rows, outage_rows = [], []
    asset_ids = [f"GRID_ASSET_{i:03d}" for i in range(1, n_assets + 1)]
    asset_types = ["Transformer", "Feeder", "Substation"]
    regions = ["North", "South", "East"]

    for asset_id in asset_ids:
        asset_type = asset_types[rng.integers(0, len(asset_types))]
        region = regions[rng.integers(0, len(regions))]
        timestamps = [start + timedelta(hours=h) for h in range(n_hours)]
        base_temp = 55 + rng.normal(0, 3)
        base_vibration = 1.5 + rng.normal(0, 0.2)
        base_load = 60 + rng.normal(0, 5)
        base_voltage = 230 + rng.normal(0, 2)
        base_current = 95 + rng.normal(0, 6)
        anomaly_windows = []
        for _ in range(rng.integers(2, 4)):
            s = rng.integers(24, n_hours - 24)
            d = rng.integers(6, 18)
            anomaly_windows.append((s, min(n_hours - 1, s + d)))

        rows = []
        for idx, ts in enumerate(timestamps):
            daily = np.sin(2 * np.pi * (idx % 24) / 24)
            weekly = np.sin(2 * np.pi * (idx % (24 * 7)) / (24 * 7))
            temp = base_temp + 4 * daily + rng.normal(0, 1.2)
            vib = base_vibration + 0.12 * daily + rng.normal(0, 0.07)
            load = base_load + 10 * daily + 6 * weekly + rng.normal(0, 2.0)
            volt = base_voltage - 0.03 * load + rng.normal(0, 1.0)
            curr = base_current + 0.55 * load + rng.normal(0, 3.0)
            in_anomaly = any(a <= idx <= b for a, b in anomaly_windows)
            if in_anomaly:
                temp += rng.uniform(8, 15)
                vib += rng.uniform(0.6, 1.3)
                load += rng.uniform(7, 16)
                volt -= rng.uniform(3, 9)
                curr += rng.uniform(8, 20)
            rows.append([ts, asset_id, asset_type, region, temp, max(0.05, vib), max(5, load), max(160, volt), max(10, curr)])
        df = pd.DataFrame(rows, columns=["timestamp","asset_id","asset_type","region","temperature_c","vibration_mm_s","load_pct","voltage_v","current_a"])

        for col in ["temperature_c","vibration_mm_s","load_pct","voltage_v","current_a"]:
            miss_idx = rng.choice(df.index, size=max(1, int(0.02 * len(df))), replace=False)
            df.loc[miss_idx, col] = np.nan
        df["timestamp"] = df["timestamp"].astype(str)
        sensor_rows.append(df)

        roll = pd.DataFrame({
            "temperature": pd.Series(df["temperature_c"]).ffill().rolling(6, min_periods=1).mean(),
            "vibration": pd.Series(df["vibration_mm_s"]).ffill().rolling(6, min_periods=1).mean(),
            "load": pd.Series(df["load_pct"]).ffill().rolling(6, min_periods=1).mean(),
        })
        risk = 0.35*(roll["temperature"]>68).astype(int)+0.35*(roll["vibration"]>2.3).astype(int)+0.30*(roll["load"]>82).astype(int)
        fps = np.where(risk >= 0.65)[0]
        chosen = fps[::max(1, len(fps)//3 + 1)][:3]
        causes = ["Overheating","Insulation Failure","Mechanical Wear","Load Imbalance","Voltage Instability"]
        failure_types = ["Forced Outage","Partial Failure","Protective Trip"]
        for n, fp in enumerate(chosen, start=1):
            event_ts = start + timedelta(hours=int(fp))
            duration = int(rng.integers(1, 6))
            outage_rows.append({
                "event_id": f"{asset_id}_OUT_{n:03d}",
                "asset_id": asset_id,
                "event_start": event_ts.strftime("%Y/%m/%d %H:%M:%S"),
                "event_end": (event_ts + timedelta(hours=duration)).strftime("%Y/%m/%d %H:%M:%S"),
                "failure_type": failure_types[rng.integers(0, len(failure_types))],
                "root_cause": causes[rng.integers(0, len(causes))],
                "severity": ["Low","Medium","High"][rng.integers(0,3)],
                "region": region
            })

    sensor_df = pd.concat(sensor_rows, ignore_index=True)
    outage_df = pd.DataFrame(outage_rows)
    os.makedirs(out_dir, exist_ok=True)
    sensor_df.to_csv(os.path.join(out_dir, "utility_sensor_data.csv"), index=False)
    outage_df.to_csv(os.path.join(out_dir, "outage_logs.csv"), index=False)
    print("raw sensor:", sensor_df.shape, "outages:", outage_df.shape)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="data/raw")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_assets", type=int, default=12)
    p.add_argument("--n_hours", type=int, default=24*90)
    a = p.parse_args()
    generate_data(a.out_dir, a.seed, a.n_assets, a.n_hours)
