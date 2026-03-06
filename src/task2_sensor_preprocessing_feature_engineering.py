#!/usr/bin/env python3
import argparse, os
import numpy as np, pandas as pd

def fft_energy(x):
    arr = np.asarray(x, dtype=float)
    arr = np.nan_to_num(arr, nan=np.nanmean(arr) if not np.isnan(np.nanmean(arr)) else 0.0)
    spec = np.fft.rfft(arr - np.mean(arr))
    return float(np.sum(np.abs(spec) ** 2) / max(1, len(spec)))

def main(sensor_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(sensor_path, parse_dates=["timestamp"]).sort_values(["asset_id","timestamp"])
    num = ["temperature_c","vibration_mm_s","load_pct","voltage_v","current_a"]

    for col in num:
        df[col] = df.groupby("asset_id")[col].transform(lambda s: s.clip(s.quantile(0.01), s.quantile(0.99)))

    grp = df.groupby("asset_id", group_keys=False)
    for col in num:
        df[f"{col}_ma_6h"] = grp[col].transform(lambda s: s.rolling(6, min_periods=1).mean())
        df[f"{col}_std_12h"] = grp[col].transform(lambda s: s.rolling(12, min_periods=3).std().fillna(0))
        df[f"{col}_trend_12h"] = grp[col].transform(lambda s: s.diff(12).fillna(0))

    df["load_change_rate_1h"] = grp["load_pct"].transform(lambda s: s.diff().fillna(0))
    df["voltage_current_ratio"] = df["voltage_v"] / df["current_a"].replace(0, np.nan)
    df["power_proxy"] = df["voltage_v"] * df["current_a"]
    df["vibration_fft_energy_24h"] = grp["vibration_mm_s"].transform(lambda s: s.rolling(24, min_periods=8).apply(fft_energy, raw=False)).fillna(0)
    df["stress_score"] = (
        0.25 * (df["temperature_c_ma_6h"] / df["temperature_c_ma_6h"].max()) +
        0.25 * (df["vibration_mm_s_ma_6h"] / df["vibration_mm_s_ma_6h"].max()) +
        0.25 * (df["load_pct_ma_6h"] / df["load_pct_ma_6h"].max()) +
        0.25 * (1 - df["voltage_v_ma_6h"] / df["voltage_v_ma_6h"].max())
    )
    path = os.path.join(out_dir, "asset_features.csv")
    df.to_csv(path, index=False)
    print("wrote", path, df.shape)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sensor_path", default="data/interim/clean_sensor_data.csv")
    p.add_argument("--out_dir", default="data/processed")
    a = p.parse_args()
    main(a.sensor_path, a.out_dir)
