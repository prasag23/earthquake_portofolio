import pandas as pd
import numpy as np
from pathlib import Path
import csv

def load_catalog(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        quoting=csv.QUOTE_NONE,
        engine="python",
        on_bad_lines="skip"
    )
    # Map relevant columns
    colmap = {
        "datetime": "time",
        "latitude": "lat",
        "longitude": "lon",
        "magnitude": "mag",
        "depth": "depth"
    }
    for old, new in colmap.items():
        if old in df.columns:
            df[new] = df[old]
    return df

def basic_cleaning(df):
    df = df.copy()
    if "mag" in df.columns:
        df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
        df = df[~df["mag"].isna()].reset_index(drop=True)
    for col in ["lat","lon","depth"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df[~df["time"].isna()]
    return df

def add_time_features(df):
    df = df.copy()
    if "time" not in df.columns:
        return df
    df["hour"] = df["time"].dt.hour.fillna(0).astype(int)
    df["dayofyear"] = df["time"].dt.dayofyear.fillna(0).astype(int)
    df["year"] = df["time"].dt.year.fillna(0).astype(int)
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["doy_sin"] = np.sin(2*np.pi*df["dayofyear"]/365.25)
    df["doy_cos"] = np.cos(2*np.pi*df["dayofyear"]/365.25)
    return df

def add_spatiotemporal_lags(df, grid_size_deg=0.5):
    df = df.copy()
    if "lat" not in df.columns or "lon" not in df.columns:
        return df
    df["grid_lat"] = (df["lat"] / grid_size_deg).round(0) * grid_size_deg
    df["grid_lon"] = (df["lon"] / grid_size_deg).round(0) * grid_size_deg
    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    df["prev_mag"] = np.nan
    df["time_since_prev_sec"] = np.nan
    last_seen = {}
    for i, row in df.iterrows():
        key = (row.get("grid_lat"), row.get("grid_lon"))
        if key in last_seen:
            prev_idx, prev_time = last_seen[key]
            df.at[i, "prev_mag"] = df.at[prev_idx, "mag"]
            if "time" in df.columns and pd.notna(prev_time) and pd.notna(row.get("time")):
                df.at[i, "time_since_prev_sec"] = (row["time"] - prev_time).total_seconds()
        last_seen[key] = (i, row.get("time"))
    return df
