import csv
import pandas as pd
import numpy as np
from pathlib import Path

def load_catalog(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")
    # read header and rows using csv module (robust to weird pandas parse settings)
    # autodetect delimiter: check first line
    sample = p.read_text(errors='replace').splitlines()
    if len(sample) == 0:
        return pd.DataFrame()
    first = sample[0]
    delim = '\t' if '\t' in first else ','
    with p.open('r', errors='replace') as f:
        reader = csv.DictReader(f, delimiter=delim)
        rows = list(reader)
    if len(rows) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df

# keep the rest of functions as before (basic_cleaning, add_time_features, add_spatiotemporal_lags)
def basic_cleaning(df):
    df = df.copy()
    if 'mag' in df.columns:
        df = df[~df['mag'].isna()].reset_index(drop=True)
    df.columns = [c.strip() for c in df.columns]
    for col in ['lat','lon','depth','mag','latitude','longitude','magnitude']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'ot' in df.columns:
        try:
            df['ot'] = pd.to_datetime(df['ot'], errors='coerce')
        except Exception:
            pass
    if 'tgl' in df.columns:
        try:
            df['tgl'] = pd.to_datetime(df['tgl'], errors='coerce')
        except Exception:
            pass
    if 'datetime' in df.columns:
        try:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        except Exception:
            pass
    if 'ot' in df.columns and df['ot'].notna().any():
        df['time'] = pd.to_datetime(df['ot'], errors='coerce')
    elif 'tgl' in df.columns and df['tgl'].notna().any():
        df['time'] = pd.to_datetime(df['tgl'], errors='coerce')
    elif 'datetime' in df.columns and df['datetime'].notna().any():
        df['time'] = pd.to_datetime(df['datetime'], errors='coerce')
    else:
        for c in df.columns:
            if df[c].dtype == object:
                try:
                    parsed = pd.to_datetime(df[c], errors='coerce')
                    if parsed.notna().sum() > 0:
                        df['time'] = parsed
                        break
                except Exception:
                    continue
    if 'time' not in df.columns:
        df['time'] = pd.NaT
    df = df[~df['time'].isna()] if df['time'].notna().any() else df
    return df

def add_time_features(df):
    df = df.copy()
    if 'time' not in df.columns:
        return df
    df['hour'] = df['time'].dt.hour.fillna(0).astype(int)
    df['dayofyear'] = df['time'].dt.dayofyear.fillna(0).astype(int)
    df['year'] = df['time'].dt.year.fillna(0).astype(int)
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    df['doy_sin'] = np.sin(2*np.pi*df['dayofyear']/365.25)
    df['doy_cos'] = np.cos(2*np.pi*df['dayofyear']/365.25)
    return df

def add_spatiotemporal_lags(df, grid_size_deg=0.5):
    df = df.copy()
    lat_col = 'lat' if 'lat' in df.columns else ('latitude' if 'latitude' in df.columns else None)
    lon_col = 'lon' if 'lon' in df.columns else ('longitude' if 'longitude' in df.columns else None)
    if lat_col is None or lon_col is None:
        return df
    df['lat'] = pd.to_numeric(df[lat_col], errors='coerce')
    df['lon'] = pd.to_numeric(df[lon_col], errors='coerce')
    df['grid_lat'] = (df['lat'] / grid_size_deg).round(0) * grid_size_deg
    df['grid_lon'] = (df['lon'] / grid_size_deg).round(0) * grid_size_deg
    if 'time' in df.columns:
        df = df.sort_values('time').reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    df['prev_mag'] = np.nan
    df['time_since_prev_sec'] = np.nan
    last_seen = {}
    mag_col = 'mag' if 'mag' in df.columns else ('magnitude' if 'magnitude' in df.columns else None)
    for i, row in df.iterrows():
        key = (row.get('grid_lat'), row.get('grid_lon'))
        if key in last_seen:
            prev_idx, prev_time = last_seen[key]
            if mag_col is not None:
                df.at[i, 'prev_mag'] = df.at[prev_idx, mag_col]
            if 'time' in df.columns and pd.notna(prev_time) and pd.notna(row.get('time')):
                df.at[i, 'time_since_prev_sec'] = (row['time'] - prev_time).total_seconds()
        last_seen[key] = (i, row.get('time'))
    return df
