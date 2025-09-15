import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
from data_loader import load_catalog, basic_cleaning, add_time_features, add_spatiotemporal_lags

DATA_PATH = Path('/mnt/data/katalog_gempa_v2.tsv')

def main():
    print('Loading catalog...')
    df = load_catalog(DATA_PATH)
    df = basic_cleaning(df)
    df = add_time_features(df)
    df = add_spatiotemporal_lags(df)
    print(f'Rows after cleaning: {len(df)}')
    # handle alternate column names
    mag_col = 'mag' if 'mag' in df.columns else ('magnitude' if 'magnitude' in df.columns else None)
    lat_col = 'lat' if 'lat' in df.columns else ('latitude' if 'latitude' in df.columns else None)
    lon_col = 'lon' if 'lon' in df.columns else ('longitude' if 'longitude' in df.columns else None)
    if mag_col is None or lat_col is None or lon_col is None:
        print('Required columns not found (mag/magnitude, lat/latitude, lon/longitude). Exiting.')
        return
    # create consistent columns
    df['mag_target'] = pd.to_numeric(df[mag_col], errors='coerce')
    df['lat'] = pd.to_numeric(df[lat_col], errors='coerce')
    df['lon'] = pd.to_numeric(df[lon_col], errors='coerce')
    features = ['lat','lon','depth','hour_sin','hour_cos','doy_sin','doy_cos','prev_mag','time_since_prev_sec']
    # ensure features exist
    for f in features:
        if f not in df.columns:
            df[f] = pd.NA
    df_features = df[features + ['mag_target']].copy()
    df_features = df_features[~df_features['mag_target'].isna()].reset_index(drop=True)
    if len(df_features) < 10:
        print('Not enough rows for training demo; exiting.')
        return
    split_idx = int(0.8 * len(df_features))
    X_train, X_test = df_features[features].values[:split_idx], df_features[features].values[split_idx:]
    y_train, y_test = df_features['mag_target'].values[:split_idx], df_features['mag_target'].values[split_idx:]
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1))
    ])
    print('Training RandomForest (demo, 50 trees)...')
    pipe.fit(X_train, y_train)
    print('Predicting...')
    preds = pipe.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    print(f'RMSE: {rmse:.3f}, MAE: {mae:.3f}')
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    joblib.dump(pipe, model_dir/'rf_baseline.joblib')
    report = f"""# Demo Report\n\nRows used: {len(df_features)}\nTrain/test split index: {split_idx}\nRMSE: {rmse:.3f}\nMAE: {mae:.3f}\n\nFeatures used: {features}\n\nNotes: This is a baseline demonstration. For portfolio, extend to XGBoost, LSTM/Transformer time-series, and domain-specific QA such as magnitude-of-completeness and declustering."""
    Path('report.md').write_text(report)
    print('Saved model to models/rf_baseline.joblib and report.md')

if __name__ == '__main__':
    main()
