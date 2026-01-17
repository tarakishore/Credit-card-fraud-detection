# utils.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Tuple, Dict

RANDOM_STATE = 42

def preprocess_df(df: pd.DataFrame, scaler: StandardScaler = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standard preprocessing used in training:
    - scale Amount -> Amount_scaled
    - drop Time if exists
    Returns processed df and scaler (fitted if provided None)
    """
    df = df.copy()
    if 'Amount' in df.columns:
        if scaler is None:
            scaler = StandardScaler()
            df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
        else:
            df['Amount_scaled'] = scaler.transform(df[['Amount']])
        df = df.drop(columns=['Amount'])
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])
    return df, scaler

def compute_risk_score(prob: float) -> int:
    """
    Convert model fraud probability to a 0-100 risk score.
    Straightforward linear scaling with small calibration:
    """
    score = int(np.clip(prob * 100, 0, 100))
    return score

def persona_from_features(row: pd.Series, prob: float) -> str:
    """
    Heuristic persona classifier
    (since we don't have user identity, we output a risk persona)
    """
    amt = row.get('Amount_scaled', None)
    # if scaled amount not available, try raw Amount (rare)
    raw_amt = None
    if amt is None and 'Amount' in row:
        raw_amt = row['Amount']
    # Heuristics
    if prob > 0.8:
        return "High-Risk / Likely Fraudulent (Stolen Card / Account Takeover)"
    if prob > 0.6:
        return "Risky User (Unusual behaviour / Large single transaction)"
    if prob > 0.3:
        return "Watchlist User (Monitor closely)"
    return "Trusted User (Low risk)"

def load_artifacts(model_path="artifacts/rf_model.joblib", scaler_path="artifacts/scaler.joblib"):
    """
    Load model and scaler if exists
    """
    model = None; scaler = None
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print("Warning: could not load model:", e)
    try:
        scaler = joblib.load(scaler_path)
    except Exception:
        scaler = None
    return model, scaler
