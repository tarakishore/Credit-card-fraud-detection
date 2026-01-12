# predict_single.py
import joblib
import pandas as pd
from utils import preprocess_df, load_artifacts

model, scaler = load_artifacts()
if model is None:
    print("Run train_model.py first.")
    exit(1)

sample = {
    'V1': -1.2, 'V2': 0.5, 'V3': -0.3, 'Amount': 300.0
}
df = pd.DataFrame([sample])
df_proc, _ = preprocess_df(df, scaler=scaler)
model_cols = model.feature_names_in_ if hasattr(model, "feature_names_in_") else df_proc.columns
for c in model_cols:
    if c not in df_proc.columns:
        df_proc[c] = 0.0
df_proc = df_proc[model_cols]
prob = model.predict_proba(df_proc)[:,1][0]
label = "Fraud" if prob >= 0.5 else "Safe"
print("Probability:", prob, "Label:", label)
