# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score, precision_recall_curve, auc, confusion_matrix)
from imblearn.over_sampling import SMOTE
import joblib
import os
import matplotlib.pyplot as plt
from utils import preprocess_df

RANDOM_STATE = 42
DATA_PATH = "creditcard.csv"   # put dataset here
OUT_DIR = "artifacts"

def load_data(path=DATA_PATH):
    print("Loading data from", path)
    df = pd.read_csv(path)
    return df

def run_training():
    df = load_data()
    print("Rows:", len(df))
    print("Class distribution:\n", df['Class'].value_counts())

    X = df.drop(columns=['Class'])
    y = df['Class']

    # Preprocess Amount
    X_proc, scaler = preprocess_df(X, scaler=None)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print("Train class ratio:", y_train.value_counts(normalize=True).to_dict())
    print("Test class ratio:", y_test.value_counts(normalize=True).to_dict())

    # SMOTE
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print("After SMOTE:", pd.Series(y_train_res).value_counts())

    # Train RandomForest
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    print("Training RandomForest...")
    rf.fit(X_train_res, y_train_res)

    # Evaluate
    probs = rf.predict_proba(X_test)[:,1]
    preds = (probs >= 0.5).astype(int)
    print("\nClassification report (threshold=0.5):")
    print(classification_report(y_test, preds, digits=4))
    roc = roc_auc_score(y_test, probs)
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)
    print(f"ROC AUC: {roc:.4f}, PR AUC: {pr_auc:.4f}")
    cm = confusion_matrix(y_test, preds)
    print("Confusion matrix:\n", cm)

    # Save artifacts
    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(rf, os.path.join(OUT_DIR, "rf_model.joblib"))
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
    print("Saved model and scaler to", OUT_DIR)

    # Feature importances plot
    try:
        importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=True).tail(20)
        plt.figure(figsize=(6,8))
        importances.plot.barh()
        plt.title("Top feature importances")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "feature_importances.png"))
        print("Saved feature importances plot.")
    except Exception as e:
        print("Could not plot importances:", e)

if __name__ == "__main__":
    run_training()
