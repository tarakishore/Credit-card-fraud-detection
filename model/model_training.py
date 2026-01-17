import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
import xgboost as xgb
import joblib

# ------------------------------
# 1. LOAD DATA
# ------------------------------
data_path = r"C:\Users\Flash\OneDrive\Desktop\flash\dataset\creditcard.csv"
df = pd.read_csv(data_path)

print("Dataset loaded!")
print(df.head())

# ------------------------------
# 2. BALANCE DATA
# ------------------------------
fraud = df[df['Class'] == 1]
non_fraud = df[df['Class'] == 0].sample(len(fraud) * 3, random_state=42)

balanced_df = pd.concat([fraud, non_fraud])
balanced_df = shuffle(balanced_df, random_state=42)

print(f"Balanced dataset shape: {balanced_df.shape}")

# ------------------------------
# 3. SPLIT FEATURES + LABEL
# ------------------------------
X = balanced_df.drop("Class", axis=1)
y = balanced_df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# 4. TRAIN MODEL (XGBOOST)
# ------------------------------
model = xgb.XGBClassifier(
    max_depth=5,
    learning_rate=0.05,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

print("Training model...")
model.fit(X_train, y_train)

# ------------------------------
# 5. EVALUATE
# ------------------------------
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ------------------------------
# 6. SAVE MODEL
# ------------------------------
joblib.dump(model, "fraud_model.pkl")
print("\nModel saved as fraud_model.pkl")
