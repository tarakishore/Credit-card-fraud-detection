# Credit Card Fraud Detection System

A machine learning-based fraud detection system that identifies fraudulent credit card transactions using Random Forest classification with SMOTE oversampling. The project includes a trained model, Streamlit web application for real-time predictions, and comprehensive utility modules.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Running the Web Application](#running-the-web-application)
  - [Single Prediction](#single-prediction)
- [Model Architecture](#model-architecture)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Evaluation Metrics](#evaluation-metrics)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Credit card fraud is a significant concern in the financial industry, with billions of dollars lost annually due to unauthorized transactions. This project implements a robust machine learning solution to detect fraudulent transactions in real-time.

The system leverages:
- **Random Forest Classifier** with 200 estimators for high accuracy
- **SMOTE (Synthetic Minority Over-sampling Technique)** to handle class imbalance
- **StandardScaler** for feature normalization
- **Streamlit** for an interactive web-based interface

---

## Features

| Feature | Description |
|---------|-------------|
| Real-time Prediction | Instant fraud detection through web interface |
| SMOTE Oversampling | Handles highly imbalanced datasets effectively |
| Risk Scoring | Converts probability to 0-100 risk score |
| Persona Classification | Categorizes users based on risk level |
| Feature Importance | Visualizes top contributing features |
| Model Persistence | Saves and loads trained models using joblib |
| Interactive Dashboard | Streamlit-based UI for easy predictions |

---

## Project Structure

```
credit-card-fraud-detection/
|
|-- app_streamlit.py      # Streamlit web application
|-- train_model.py        # Model training pipeline
|-- predict_single.py     # Single transaction prediction script
|-- test.py               # Utility functions and helpers (utils)
|-- data.py               # Data handling module
|-- fraud_model.pkl       # Pre-trained Random Forest model
|-- requirement.txt       # Python dependencies
|-- .gitignore            # Git ignore configuration
|
|-- dataset/
|   |-- creditcard.csv    # Transaction dataset (not included in repo)
|   |-- archive/          # Additional data files
|
|-- model/
|   |-- [saved models]    # Directory for model artifacts
|
|-- artifacts/
|   |-- rf_model.joblib   # Trained Random Forest model
|   |-- scaler.joblib     # Fitted StandardScaler
|   |-- feature_importances.png  # Feature importance visualization
```

---

## Dataset

The project uses the **Credit Card Fraud Detection Dataset** from Kaggle, which contains:

| Attribute | Details |
|-----------|---------|
| Transactions | 284,807 total transactions |
| Frauds | 492 fraudulent transactions (0.172%) |
| Features | 30 features (Time, V1-V28, Amount) |
| PCA Features | V1-V28 are PCA-transformed (anonymized) |
| Time | Seconds elapsed since first transaction |
| Amount | Transaction amount in USD |
| Class | Target variable (0=Legitimate, 1=Fraud) |

**Note:** Due to file size (~150MB), the dataset is excluded from the repository. Download it from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `dataset/` directory.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/tarakishore/Credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   ```

3. **Activate Virtual Environment**
   
   Windows:
   ```bash
   venv\Scripts\activate
   ```
   
   Linux/macOS:
   ```bash
   source venv/bin/activate
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirement.txt
   ```

### Dependencies

| Package | Purpose |
|---------|---------|
| pandas | Data manipulation and analysis |
| numpy | Numerical computing |
| scikit-learn | Machine learning algorithms |
| imbalanced-learn | SMOTE oversampling |
| joblib | Model serialization |
| streamlit | Web application framework |
| matplotlib | Data visualization |
| shap | Model interpretability |
| plotly | Interactive visualizations |
| reportlab | PDF report generation |
| xgboost | Gradient boosting (alternative model) |

---

## Usage

### Training the Model

To train the fraud detection model from scratch:

```bash
python train_model.py
```

**Training Pipeline:**

1. Loads transaction data from `creditcard.csv`
2. Preprocesses features (scales Amount, drops Time)
3. Splits data into 80% training and 20% testing
4. Applies SMOTE to balance class distribution
5. Trains Random Forest with 200 estimators
6. Evaluates model performance
7. Saves model and scaler to `artifacts/` directory
8. Generates feature importance plot

**Expected Output:**
```
Loading data from creditcard.csv
Rows: 284807
Class distribution:
 0    284315
 1       492
Train class ratio: {0: 0.998..., 1: 0.001...}
Test class ratio: {0: 0.998..., 1: 0.001...}
After SMOTE: 0    227451
             1    227451
Training RandomForest...
Classification report (threshold=0.5):
              precision    recall  f1-score   support
           0     0.9999    0.9996    0.9998     56864
           1     0.9375    0.7959    0.8610        98
ROC AUC: 0.97xx, PR AUC: 0.85xx
Saved model and scaler to artifacts
```

### Running the Web Application

Launch the interactive Streamlit dashboard:

```bash
streamlit run app_streamlit.py
```

The application opens in your default browser at `http://localhost:8501`

**Features of the Web Application:**

1. **Input Form** - Enter transaction details:
   - Time (seconds elapsed)
   - Amount (transaction value in USD)
   - V1-V28 (PCA-transformed features)

2. **Real-time Prediction** - Displays:
   - Fraud status (FRAUD DETECTED or LEGITIMATE)
   - Fraud probability percentage
   - Confidence progress bar
   - Input data table

3. **Sidebar** - Shows model status and feature information

### Single Prediction

For programmatic single transaction prediction:

```bash
python predict_single.py
```

**Example Usage in Python:**

```python
from utils import preprocess_df, load_artifacts
import pandas as pd

# Load model and scaler
model, scaler = load_artifacts()

# Create sample transaction
sample = {
    'V1': -1.2, 
    'V2': 0.5, 
    'V3': -0.3, 
    'Amount': 300.0
}

# Preprocess and predict
df = pd.DataFrame([sample])
df_proc, _ = preprocess_df(df, scaler=scaler)

# Get prediction probability
prob = model.predict_proba(df_proc)[:, 1][0]
label = "Fraud" if prob >= 0.5 else "Safe"

print(f"Probability: {prob:.4f}, Label: {label}")
```

---

## Model Architecture

### Random Forest Classifier

| Parameter | Value | Description |
|-----------|-------|-------------|
| n_estimators | 200 | Number of decision trees |
| class_weight | balanced | Adjusts weights inversely proportional to class frequencies |
| random_state | 42 | Ensures reproducibility |
| n_jobs | -1 | Uses all available CPU cores |

### Class Imbalance Handling

The dataset is highly imbalanced (99.83% legitimate, 0.17% fraud). Two strategies are employed:

1. **SMOTE Oversampling**
   - Generates synthetic fraud samples
   - Balances training set to 50-50 distribution
   - Applied only to training data (not test data)

2. **Balanced Class Weights**
   - Random Forest automatically adjusts decision thresholds
   - Penalizes misclassification of minority class

---

## Preprocessing Pipeline

```
Raw Data --> Scale Amount --> Drop Time --> Feature Matrix
    |              |              |              |
    v              v              v              v
[Time, V1-V28,  [V1-V28,      [V1-V28,      Ready for
 Amount]         Amount_scaled] Amount_scaled] prediction
```

### Preprocessing Steps

| Step | Operation | Reason |
|------|-----------|--------|
| 1 | StandardScaler on Amount | Normalizes transaction values |
| 2 | Drop Time column | Low predictive value |
| 3 | Handle missing features | Fills with 0.0 if not present |

### Utility Functions

**preprocess_df(df, scaler)**
- Scales Amount feature
- Removes Time column
- Returns processed DataFrame and scaler

**compute_risk_score(prob)**
- Converts probability (0-1) to risk score (0-100)
- Linear scaling with clipping

**persona_from_features(row, prob)**
- Classifies user risk personas:
  - prob > 0.8: High-Risk / Likely Fraudulent
  - prob > 0.6: Risky User
  - prob > 0.3: Watchlist User
  - prob <= 0.3: Trusted User

---

## Evaluation Metrics

### Primary Metrics

| Metric | Description | Importance |
|--------|-------------|------------|
| ROC AUC | Area under ROC curve | Overall discrimination ability |
| PR AUC | Area under Precision-Recall curve | Performance on imbalanced data |
| Recall | True Positive Rate | Critical for fraud detection |
| Precision | Positive Predictive Value | Reduces false alarms |

### Confusion Matrix Interpretation

```
                    Predicted
                 Legitimate  Fraud
Actual Legitimate     TN      FP
       Fraud          FN      TP
```

- **TN (True Negative):** Correctly identified legitimate transactions
- **FP (False Positive):** Legitimate transactions flagged as fraud
- **FN (False Negative):** Missed fraud cases (most costly)
- **TP (True Positive):** Correctly detected fraud

### Performance Targets

| Metric | Target | Priority |
|--------|--------|----------|
| Recall | > 80% | HIGH - Minimize missed frauds |
| Precision | > 70% | MEDIUM - Reduce false positives |
| ROC AUC | > 0.95 | HIGH - Overall model quality |
| PR AUC | > 0.80 | HIGH - Imbalanced data performance |

---

## API Reference

### train_model.py

```python
load_data(path: str) -> pd.DataFrame
    """Load transaction data from CSV file."""

run_training() -> None
    """Execute complete training pipeline."""
```

### test.py (utils)

```python
preprocess_df(df: pd.DataFrame, scaler: StandardScaler = None) 
    -> Tuple[pd.DataFrame, StandardScaler]
    """Preprocess DataFrame for model prediction."""

compute_risk_score(prob: float) -> int
    """Convert probability to 0-100 risk score."""

persona_from_features(row: pd.Series, prob: float) -> str
    """Classify user into risk persona category."""

load_artifacts(model_path: str, scaler_path: str) 
    -> Tuple[model, scaler]
    """Load trained model and scaler from disk."""
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| RANDOM_STATE | 42 | Random seed for reproducibility |
| DATA_PATH | creditcard.csv | Path to dataset |
| OUT_DIR | artifacts | Output directory for models |

### Model Parameters

Modify in `train_model.py`:

```python
# Training configuration
RANDOM_STATE = 42
DATA_PATH = "creditcard.csv"
OUT_DIR = "artifacts"

# Model hyperparameters
n_estimators = 200
class_weight = 'balanced'
test_size = 0.2
```

### Prediction Threshold

Default threshold is 0.5. Adjust based on use case:
- Lower threshold (0.3): More sensitive, catches more fraud, more false positives
- Higher threshold (0.7): More specific, fewer false positives, may miss some fraud

---

## Contributing

Contributions are welcome. Please follow these guidelines:

### Development Workflow

1. Fork the repository
2. Create a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes
4. Run tests and validation
5. Commit with clear messages
   ```bash
   git commit -m "Add: description of changes"
   ```
6. Push to your fork
   ```bash
   git push origin feature/your-feature-name
   ```
7. Open a Pull Request

### Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints where appropriate
- Update documentation for new features
- Add unit tests for new functionality

### Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces

---

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2026 Tara Kishore

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Acknowledgments

- Dataset: Credit Card Fraud Detection Dataset from Kaggle (ULB Machine Learning Group)
- Libraries: scikit-learn, imbalanced-learn, Streamlit, and the Python data science ecosystem

---

## Contact

For questions, issues, or contributions, please open an issue on GitHub or submit a pull request.

---

**Last Updated:** January 2026