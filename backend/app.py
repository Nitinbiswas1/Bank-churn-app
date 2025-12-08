from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd

# -----------------------------------------------------
# FastAPI App
# -----------------------------------------------------
app = FastAPI(title="Bank Churn Prediction API")

# -----------------------------------------------------
# Load ML Models (Pipelines with preprocessing included)
# -----------------------------------------------------
xgb_model = joblib.load("models/nate_random_forest.sav")
rf_model = joblib.load("models/XGBoost_model.sav")

# -----------------------------------------------------
# Try loading test data for dashboard metrics
# -----------------------------------------------------
try:
    X_test = joblib.load("models/X_test.sav")
    y_test = joblib.load("models/y_test.sav")
    HAS_TEST = True
except:
    X_test = None
    y_test = None
    HAS_TEST = False

# -----------------------------------------------------
# Expected input order for raw data (VERY IMPORTANT)
# -----------------------------------------------------
EXPECTED_COLS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]

# -----------------------------------------------------
# Pydantic Models for API Input
# -----------------------------------------------------
class InputData(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

class BatchInput(BaseModel):
    customers: List[InputData]

# -----------------------------------------------------
# Convert input → DataFrame (DO NOT preprocess here!)
# -----------------------------------------------------
def preprocess_inputs(items):
    rows = []
    for item in items:
        d = item.dict()
        row = [d[col] for col in EXPECTED_COLS]  # keep order exact
        rows.append(row)
    return pd.DataFrame(rows, columns=EXPECTED_COLS)

# -----------------------------------------------------
# Single Prediction Endpoint
# -----------------------------------------------------
@app.post("/predict")
def predict(data: InputData):
    df = preprocess_inputs([data])

    # model pipeline handles preprocessing internally
    xgb_prob = xgb_model.predict_proba(df)[0][1]
    rf_prob = rf_model.predict_proba(df)[0][1]

    return {
        "xgboost_churn_probability": float(xgb_prob),
        "random_forest_churn_probability": float(rf_prob),
        "average_probability": float((xgb_prob + rf_prob) / 2)
    }

# -----------------------------------------------------
# Batch Prediction Endpoint
# -----------------------------------------------------
@app.post("/predict-batch")
def predict_batch(batch: BatchInput):
    df = preprocess_inputs(batch.customers)

    # Model predictions
    xgb_probs = xgb_model.predict_proba(df)[:, 1]
    rf_probs = rf_model.predict_proba(df)[:, 1]
    avg_probs = (xgb_probs + rf_probs) / 2

    # Determine churn YES/NO for each row
    exit_labels = ["YES" if p >= 0.5 else "NO" for p in avg_probs]

    # Build structured output
    results = []
    for i in range(len(df)):
        results.append({
            "xgboost_probability": float(xgb_probs[i]),
            "random_forest_probability": float(rf_probs[i]),
            "average_probability": float(avg_probs[i]),
            "will_exit": exit_labels[i]
        })

    return {"predictions": results}


# -----------------------------------------------------
# Metrics Dashboard Endpoint (ROC, Confusion Matrix)
# -----------------------------------------------------
@app.get("/metrics")
def metrics():
    if not HAS_TEST:
        return {"detail": "No X_test / y_test available on server."}

    from sklearn.metrics import roc_curve, auc, confusion_matrix

    # model pipelines internally preprocess X_test
    xgb_scores = xgb_model.predict_proba(X_test)[:, 1]
    rf_scores = rf_model.predict_proba(X_test)[:, 1]

    # ROC curves
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_scores)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_scores)

    auc_xgb = auc(fpr_xgb, tpr_xgb)
    auc_rf = auc(fpr_rf, tpr_rf)

    # Confusion matrices
    xgb_preds = xgb_model.predict(X_test)
    rf_preds = rf_model.predict(X_test)

    cm_xgb = confusion_matrix(y_test, xgb_preds).tolist()
    cm_rf = confusion_matrix(y_test, rf_preds).tolist()

    return {
        "roc": {
            "xgboost": {
                "fpr": fpr_xgb.tolist(),
                "tpr": tpr_xgb.tolist(),
                "auc": auc_xgb
            },
            "random_forest": {
                "fpr": fpr_rf.tolist(),
                "tpr": tpr_rf.tolist(),
                "auc": auc_rf
            },
        },
        "confusion_matrix": {
            "xgboost": cm_xgb,
            "random_forest": cm_rf
        }
    }
