
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"

# Create FastAPI application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using machine learning.",
    version="1.0.0",
)

# Customer input schema
class Customer(BaseModel):
    tenure: int = Field(ge=1, le=72)
    monthly_charges: float = Field(ge=20, le=130)
    support_calls: int = Field(ge=0)
    contract: str
    internet_service: str


# Load the trained model
model = None


@app.on_event("startup")
def load_model():
    global model

    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print("Trained model loaded successfully.")
    else:
        print(f"Model not found: {MODEL_PATH}")
        print("Run python train.py before making predictions.")


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running",
        "status": "success",
    }


# Health-check endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


# Prediction endpoint
@app.post("/predict")
def predict_churn(customer: Customer):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run python train.py first.",
        )

    # Validate categorical inputs
    valid_contracts = ["Monthly", "One Year", "Two Year"]
    valid_internet = ["DSL", "Fiber", "None"]

    if customer.contract not in valid_contracts:
        raise HTTPException(
            status_code=422,
            detail="Invalid contract. Use Monthly, One Year, or Two Year.",
        )

    if customer.internet_service not in valid_internet:
        raise HTTPException(
            status_code=422,
            detail="Invalid internet service. Use DSL, Fiber, or None.",
        )

    # Prepare customer data
    customer_data = pd.DataFrame(
        [
            {
                "tenure": customer.tenure,
                "monthly_charges": customer.monthly_charges,
                "support_calls": customer.support_calls,
                "contract": customer.contract,
                "internet_service": customer.internet_service,
            }
        ]
    )

    # Predict churn
    prediction = int(model.predict(customer_data)[0])

    probability = float(
        model.predict_proba(customer_data)[0][1]
    )

    return {
        "churn_prediction": prediction,
        "churn_label": (
            "Likely to churn"
            if prediction == 1
            else "Likely to stay"
        ),
        "churn_probability": round(probability, 4),
    }

