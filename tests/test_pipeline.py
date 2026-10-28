
from pathlib import Path
import subprocess
import sys

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "customers.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"

sys.path.insert(0, str(BASE_DIR))

from api import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_dataset_exists():
    """The customer dataset must exist."""
    assert DATA_PATH.exists(), f"Dataset not found: {DATA_PATH}"

def test_dataset_has_required_columns():
    """Check that the dataset has all required features."""
    assert DATA_PATH.exists()

    df = pd.read_csv(
        DATA_PATH,
        keep_default_na=False,
        na_values=[""]
    )

    required_columns = {
        "tenure",
        "monthly_charges",
        "support_calls",
        "contract",
        "internet_service",
        "churn",
    }

    assert required_columns.issubset(set(df.columns))
    assert len(df) > 0

def test_dataset_has_no_missing_values():
    """Check that the dataset contains no missing values."""
    assert DATA_PATH.exists()

    df = pd.read_csv(
        DATA_PATH,
        keep_default_na=False,
        na_values=[""]
    )

    assert not df.isnull().any().any()

def test_churn_values_are_valid():
    """The target must contain only 0 and 1."""
    df = pd.read_csv(
        DATA_PATH,
        keep_default_na=False,
        na_values=[""]
    )

    assert set(df["churn"].unique()).issubset({0, 1})
    assert set(df["churn"].unique()) == {0, 1}

def test_validation_script():
    """Run the project's data validation script."""
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "validation.py")],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"Validation failed:\n{result.stdout}\n{result.stderr}"
    )

def test_model_artifact_exists():
    """The trained model must be saved."""
    assert MODEL_PATH.exists(), (
        "Model not found. Run python train.py before running this test."
    )

def test_model_prediction():
    """The model must predict a valid churn class."""
    assert MODEL_PATH.exists()

    model = joblib.load(MODEL_PATH)

    customer = pd.DataFrame([{
        "tenure": 6,
        "monthly_charges": 110.0,
        "support_calls": 4,
        "contract": "Monthly",
        "internet_service": "Fiber",
    }])

    prediction = model.predict(customer)

    assert len(prediction) == 1
    assert int(prediction[0]) in (0, 1)

def test_model_probability():
    """The model must return a valid probability."""
    assert MODEL_PATH.exists()

    model = joblib.load(MODEL_PATH)

    customer = pd.DataFrame([{
        "tenure": 6,
        "monthly_charges": 110.0,
        "support_calls": 4,
        "contract": "Monthly",
        "internet_service": "Fiber",
    }])

    probability = float(model.predict_proba(customer)[0][1])

    assert 0.0 <= probability <= 1.0

def test_home_endpoint(client):
    """The home endpoint must respond successfully."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_health_endpoint(client):
    """The health endpoint must return a valid status."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert isinstance(response.json()["model_loaded"], bool)

def test_predict_endpoint(client):
    """The prediction API must return the expected response fields."""
    response = client.post(
        "/predict",
        json={
            "tenure": 6,
            "monthly_charges": 110,
            "support_calls": 4,
            "contract": "Monthly",
            "internet_service": "Fiber",
        },
    )

    if not MODEL_PATH.exists():
        pytest.skip("Train the model before testing predictions.")

    assert response.status_code == 200

    result = response.json()

    assert result["churn_prediction"] in (0, 1)
    assert result["churn_label"] in (
        "Likely to churn",
        "Likely to stay",
    )
    assert 0.0 <= result["churn_probability"] <= 1.0

def test_predict_rejects_invalid_contract(client):
    """Invalid contract values must be rejected."""
    response = client.post(
        "/predict",
        json={
            "tenure": 6,
            "monthly_charges": 110,
            "support_calls": 4,
            "contract": "Invalid Contract",
            "internet_service": "Fiber",
        },
    )

    assert response.status_code == 422
    assert "Invalid contract" in response.json()["detail"]