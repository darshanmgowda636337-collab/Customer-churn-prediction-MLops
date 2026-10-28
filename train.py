

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from validation import validate_data


SEED = 42
MINIMUM_AUC = 0.60
TEST_SIZE = 0.20
MAX_ITER = 1000

NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "support_calls",
]

CATEGORICAL_FEATURES = [
    "contract",
    "internet_service",
]

TARGET = "churn"


def build_model():
    """Build the preprocessing and classification pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=MAX_ITER,
                    random_state=SEED,
                ),
            ),
        ]
    )

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the customer churn model."
    )

    parser.add_argument(
        "--check-quality-gate",
        action="store_true",
        help="Fail if ROC-AUC is below the minimum threshold.",
    )

    args = parser.parse_args()

    # Step 1: Validate the dataset
    df = validate_data()

    # Step 2: Separate features and target
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # Step 3: Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    # Step 4: Create the artifacts directory
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Step 5: Configure MLflow
    mlflow.set_experiment("Customer Churn Prediction")

    # Step 6: Train, evaluate, and log the model
    with mlflow.start_run() as run:
        model = build_model()

        model.fit(X_train, y_train)

        # Generate predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)

        # Log model parameters
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "random_seed": SEED,
                "test_size": TEST_SIZE,
                "max_iter": MAX_ITER,
                "dataset_rows": len(df),
            }
        )

        # Log evaluation metrics
        mlflow.log_metrics(
            {
                "accuracy": float(accuracy),
                "auc": float(auc),
            }
        )

        # Log the model in MLflow
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
        )

        # Save the model locally
        model_path = artifacts_dir / "model.joblib"
        joblib.dump(model, model_path)

        # Save evaluation metrics
        metrics = {
            "accuracy": float(accuracy),
            "auc": float(auc),
            "minimum_auc": MINIMUM_AUC,
            "run_id": run.info.run_id,
            "rows": len(df),
        }

        metrics_path = artifacts_dir / "metrics.json"

        with open(metrics_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)

        # Display results
        print("\nModel Training Completed")
        print("-" * 35)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {auc:.4f}")
        print(f"MLflow run ID: {run.info.run_id}")
        print(f"Model saved to: {model_path}")
        print(f"Metrics saved to: {metrics_path}")

        # Step 7: Check the quality gate
        if args.check_quality_gate and auc < MINIMUM_AUC:
            raise RuntimeError(
                f"Quality gate failed: ROC-AUC {auc:.4f} "
                f"is below {MINIMUM_AUC:.2f}"
            )

        print("Quality gate passed.")


if __name__ == "__main__":
    main()

