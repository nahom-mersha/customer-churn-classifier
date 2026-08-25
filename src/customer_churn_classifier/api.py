import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

CONFIG_PATH = Path("configs/final_model.yaml")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load final model configuration."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_metadata(path: Path) -> dict[str, Any]:
    """Load saved model metadata."""
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {path}. Run scripts/train_final_model.py first."
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_model(path: Path) -> Any:
    """Load saved preprocessing-and-model pipeline."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. Run scripts/train_final_model.py first."
        )

    return joblib.load(path)


config = load_config()
metadata = load_metadata(Path(config["artifacts"]["metadata_path"]))
model = load_model(Path(config["artifacts"]["model_path"]))


app = FastAPI(
    title="Customer Churn Classifier API",
    description="API for predicting customer churn from one customer record.",
    version="0.1.0",
)


class CustomerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    SeniorCitizen: int
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str


@app.get("/")
def read_root() -> dict[str, str]:
    """Basic API message."""
    return {"message": "Customer Churn Classifier API is running."}


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict")
def predict_churn(customer: CustomerRecord) -> dict[str, float | int]:
    """Predict churn probability and label for one customer."""
    try:
        input_data = pd.DataFrame([customer.model_dump()])
        input_data = input_data[metadata["features"]]

        probability = float(model.predict_proba(input_data)[:, 1][0])
        threshold = float(metadata["threshold"])
        predicted_churn = int(probability >= threshold)

        return {
            "churn_probability": probability,
            "decision_threshold": threshold,
            "predicted_churn": predicted_churn,
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {error}",
        ) from error
