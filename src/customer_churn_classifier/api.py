import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Request
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


def create_app(
    model: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> FastAPI:
    """Create the FastAPI application.

    Optional model and metadata arguments make the app easier to test without
    requiring a real saved model artifact.
    """
    app = FastAPI(
        title="Customer Churn Classifier API",
        description="API for predicting customer churn from one customer record.",
        version="0.1.0",
    )

    app.state.model = model
    app.state.metadata = metadata

    @app.get("/")
    def read_root() -> dict[str, str]:
        """Basic API message."""
        return {"message": "Customer Churn Classifier API is running."}

    @app.get("/health")
    def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    @app.post("/predict")
    def predict_churn(
        customer: CustomerRecord,
        request: Request,
    ) -> dict[str, float | int]:
        """Predict churn probability and label for one customer."""
        try:
            if request.app.state.metadata is None or request.app.state.model is None:
                config = load_config()
                request.app.state.metadata = load_metadata(
                    Path(config["artifacts"]["metadata_path"])
                )
                request.app.state.model = load_model(
                    Path(config["artifacts"]["model_path"])
                )

            saved_metadata = request.app.state.metadata
            saved_model = request.app.state.model

            input_data = pd.DataFrame([customer.model_dump()])
            input_data = input_data[saved_metadata["features"]]

            probability = float(saved_model.predict_proba(input_data)[:, 1][0])
            threshold = float(saved_metadata["threshold"])
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

    return app


app = create_app()
