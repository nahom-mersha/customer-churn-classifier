import numpy as np
from fastapi.testclient import TestClient

from customer_churn_classifier.api import create_app

FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


VALID_CUSTOMER = {
    "SeniorCitizen": 0,
    "tenure": 1,
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
    "gender": "Female",
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}


class FakeModel:
    def predict_proba(self, input_data):
        return np.array([[0.2, 0.8]])


def test_health_endpoint_returns_ok():
    app = create_app(
        model=FakeModel(),
        metadata={"features": FEATURES, "threshold": 0.5},
    )
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_probability_threshold_and_label():
    app = create_app(
        model=FakeModel(),
        metadata={"features": FEATURES, "threshold": 0.5},
    )
    client = TestClient(app)

    response = client.post("/predict", json=VALID_CUSTOMER)

    assert response.status_code == 200
    assert response.json() == {
        "churn_probability": 0.8,
        "decision_threshold": 0.5,
        "predicted_churn": 1,
    }


def test_predict_endpoint_rejects_missing_required_field():
    app = create_app(
        model=FakeModel(),
        metadata={"features": FEATURES, "threshold": 0.5},
    )
    client = TestClient(app)

    invalid_customer = VALID_CUSTOMER.copy()
    invalid_customer.pop("tenure")

    response = client.post("/predict", json=invalid_customer)

    assert response.status_code == 422
