import json

import numpy as np
import pandas as pd
import yaml

from scripts.predict_batch import predict_batch

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


class FakeModel:
    def predict_proba(self, input_data):
        return np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
            ]
        )


def make_valid_input_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
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
            },
            {
                "SeniorCitizen": 1,
                "tenure": 10,
                "MonthlyCharges": 90.0,
                "TotalCharges": 900.0,
                "gender": "Male",
                "Partner": "No",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
            },
        ]
    )


def test_predict_batch_writes_expected_prediction_columns(tmp_path, monkeypatch):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    metadata_path = tmp_path / "metadata.json"
    config_path = tmp_path / "config.yaml"

    input_data = make_valid_input_data()
    input_data.to_csv(input_path, index=False)

    metadata_path.write_text(
        json.dumps(
            {
                "features": FEATURES,
                "threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )

    config_path.write_text(
        yaml.safe_dump(
            {
                "artifacts": {
                    "model_path": "fake_model.joblib",
                    "metadata_path": str(metadata_path),
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scripts.predict_batch.load_model",
        lambda path: FakeModel(),
    )

    predict_batch(
        input_path=input_path,
        output_path=output_path,
        config_path=config_path,
    )

    output_data = pd.read_csv(output_path)

    assert "churn_probability" in output_data.columns
    assert "predicted_churn" in output_data.columns
    assert "decision_threshold" in output_data.columns

    assert output_data["churn_probability"].tolist() == [0.2, 0.7]
    assert output_data["predicted_churn"].tolist() == [0, 1]
    assert output_data["decision_threshold"].tolist() == [0.5, 0.5]
