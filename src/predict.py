from pathlib import Path

import joblib
import pandas as pd

from src.features import create_features


# --------------------------------------------------
# 1. Locate the project root
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "predictive_maintenance_model.joblib"
)


# --------------------------------------------------
# 2. Load the trained model
# --------------------------------------------------

model_package = joblib.load(MODEL_PATH)

model = model_package["model"]
threshold = model_package["threshold"]


# --------------------------------------------------
# 3. Prediction function
# --------------------------------------------------

def predict_machine(
    machine_data: pd.DataFrame,
) -> dict:
    """
    Generate a predictive maintenance
    failure prediction for a machine.
    """

    # Create model features
    machine_features = create_features(
        machine_data
    )

    # Generate failure probability
    failure_probability = (
        model.predict_proba(
            machine_features
        )[:, 1][0]
    )

    # Apply decision threshold
    prediction = int(
        failure_probability >= threshold
    )

    # Determine status
    if prediction == 1:
        status = "FAILURE RISK"
    else:
        status = "NO FAILURE"

    return {
        "failure_probability": round(
            float(failure_probability),
            4,
        ),
        "prediction": prediction,
        "status": status,
    }


# --------------------------------------------------
# 4. Test when running this file directly
# --------------------------------------------------

if __name__ == "__main__":

    machine = pd.DataFrame(
        [
            {
                "product_type": "L",
                "air_temperature": 298.5,
                "process_temperature": 308.7,
                "rotational_speed": 1450,
                "torque": 48.3,
                "tool_wear": 180,
            }
        ]
    )

    result = predict_machine(machine)

    print("Prediction result:")
    print(result)