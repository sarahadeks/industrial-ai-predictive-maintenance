import pandas as pd

from src.predict import predict_machine


def test_prediction_returns_expected_fields():

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

    assert "failure_probability" in result
    assert "prediction" in result
    assert "status" in result