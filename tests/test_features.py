import pandas as pd

from src.features import create_features


def test_feature_engineering():

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

    result = create_features(machine)

    expected_features = [
        "temperature_difference",
        "mechanical_power",
        "tool_stress",
        "temperature_ratio",
        "torque_speed_ratio",
    ]

    for feature in expected_features:
        assert feature in result.columns