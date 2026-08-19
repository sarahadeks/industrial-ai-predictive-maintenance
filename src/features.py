import numpy as np
import pandas as pd


def create_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the features required by the
    predictive maintenance model.
    """

    df = data.copy()

    df["temperature_difference"] = (
        df["process_temperature"]
        - df["air_temperature"]
    )

    df["mechanical_power"] = (
        df["torque"]
        * df["rotational_speed"]
        * (2 * np.pi / 60)
    )

    df["tool_stress"] = (
        df["torque"]
        * df["tool_wear"]
    )

    df["temperature_ratio"] = (
        df["process_temperature"]
        / df["air_temperature"]
    )

    df["torque_speed_ratio"] = (
        df["torque"]
        / df["rotational_speed"]
    )

    return df