from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd

from src.predict import predict_machine


# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Industrial AI Predictive Maintenance API",
    description=(
        "API for predicting machine failure "
        "using the AI4I predictive maintenance model."
    ),
    version="1.0.0",
)


# --------------------------------------------------
# Define input data structure
# --------------------------------------------------

class MachineData(BaseModel):
    product_type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float


# --------------------------------------------------
# Health check endpoint
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": (
            "Industrial AI Predictive "
            "Maintenance API is running."
        )
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(data: MachineData):

    machine = pd.DataFrame(
        [
            data.model_dump()
        ]
    )

    result = predict_machine(machine)

    return result