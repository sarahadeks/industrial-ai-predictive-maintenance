from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": (
            "Industrial AI Predictive "
            "Maintenance API is running."
        )
    }


def test_prediction_endpoint():

    machine = {
        "product_type": "L",
        "air_temperature": 298.5,
        "process_temperature": 308.7,
        "rotational_speed": 1450,
        "torque": 48.3,
        "tool_wear": 180,
    }

    response = client.post(
        "/predict",
        json=machine,
    )

    assert response.status_code == 200

    result = response.json()

    assert "failure_probability" in result
    assert "prediction" in result
    assert "status" in result