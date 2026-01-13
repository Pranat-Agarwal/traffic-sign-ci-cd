import numpy as np
from traffic import load_traffic_model, predict_image


def test_model_loads():
    model = load_traffic_model()
    assert model is not None


def test_model_prediction_shape():
    model = load_traffic_model()
    dummy_image = np.random.rand(32, 32, 1).tolist()
    prediction = predict_image(model, dummy_image)
    assert isinstance(prediction, int)
