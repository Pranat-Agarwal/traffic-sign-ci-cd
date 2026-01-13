import os
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "model/traffic_model.h5"

def test_model_file_exists():
    """
    Check if trained model file exists
    """
    assert os.path.exists(MODEL_PATH), "❌ Model file not found. Training step may have failed."

def test_model_prediction_shape():
    """
    Check model output shape
    """
    model = load_model(MODEL_PATH)
    dummy_input = np.random.rand(1, 32, 32, 1)
    prediction = model.predict(dummy_input)

    assert prediction.shape == (1, 43), "❌ Model output shape mismatch"
