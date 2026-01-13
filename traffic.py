import numpy as np
from tensorflow.keras.models import load_model
import os

MODEL_PATH = "model/traffic_model.h5"


def load_traffic_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found")
    return load_model(MODEL_PATH)


def preprocess_image(image):
    img = np.array(image, dtype=np.float32)

    # Ensure correct shape
    img = img.reshape(1, 32, 32, 1)

    # Normalize (same as training)
    img = (img - 128) / 128
    return img


def predict_image(model, image):
    img = preprocess_image(image)
    prediction = model.predict(img)
    return int(np.argmax(prediction))
