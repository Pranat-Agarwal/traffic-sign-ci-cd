from fastapi import FastAPI
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

model = load_model("model/traffic_model.h5")

@app.get("/")
def home():
    return {"message": "Traffic Sign Model API is running"}

@app.post("/predict")
def predict(image: list):
    img = np.array(image).reshape(1, 32, 32, 1)
    prediction = model.predict(img)
    return {"predicted_class": int(np.argmax(prediction))}
