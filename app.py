from fastapi import FastAPI
from traffic import load_traffic_model, predict_image

app = FastAPI(title="Traffic Sign CI/CD API")

# Load model once at startup
model = load_traffic_model()


@app.get("/")
def home():
    return {"message": "Traffic Sign Model API is running"}


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(image: list):
    """
    Expects image as a 32x32 grayscale image flattened or nested list
    """
    predicted_class = predict_image(model, image)
    return {"predicted_class": predicted_class}
