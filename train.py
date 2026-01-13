import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from traffic import preprocess_data, build_cnn_model

# ---------------------------------------------------
# Detect CI environment
# GitHub Actions automatically sets CI=true
# ---------------------------------------------------
CI = os.getenv("CI", "false").lower() == "true"

# Create model directory
os.makedirs("model", exist_ok=True)

# ---------------------------------------------------
# Load Data
# ---------------------------------------------------
if CI:
    print("Running in CI environment → using MOCK data")

    # Small dummy dataset for CI/CD
    x_train = np.random.rand(20, 32, 32, 3)
    y_train = np.random.randint(0, 43, 20)

    x_valid = np.random.rand(10, 32, 32, 3)
    y_valid = np.random.randint(0, 43, 10)

else:
    print("Running locally → loading REAL dataset")

    import pickle

    with open("train.p", "rb") as f:
        train = pickle.load(f)

    with open("valid.p", "rb") as f:
        valid = pickle.load(f)

    x_train, y_train = train["features"], train["labels"]
    x_valid, y_valid = valid["features"], valid["labels"]

# ---------------------------------------------------
# Preprocess Data
# ---------------------------------------------------
x_train = preprocess_data(x_train)
x_valid = preprocess_data(x_valid)

# ---------------------------------------------------
# Build & Compile Model
# ---------------------------------------------------
model = build_cnn_model()

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------------------------------
# Train Model
# ---------------------------------------------------
model.fit(
    x_train,
    y_train,
    epochs=2 if CI else 5,   # short training in CI
    batch_size=64,
    validation_data=(x_valid, y_valid),
    verbose=1
)

# ---------------------------------------------------
# Save Model
# ---------------------------------------------------
model.save("model/traffic_model.h5")
print("✅ Model saved at model/traffic_model.h5")
