import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Flatten

def preprocess_data(x):
    """
    Convert RGB images to grayscale and normalize
    """
    x_gray = np.sum(x / 3, axis=3, keepdims=True)
    x_norm = (x_gray - 128) / 128
    return x_norm

def build_cnn_model():
    """
    Build CNN model
    """
    model = Sequential()

    model.add(Conv2D(6, (5,5), activation='relu', input_shape=(32,32,1)))
    model.add(AveragePooling2D((2,2)))

    model.add(Conv2D(16, (5,5), activation='relu'))
    model.add(AveragePooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(43, activation='softmax'))

    return model
