from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

input_shape = (224, 224, 3)  

def get_model():

    model = Sequential()


    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))


    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid')) 

    return model

app = Flask(__name__)

def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust based on model input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image uploads and return class prediction."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    processed_img = preprocess_image(img)

    # Model Prediction
    prediction = model.predict(processed_img)[0][0]  # Extract single prediction score

    # Convert to class label
    label = "Normal" if prediction > 0.5 else "Cataract"

    if label == 'Cataract':
        prediction = 1 - float(prediction)
    
    return jsonify({"class": label, "confidence": float(prediction)})

if __name__ == "__main__":
    model = get_model()
    model.load_weights("models/cataract_best.weights.h5")
    app.run(host="0.0.0.0", port=5003, debug=True)