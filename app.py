from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model("model/fashion_image_classifier.h5")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@app.route("/")
def home():
    return "<h1>AI Image Classifier API is Running!</h1>"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["image"]
        img = Image.open(io.BytesIO(file.read())).convert("RGB").resize((32, 32))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 32, 32, 3)
        prediction = model.predict(img_array, verbose=0)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction) * 100)
        return jsonify({"predicted_class": predicted_class, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
