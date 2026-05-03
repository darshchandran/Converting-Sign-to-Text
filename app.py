from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

# Load model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(25, activation='softmax')
])

model.load_weights("model.h5")

# Load labels
with open("labels.json") as f:
    labels = json.load(f)

def normalize_landmarks(sample):
    sample = sample.reshape(21, 3)
    wrist = sample[0]
    sample = sample - wrist
    
    distances = np.linalg.norm(sample, axis=1)
    max_dist = np.max(distances)
    
    if max_dist > 0:
        sample = sample / max_dist
    
    return sample.flatten()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["landmarks"]

    sample = np.array(data)
    sample = normalize_landmarks(sample)
    X = sample.reshape(1, -1)

    prediction = model.predict(X)
    index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return jsonify({
        "label": labels[index],
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)