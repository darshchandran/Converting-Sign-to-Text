from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

SEQUENCE_LENGTH = 15

# -----------------------------
# LOAD MODEL (IMPORTANT FIX)
# -----------------------------
model = tf.keras.models.load_model("model.h5")

# -----------------------------
# LOAD LABELS
# -----------------------------
with open("labels.json") as f:
    labels = json.load(f)

# -----------------------------
# NORMALIZE SEQUENCE (MATCH TRAINING)
# -----------------------------
def normalize_sequence(sequence):
    sequence = sequence.reshape(SEQUENCE_LENGTH, 21, 3)

    normalized = []
    for frame in sequence:
        wrist = frame[0]
        frame = frame - wrist

        max_dist = np.max(np.linalg.norm(frame, axis=1))
        if max_dist > 0:
            frame = frame / max_dist

        normalized.append(frame.flatten())

    return np.array(normalized)  # (15, 63)

# -----------------------------
# PREDICT ROUTE
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["landmarks"]

    sample = np.array(data)

    # DEBUG (optional)
    print("Received:", sample.shape)

    # reshape → (15, 63)
    sample = sample.reshape(SEQUENCE_LENGTH, 63)

    # normalize
    sample = normalize_sequence(sample)

    # final shape → (1, 15, 63)
    X = sample.reshape(1, SEQUENCE_LENGTH, 63)

    print("Model input:", X.shape)

    prediction = model.predict(X)

    index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return jsonify({
        "label": labels[index],
        "confidence": confidence
    })

# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)