import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import json

# -----------------------------
# CONFIG
# -----------------------------
SEQUENCE_LENGTH = 15

# -----------------------------
# NORMALIZATION (SEQUENCE)
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
    
    return np.array(normalized)

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("landmarks.csv")
data = data.replace("####", np.nan)
data = data.dropna()

X = data.iloc[:, 1:].astype(float).values   # (samples, 945)
y = data.iloc[:, 0].values

# -----------------------------
# RESHAPE + NORMALIZE
# -----------------------------
X = X.reshape(-1, SEQUENCE_LENGTH, 63)
X = np.array([normalize_sequence(sample) for sample in X])

print("X shape:", X.shape)  # should be (samples, 15, 63)

# -----------------------------
# ENCODE LABELS
# -----------------------------
encoder = LabelEncoder()
y = encoder.fit_transform(y)

with open("labels.json", "w") as f:
    json.dump(list(encoder.classes_), f)

# -----------------------------
# SHUFFLE + SPLIT
# -----------------------------
X, y = shuffle(X, y, random_state=42)

if len(set(y)) > 1:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

# -----------------------------
# MODEL (LSTM)
# -----------------------------
model = models.Sequential([
    layers.LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 63)),
    layers.Dropout(0.3),

    layers.LSTM(64),
    layers.Dropout(0.3),

    layers.Dense(32, activation='relu'),
    layers.Dense(len(set(y)), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# TRAIN
# -----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=8,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("model.h5")

print("Model trained and saved as model.h5")