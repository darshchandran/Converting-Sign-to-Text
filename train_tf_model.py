import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models

import numpy as np

def normalize_landmarks(sample):
    sample = sample.reshape(21, 3)
    
    # Step 1: wrist as origin
    wrist = sample[0]
    sample = sample - wrist
    
    # Step 2: scale normalization
    distances = np.linalg.norm(sample, axis=1)
    max_dist = np.max(distances)
    
    if max_dist > 0:
        sample = sample / max_dist
    
    return sample.flatten()

# Load dataset
data = pd.read_csv("landmarks.csv")
data = data.replace("####", np.nan)
data = data.dropna()
# Split features and labels
# Split correctly
X = data.iloc[:, 1:].astype(float).values   # force numeric
y = data.iloc[:, 0].values
 
X = np.array([normalize_landmarks(sample) for sample in X])

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Save labels
import json
with open("labels.json", "w") as f:
    json.dump(list(encoder.classes_), f) 

from sklearn.utils import shuffle

X, y = shuffle(X, y, random_state=42)

# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)




# Build model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(set(y)), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save model
model.save("model.h5")

print("✅ Model trained and saved as model.h5")

