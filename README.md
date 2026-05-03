# Real-time Indian Sign Language Translator

This is a single-page webcam app for recognizing ISL fingerspelling gestures from MediaPipe hand landmarks. It includes tools to collect landmark samples, train a classifier, export a TensorFlow.js model, build a sentence, and speak the sentence with the Web Speech API.

## Files

- `index.html` - browser UI, webcam feed, landmark overlay, TF.js prediction, sentence builder, speech.
- `data_collection.py` - OpenCV + MediaPipe webcam collector that saves `landmarks.csv`.
- `model_training.py` - trains and evaluates a scikit-learn RandomForest, then exports a browser-ready TF.js model.
- `landmarks.csv` - generated training data.
- `tfjs_model/model.json` - generated model loaded by the web app.

## Install Python Dependencies

Use Python 3.10 or 3.11 if possible.

```bash
py -m pip install opencv-python mediapipe pandas numpy scikit-learn joblib tensorflow tensorflowjs
```

On Windows, use `py` instead of `python` if `python` is not on your PATH.

## Collect Gesture Data

Run:

```bash
py data_collection.py
```

The webcam window will open with MediaPipe landmarks drawn over your hand.

- Press any letter key `A-Z` to record 100 frames for that gesture.
- Keep one hand clearly visible while recording.
- Repeat for every ISL fingerspelling class you want to support.
- Press `Esc` to quit.

The script writes rows to `landmarks.csv` with:

```text
label, x0, y0, z0, ..., x20, y20, z20
```

Each row has one label plus 63 landmark features.

## Train and Export

Run:

```bash
py model_training.py
```

The script:

1. Loads `landmarks.csv`.
2. Normalizes landmarks relative to the wrist and scales them by hand size.
3. Trains a scikit-learn `RandomForestClassifier`.
4. Prints a classification report and confusion matrix.
5. Saves the RandomForest to `random_forest_model.joblib`.
6. Trains a small Keras classifier on the same normalized data.
7. Exports the browser model to `tfjs_model/model.json` and writes `tfjs_model/labels.json`.

TensorFlow.js cannot load a scikit-learn RandomForest directly, so the RandomForest is used for the required classical ML evaluation while the Keras model is exported for live browser inference.

## Run the Web App

Because browsers require a secure origin or localhost for webcam access, serve the folder instead of opening the file directly:

```bash
py -m http.server 8000
```

Open:

```text
http://localhost:8000
```

Allow camera access when prompted. The app loads:

```text
./tfjs_model/model.json
./tfjs_model/labels.json
```

## Browser Controls

- Current sign shows the top model prediction.
- Confidence bar shows the prediction probability.
- Add to sentence appends the current prediction.
- Space adds a space to the sentence.
- Speak reads the sentence aloud with the Web Speech API.
- Clear empties the sentence and stops speech.

## Data Quality Tips

- Collect examples in different lighting conditions and hand positions.
- Keep labels balanced; collect roughly the same number of samples per class.
- Use a plain background when possible.
- Re-record confusing classes with more varied examples.
- Train with all 26 letters for full A-Z fingerspelling support.

## Troubleshooting

- If the web app says the model is missing, run `py model_training.py` and refresh.
- If webcam access fails, use `http://localhost:8000` and allow camera permission.
- If MediaPipe cannot detect a hand, improve lighting and keep the hand fully inside the frame.
- If `tensorflowjs` fails to install, try a fresh virtual environment with Python 3.10 or 3.11.
