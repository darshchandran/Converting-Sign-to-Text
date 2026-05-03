import csv
import os
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


OUTPUT_FILE = Path("landmarks.csv")
MODEL_FILE = Path("hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
SAMPLES_PER_GESTURE = 100
FEATURE_COLUMNS = [f"{axis}{index}" for index in range(21) for axis in ("x", "y", "z")]
CSV_COLUMNS = ["label", *FEATURE_COLUMNS]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def ensure_csv_header(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_COLUMNS)


def ensure_model_file() -> None:
    if MODEL_FILE.exists():
        return
    print(f"Downloading MediaPipe hand model to {MODEL_FILE}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)


def flatten_landmarks(hand_landmarks) -> list[float]:
    values = []
    for landmark in hand_landmarks:
        values.extend([landmark.x, landmark.y, landmark.z])
    return values


def draw_landmarks(frame, landmarks) -> None:
    height, width = frame.shape[:2]
    for start, end in HAND_CONNECTIONS:
        a = landmarks[start]
        b = landmarks[end]
        cv2.line(
            frame,
            (int(a.x * width), int(a.y * height)),
            (int(b.x * width), int(b.y * height)),
            (55, 211, 153),
            3,
        )

    for landmark in landmarks:
        cv2.circle(
            frame,
            (int(landmark.x * width), int(landmark.y * height)),
            5,
            (78, 193, 242),
            -1,
        )


def append_sample(path: Path, label: str, features: list[float]) -> None:
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([label, *features])


def main() -> None:
    ensure_csv_header(OUTPUT_FILE)
    ensure_model_file()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions and device availability.")

    recording_label: str | None = None
    recorded_count = 0
    frame_index = 0

    base_options = python.BaseOptions(model_asset_path=os.fspath(MODEL_FILE))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.65,
        min_tracking_confidence=0.65,
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a frame from the webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(frame_index * 1000 / max(cap.get(cv2.CAP_PROP_FPS), 30))
            frame_index += 1
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            hand_detected = bool(results.hand_landmarks)
            if hand_detected:
                hand_landmarks = results.hand_landmarks[0]
                draw_landmarks(frame, hand_landmarks)

                if recording_label:
                    append_sample(OUTPUT_FILE, recording_label, flatten_landmarks(hand_landmarks))
                    recorded_count += 1
                    if recorded_count >= SAMPLES_PER_GESTURE:
                        print(f"Saved {SAMPLES_PER_GESTURE} samples for {recording_label}")
                        recording_label = None
                        recorded_count = 0

            status = "Press A-Z to record 100 frames. Press Esc to quit."
            if recording_label:
                status = f"Recording {recording_label}: {recorded_count}/{SAMPLES_PER_GESTURE}"
                if not hand_detected:
                    status += " - show one hand"

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (17, 19, 21), -1)
            cv2.putText(
                frame,
                status,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (55, 211, 153) if hand_detected else (80, 120, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("ISL landmark collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if ord("a") <= key <= ord("z") or ord("A") <= key <= ord("Z"):
                recording_label = chr(key).upper()
                recorded_count = 0
                print(f"Recording {SAMPLES_PER_GESTURE} samples for {recording_label}. Keep the sign steady.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Data saved to {os.fspath(OUTPUT_FILE.resolve())}")


if __name__ == "__main__":
    main()
