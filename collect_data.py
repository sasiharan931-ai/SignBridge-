"""
collect_data.py
--------------------------------------------------------------
Fast webcam-based data collector for training the sign classifier.
Skips the "save raw images" step entirely — captures hand landmarks
directly from the webcam and appends them to dataset/landmarks.csv.

Built for speed: collect ~50 signs within a 24h hackathon window.

Controls:
    - Type the current sign label in the terminal, press Enter.
    - Hold your hand steady in frame.
    - Press SPACE to capture a sample (repeat 30-50x per sign for
      good accuracy, varying angle/distance/background each time).
    - Press 'n' to move to a new sign label.
    - Press 'q' to quit and save.
--------------------------------------------------------------
Run:
    python collect_data.py
Output:
    dataset/landmarks.csv  (appended, not overwritten)
--------------------------------------------------------------
"""

import os
import csv
import cv2
import numpy as np
import mediapipe as mp

BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset")
CSV_PATH = os.path.join(DATASET_DIR, "landmarks.csv")
os.makedirs(DATASET_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def normalize_landmarks(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = coords[0]
    coords -= wrist
    max_val = np.max(np.linalg.norm(coords, axis=1))
    if max_val > 0:
        coords /= max_val
    return coords.flatten()  # 63 values


def ensure_csv_header():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"f{i}" for i in range(63)] + ["label"])


def main():
    ensure_csv_header()

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    current_label = input("Enter sign label to start with (e.g. HELLO): ").strip().upper()
    sample_count = 0

    print("\nControls: SPACE = capture | n = new label | q = quit\n")

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            hand_detected = False
            landmarks_row = None

            if result.multi_hand_landmarks:
                hand_detected = True
                hand_landmarks = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks_row = normalize_landmarks(hand_landmarks)

            # HUD overlay
            cv2.putText(frame, f"Label: {current_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples this label: {sample_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            status = "Hand detected" if hand_detected else "No hand"
            color = (0, 255, 0) if hand_detected else (0, 0, 255)
            cv2.putText(frame, status, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Data Collector - SPACE=capture, n=new label, q=quit", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" ") and hand_detected:
                writer.writerow(list(landmarks_row) + [current_label])
                f.flush()
                sample_count += 1
                print(f"  captured sample #{sample_count} for '{current_label}'")

            elif key == ord("n"):
                current_label = input("\nEnter next sign label: ").strip().upper()
                sample_count = 0

            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Data saved to {CSV_PATH}")


if __name__ == "__main__":
    main()