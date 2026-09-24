"""
recognizer.py
--------------------------------------------------------------
Extracts hand landmarks from an image using MediaPipe Hands and
classifies the gesture using a pre-trained scikit-learn model.

Landmark-based approach (not raw pixels) is used because it stays
robust across different backgrounds, lighting and hand orientation,
which is a key constraint for the hackathon problem statement.
--------------------------------------------------------------
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "sign_classifier.pkl")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.pkl")

mp_hands = mp.solutions.hands


class SignRecognizer:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            model_complexity=1,
        )

        self.model = None
        self.labels = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.labels = joblib.load(LABELS_PATH)
            print(f"[recognizer] Loaded model with {len(self.labels)} classes.")
        else:
            print("[recognizer] WARNING: No trained model found in /model. "
                  "Run train_model.py after collecting landmark data.")

    # -----------------------------------------------------------
    def extract_landmarks(self, image_bgr):
        """Returns a normalized 63-length feature vector (21 landmarks x,y,z)
        or None if no hand is detected."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
        )

        # Normalize relative to the wrist (landmark 0) so position in
        # frame doesn't affect the prediction, then scale by the hand size.
        wrist = coords[0]
        coords -= wrist
        max_val = np.max(np.linalg.norm(coords, axis=1))
        if max_val > 0:
            coords /= max_val

        return coords.flatten()  # shape (63,)

    # -----------------------------------------------------------
    def predict(self, image_bgr):
        """Returns (sign: str|None, confidence: float)."""
        if self.model is None:
            return None, 0.0

        features = self.extract_landmarks(image_bgr)
        if features is None:
            return None, 0.0

        features = features.reshape(1, -1)
        probs = self.model.predict_proba(features)[0]
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        sign = self.labels[best_idx]

        return sign, confidence
