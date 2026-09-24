"""
app.py
--------------------------------------------------------------
Flask backend for the Sign Language Communication Bridge.
Receives a base64-encoded frame from the frontend, runs it through
the SignRecognizer, and returns the recognized sign + confidence.
--------------------------------------------------------------
Run:
    pip install -r requirements.txt
    python app.py
Server starts at http://localhost:5000
--------------------------------------------------------------
"""

import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

from recognizer import SignRecognizer

app = Flask(__name__)
CORS(app)  # allow the frontend (served separately) to call this API

recognizer = SignRecognizer()


def decode_base64_image(data_url: str) -> np.ndarray:
    """Converts a 'data:image/jpeg;base64,...' string into a BGR OpenCV image."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    img_bytes = base64.b64decode(data_url)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": recognizer.model is not None,
        "num_classes": len(recognizer.labels) if recognizer.labels is not None else 0,
    })


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not payload or "image" not in payload:
        return jsonify({"error": "Missing 'image' field in request body"}), 400

    try:
        image = decode_base64_image(payload["image"])
        if image is None:
            return jsonify({"error": "Could not decode image"}), 400
    except Exception as e:
        return jsonify({"error": f"Invalid image data: {e}"}), 400

    sign, confidence = recognizer.predict(image)

    return jsonify({
        "sign": sign,
        "confidence": round(confidence, 4),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
