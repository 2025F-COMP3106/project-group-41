"""
Flask API server for skin lesion prediction.

Run from project root:
    python server.py

Exposes POST /predict  (accepts multipart file upload)
"""

import os
import sys
import tempfile

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from codebase.models import ResNetModel
from codebase.utils import get_val_test_transforms

LABEL_MAP = {0: "BENIGN", 1: "MALIGNANT"}
CHECKPOINT_PATH = "checkpoints/best_model.pth"
CONF_THRESHOLD = 0.75

app = Flask(__name__)
CORS(app)

_model = None


def get_model():
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(CHECKPOINT_PATH):
        return None

    model_config = {
        "num_classes": 2,
        "pretrained": False,
        "resnet_version": "resnet18",
    }
    model = ResNetModel(model_config)
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    _model = model
    return _model


def preprocess_pil_image(pil_img):
    transform = get_val_test_transforms()
    img = pil_img.convert("RGB")
    return transform(img).unsqueeze(0)


def bayesian_adjustment(prob, prior=0.5):
    posterior = (prob * prior) / (prob * prior + (1 - prob) * (1 - prior))
    return posterior


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send an image as 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    model = get_model()
    if model is None:
        return jsonify({
            "error": (
                "Model checkpoint not found. "
                "Train the model first with: python codebase/main.py"
            )
        }), 503

    try:
        pil_img = Image.open(file.stream)
    except Exception:
        return jsonify({"error": "Could not read image. Make sure it is a valid image file."}), 400

    input_tensor = preprocess_pil_image(pil_img)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]

    predicted_class = torch.argmax(probs).item()
    confidence = probs[predicted_class].item()
    adjusted_confidence = bayesian_adjustment(confidence)
    flag_for_review = adjusted_confidence < CONF_THRESHOLD

    return jsonify({
        "prediction": LABEL_MAP[predicted_class],
        "confidence": round(confidence, 4),
        "adjusted_confidence": round(adjusted_confidence, 4),
        "flag_for_review": flag_for_review,
    })


@app.route("/health", methods=["GET"])
def health():
    model_ready = os.path.exists(CHECKPOINT_PATH)
    return jsonify({"status": "ok", "model_ready": model_ready})


if __name__ == "__main__":
    print("Starting prediction server on http://127.0.0.1:8000")
    print(f"Model checkpoint: {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        print("[WARNING] Checkpoint not found! Train the model first: python codebase/main.py")
    app.run(host="127.0.0.1", port=8000, debug=False)
