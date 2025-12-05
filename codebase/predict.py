import sys
import os
import torch
import torch.nn.functional as F
from PIL import Image

# Fix import paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codebase.models import ResNetModel
from codebase.utils import get_val_test_transforms


LABEL_MAP = {0: "BENIGN", 1: "MALIGNANT"}


def load_model(checkpoint_path="checkpoints/best_model.pth"):
    """Load the trained model and weights."""
    print(f"[INFO] Loading model from {checkpoint_path}...")

    model_config = {
        "num_classes": 2,
        "pretrained": False,  # Don't load pretrained weights, we'll load our trained weights
        "resnet_version": "resnet18"
    }

    model = ResNetModel(model_config)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print("[INFO] Model loaded successfully!")
    return model


def preprocess_image(image_path):
    """Apply the same transforms as validation/test."""
    transform = get_val_test_transforms()

    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # (1, 3, 224, 224)


def bayesian_adjustment(prob, prior=0.5):
    """
    Optional Bayesian confidence adjustment.
    Uses simple P(y|x) blending with a prior.
    """
    posterior = (prob * prior) / (prob * prior + (1 - prob) * (1 - prior))
    return posterior


def predict(image_path):
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return

    # Load model & image
    model = load_model()
    input_tensor = preprocess_image(image_path)

    # Forward pass
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]

    predicted_class = torch.argmax(probs).item()
    confidence = probs[predicted_class].item()

    # Bayesian adjusted confidence
    adjusted_conf = bayesian_adjustment(confidence)

    # ---- RULE-BASED LAYER: FLAG LOW CONFIDENCE ----
    CONF_THRESHOLD = 0.75
    flag_for_review = adjusted_conf < CONF_THRESHOLD

    # Display results
    print("\n========== PREDICTION RESULT ==========")
    print(f"Image: {image_path}")
    print(f"Prediction: {LABEL_MAP[predicted_class]}")
    print(f"Model confidence: {confidence*100:.2f}%")
    print(f"Bayesian adjusted confidence: {adjusted_conf*100:.2f}%")
    print("========================================\n")

    if flag_for_review:
        print(f"\n[AGENT RULE] Confidence below {CONF_THRESHOLD*100:.0f}% → FLAGGING for dermatologist review.")
    else:
        print(f"\n[AGENT RULE] Confidence above {CONF_THRESHOLD*100:.0f}% → Model decision accepted; no review needed.")
    print("========================================\n")

    return predicted_class, confidence, adjusted_conf


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m codebase.predict <image_path>")
    else:
        predict(sys.argv[1])
