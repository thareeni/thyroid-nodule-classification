"""
predict.py
----------
Standalone prediction script.
Handles both sigmoid output (MobileNetV2) and softmax output (Custom CNN).

Usage:
    python predict.py --image dataset/Main_data/JPEGImages/000001.jpg
    python predict.py --image dataset/Main_data/JPEGImages/000001.jpg --model models/custom_cnn_thyroid_model.h5
    python predict.py --image dataset/Main_data/JPEGImages/000001.jpg --model models/pretrained_mobilenet_thyroid_model.h5
"""

import argparse
import numpy as np
import tensorflow as tf
import cv2
import sys
import os

# Suppress TF logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path


CLASS_NAMES   = ["Benign Thyroid Nodule", "Malignant Thyroid Nodule"]
IMG_SIZE      = (224, 224)
DEFAULT_MODEL = "models/pretrained_mobilenet_thyroid_model.h5"


# ─────────────────────────────────────────────
# Image preprocessing
# ─────────────────────────────────────────────
def preprocess_image(image_path: str) -> np.ndarray:
    """Load, resize, and normalise image to [0, 1]."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)   # (1, 224, 224, 3)


# ─────────────────────────────────────────────
# Prediction — handles sigmoid AND softmax output
# ─────────────────────────────────────────────
def predict(image_path: str, model_path: str = DEFAULT_MODEL):
    """
    Predict class for a single ultrasound image.

    Automatically detects output type:
      - sigmoid (shape (1,1))  → MobileNetV2 / EfficientNet
      - softmax (shape (1,2))  → Custom CNN
    """
    # ── Load model ──
    print(f"[INFO] Loading model : {model_path}")
    model = tf.keras.models.load_model(
        model_path,
        compile=False,   # skip recompile, we only need inference
    )

    # ── Preprocess image ──
    print(f"[INFO] Processing    : {image_path}")
    x = preprocess_image(image_path)

    # ── Run inference ──
    raw_output = model.predict(x, verbose=0)   # shape: (1,1) or (1,2)
    output_shape = raw_output.shape

    if output_shape[-1] == 1:
        # ── Sigmoid output (MobileNetV2 / EfficientNet) ──
        malignant_prob = float(raw_output[0][0])
        benign_prob    = 1.0 - malignant_prob
        probs = [benign_prob, malignant_prob]

    elif output_shape[-1] == 2:
        # ── Softmax output (Custom CNN) ──
        probs = raw_output[0].tolist()

    else:
        raise ValueError(f"Unexpected model output shape: {output_shape}")

    idx        = int(np.argmax(probs))
    label      = CLASS_NAMES[idx]
    confidence = float(probs[idx])
    all_probs  = dict(zip(CLASS_NAMES, probs))

    return label, confidence, all_probs


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Thyroid Nodule Classifier",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--image", required=True,
        help=(
            "Path to ultrasound image.\n"
            "Example: dataset/Main_data/JPEGImages/000001.jpg"
        ),
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=(
            f"Path to trained .h5 model file.\n"
            f"Options:\n"
            f"  models/pretrained_mobilenet_thyroid_model.h5  (best)\n"
            f"  models/custom_cnn_thyroid_model.h5\n"
            f"Default: {DEFAULT_MODEL}"
        ),
    )
    args = parser.parse_args()

    # ── Validate paths ──
    if not Path(args.image).exists():
        print(f"\n[ERROR] Image not found: {args.image}")
        print("  Make sure you give the correct path to a real image file.")
        print("  Example:")
        print("    python predict.py --image dataset/Main_data/JPEGImages/000001.jpg")
        sys.exit(1)

    if not Path(args.model).exists():
        print(f"\n[ERROR] Model not found: {args.model}")
        print("  Available models in models/ folder:")
        for f in Path("models").glob("*.h5"):
            print(f"    {f}")
        sys.exit(1)

    # ── Predict ──
    label, confidence, all_probs = predict(args.image, args.model)

    # ── Display result ──
    print("\n" + "═" * 50)
    print("      Thyroid Nodule Classification Result")
    print("═" * 50)

    icon = "⚠  MALIGNANT" if "Malignant" in label else "✓  BENIGN"
    print(f"\n  Result     : {icon}")
    print(f"  Confidence : {confidence * 100:.2f}%")
    print(f"  Model used : {Path(args.model).name}")

    print("\n  Class Probabilities:")
    print("  " + "─" * 46)
    for cls, prob in all_probs.items():
        bar    = "█" * int(prob * 35)
        marker = " ◄" if cls == label else ""
        print(f"  {cls:<28} {prob*100:5.1f}%  {bar}{marker}")

    print("═" * 50)
    print()

    # Medical disclaimer
    print("  ⚠  DISCLAIMER: For research use only.")
    print("     Always consult a qualified medical professional.\n")


if __name__ == "__main__":
    main()
