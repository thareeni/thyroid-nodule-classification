import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from utils.data_utils import load_dataset_from_voc
import os

# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════
MODEL_PATH = "models/pretrained_mobilenet_thyroid_model.h5"
DATASET_ROOT = "dataset/Main_data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SAVE_DIR = "results/mobilenet"

os.makedirs(SAVE_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════
# 🔥 REBUILD MODEL (same as training)
# ═══════════════════════════════════════════════════
def build_model(input_shape=(224, 224, 3)):
    base_model = tf.keras.applications.MobileNetV2(
        weights=None,   # ❗ no imagenet (we load weights manually)
        include_top=False,
        input_shape=input_shape
    )

    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Rescaling(255.0)(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    x = base_model(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    return model


print("[INFO] Rebuilding model...")
model = build_model()

print("[INFO] Loading weights...")
model.load_weights(MODEL_PATH)   # 🔥 LOAD ONLY WEIGHTS

# ═══════════════════════════════════════════════════
# LOAD TEST DATA
# ═══════════════════════════════════════════════════
print("[INFO] Loading test dataset...")

X_list, y_list = [], []

for imgs, lbls in load_dataset_from_voc(
    DATASET_ROOT, "test", IMG_SIZE, BATCH_SIZE
):
    X_list.append(imgs.astype(np.float32) / 255.0)
    y_list.append(lbls)

X_test = np.concatenate(X_list)
y_test = np.concatenate(y_list)

print(f"[INFO] Total test samples: {len(X_test)}")

# ═══════════════════════════════════════════════════
# PREDICTIONS
# ═══════════════════════════════════════════════════
print("[INFO] Running predictions...")

y_pred = (model.predict(X_test, verbose=1) > 0.5).astype(int).flatten()

# ═══════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n" + "="*50)
print(" CONFUSION MATRIX ")
print("="*50)
print(cm)

print("\n" + "="*50)
print(" CLASSIFICATION REPORT ")
print("="*50)
print(report)

# SAVE
output_file = os.path.join(SAVE_DIR, "evaluation_results.txt")

with open(output_file, "w") as f:
    f.write("CONFUSION MATRIX\n")
    f.write(str(cm))
    f.write("\n\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write(report)

print(f"\n[INFO] Results saved to: {output_file}")
print("[DONE] SUCCESS ✅")