# training/train_custom_cnn.py
import os
# Disable oneDNN optimizations to prevent memory errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Reduce TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Limit GPU memory growth (if using GPU)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

from utils.data_utils import load_dataset_from_voc, get_dataset_info, CLASS_NAMES
from utils.viz_utils import plot_training_history, evaluate_model
from models.model_builder import build_custom_cnn

# Configure GPU memory growth to prevent OOM
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"[INFO] GPU memory growth setting failed: {e}")
else:
    print("[INFO] No GPU found, using CPU")

# Configuration
DATASET_ROOT  = "dataset/Main_data"
MODEL_SAVE    = "models/custom_cnn_thyroid_model.h5"
PLOTS_DIR     = "results/custom_cnn"
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 8  # Reduced from 16 to save memory
EPOCHS        = 50
NUM_CLASSES   = 1  # Binary classification
RANDOM_SEED   = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def create_tf_dataset(dataset_root, split, batch_size, target_size, augment=False):
    """Create a tf.data.Dataset from the generator"""
    
    def generator():
        for batch_images, batch_labels in load_dataset_from_voc(dataset_root, split, target_size, batch_size):
            # Reshape labels to (batch_size, 1) for binary classification
            batch_labels = batch_labels.reshape(-1, 1)
            yield batch_images, batch_labels
    
    # Create dataset from generator
    output_signature = (
        tf.TensorSpec(shape=(None, *target_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.int32)
    )
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )
    
    # Apply augmentation if needed (simplified to avoid memory issues)
    if augment:
        def augment_image(image, label):
            # Random horizontal flip only (less memory intensive)
            image = tf.image.random_flip_left_right(image)
            return image, label
        
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Repeat the dataset for multiple epochs
    dataset = dataset.repeat()
    
    # Simple batching without unbatch (saves memory)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def load_all_labels(dataset_root, split, target_size, batch_size):
    """Load all labels for class weight calculation"""
    all_labels = []
    total_samples, _ = get_dataset_info(dataset_root, split)
    
    for batch_images, batch_labels in load_dataset_from_voc(dataset_root, split, target_size, batch_size):
        all_labels.extend(batch_labels)
        if len(all_labels) >= total_samples:
            break
    
    return np.array(all_labels)

def main():
    # ─── 1. Get dataset info ──────────────────────────
    print("[INFO] Loading datasets ...")
    
    train_total, train_counts = get_dataset_info(DATASET_ROOT, "train")
    val_total, val_counts = get_dataset_info(DATASET_ROOT, "val")
    test_total, test_counts = get_dataset_info(DATASET_ROOT, "test")
    
    print(f"  Train   : {train_total} samples (Benign: {train_counts[0]}, Malignant: {train_counts[1]})")
    print(f"  Val     : {val_total} samples (Benign: {val_counts[0]}, Malignant: {val_counts[1]})")
    print(f"  Test    : {test_total} samples (Benign: {test_counts[0]}, Malignant: {test_counts[1]})")
    
    # ─── 2. Calculate class weights for imbalanced dataset ──
    print("\n[INFO] Computing class weights...")
    train_labels = load_all_labels(DATASET_ROOT, "train", IMG_SIZE, BATCH_SIZE)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.array([0, 1]),
        y=train_labels
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"  Class weights: Benign={class_weights[0]:.3f}, Malignant={class_weights[1]:.3f}")
    
    # ─── 3. Create TensorFlow datasets ─────────────────
    print("\n[INFO] Creating datasets...")
    train_dataset = create_tf_dataset(DATASET_ROOT, "train", BATCH_SIZE, IMG_SIZE, augment=True)
    val_dataset = create_tf_dataset(DATASET_ROOT, "val", BATCH_SIZE, IMG_SIZE, augment=False)
    test_dataset = create_tf_dataset(DATASET_ROOT, "test", BATCH_SIZE, IMG_SIZE, augment=False)
    
    # ─── 4. Build model ───────────────────────────────
    print("\n[INFO] Building Custom CNN …")
    model = build_custom_cnn(num_classes=1, input_shape=(*IMG_SIZE, 3))
    model.summary()
    
    # ─── 5. Compile with binary crossentropy ───────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    # ─── 6. Callbacks ─────────────────────────────────
    Path("models").mkdir(exist_ok=True)
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        ModelCheckpoint(
            filepath=MODEL_SAVE, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
    ]
    
    # ─── 7. Train ─────────────────────────────────────
    print("\n[INFO] Training Custom CNN …")
    
    steps_per_epoch = max(1, train_total // BATCH_SIZE)
    validation_steps = max(1, val_total // BATCH_SIZE)
    
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Validation steps: {validation_steps}")
    print(f"  Total epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
    )
    
    # ─── 8. Plot training curves ──────────────────────
    plot_training_history(
        history,
        save_path=f"{PLOTS_DIR}/training_curves.png",
        model_name="Custom CNN",
    )
    
    # ─── 9. Evaluate on test set ──────────────────────
    print("\n[INFO] Evaluating on test set …")
    
    # Collect all test predictions and labels
    all_preds = []
    all_labels = []
    test_steps = test_total // BATCH_SIZE
    
    for batch_images, batch_labels in test_dataset.take(test_steps):
        batch_preds = model.predict(batch_images, verbose=0)
        batch_preds_classes = (batch_preds > 0.5).astype(int).flatten()
        all_preds.extend(batch_preds_classes)
        all_labels.extend(batch_labels.numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate accuracy
    test_acc = np.mean(all_preds == all_labels)
    print(f"  Test Accuracy : {test_acc:.4f}")
    
    # Generate evaluation plots
    evaluate_model(all_labels, all_preds, save_dir=PLOTS_DIR, model_name="Custom CNN")
    
    print(f"\n[DONE] Model saved → {MODEL_SAVE}")

if __name__ == "__main__":
    main()