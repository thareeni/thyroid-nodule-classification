"""
training/train_pretrained_model.py
----------------------------------
Trains MobileNetV2 (transfer learning) on thyroid ultrasound dataset.
Optimised for imbalanced datasets (~70% Malignant / 30% Benign).

Plots saved automatically to results/mobilenet/:
  - training_curves.png   (accuracy + loss over epochs)
  - confusion_matrix.png  (test set)
  - classification_report.png (precision / recall / F1 per class)

Usage:
    python training/train_pretrained_model.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'

import numpy as np
import matplotlib
matplotlib.use('Agg')           # non-interactive backend — works on CMD & Colab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout,
    BatchNormalization, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc as sk_auc
)
from pathlib import Path

from utils.data_utils import load_dataset_from_voc, get_dataset_info


# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════
DATASET_ROOT = "dataset/Main_data"
MODEL_SAVE   = "models/pretrained_mobilenet_thyroid_model.h5"
PLOTS_DIR    = "results/mobilenet"
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
EPOCHS_P1    = 15
EPOCHS_P2    = 35
RANDOM_SEED  = 42
CLASS_NAMES  = ["Benign", "Malignant"]

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Dark theme for all plots
DARK_BG   = "#0a0f1e"
CARD_BG   = "#111827"
GRID_COL  = "#1e2d45"
TEXT_COL  = "#e2e8f0"
MUTED_COL = "#64748b"
BLUE      = "#3b82f6"
TEAL      = "#06b6d4"
GREEN     = "#10b981"
RED       = "#ef4444"
PURPLE    = "#8b5cf6"
ORANGE    = "#f59e0b"

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# ═══════════════════════════════════════════════════════════
# Focal Loss
# ═══════════════════════════════════════════════════════════
def focal_loss(gamma=2.0, alpha=0.6):
    def loss_fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce     = -y_true * tf.math.log(y_pred) \
                  - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha  + (1 - y_true) * (1 - alpha)
        focal   = alpha_t * tf.pow(1.0 - p_t, gamma) * bce
        return tf.reduce_mean(focal)
    return loss_fn


# ═══════════════════════════════════════════════════════════
# tf.data pipeline
# ═══════════════════════════════════════════════════════════
def create_tf_dataset(dataset_root, split, batch_size,
                      target_size, augment=False):
    def sample_generator():
        for batch_images, batch_labels in load_dataset_from_voc(
            dataset_root, split, target_size, batch_size
        ):
            for img, lbl in zip(batch_images, batch_labels):
                img = img.astype(np.float32) / 255.0
                yield img, int(lbl)

    output_signature = (
        tf.TensorSpec(shape=(*target_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(),               dtype=tf.int32),
    )

    ds = tf.data.Dataset.from_generator(
        sample_generator, output_signature=output_signature
    )

    if augment:
        @tf.function
        def augment_fn(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image, label
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    ds = (
        ds
        .shuffle(1000, seed=RANDOM_SEED, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=False)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds


# ═══════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════
def build_mobilenet_model(input_shape=(224, 224, 3)):
    base_model = MobileNetV2(
        weights='imagenet', include_top=False, input_shape=input_shape
    )
    base_model.trainable = False

    inputs = Input(shape=input_shape, name="input_image")
    x = tf.keras.layers.Rescaling(scale=255.0, name="rescale")(inputs)
    x = tf.keras.layers.Lambda(
        tf.keras.applications.mobilenet_v2.preprocess_input,
        name="mobilenet_preprocess"
    )(x)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D(name="gap")(x)
    x = BatchNormalization(name="bn1")(x)
    x = Dropout(0.5, name="drop1")(x)
    x = Dense(256, activation='relu', name="fc1")(x)
    x = BatchNormalization(name="bn2")(x)
    x = Dropout(0.3, name="drop2")(x)
    x = Dense(128, activation='relu', name="fc2")(x)
    x = Dropout(0.2, name="drop3")(x)
    outputs = Dense(1, activation='sigmoid', name="prediction")(x)

    model = Model(inputs, outputs, name="MobileNetV2_Thyroid")
    return model, base_model


# ═══════════════════════════════════════════════════════════
# ── PLOT 1: Training History ────────────────────────────
# ═══════════════════════════════════════════════════════════
def save_training_history(history_dict, save_path, phase1_end=None):
    """
    Saves a 2×2 grid of training curves:
      Top-left  : Accuracy (train + val)
      Top-right : Loss     (train + val)
      Bot-left  : AUC      (train + val)
      Bot-right : Precision & Recall (val only)
    """
    plt.rcParams.update({
        'font.family':      'DejaVu Sans',
        'axes.facecolor':   CARD_BG,
        'figure.facecolor': DARK_BG,
        'axes.edgecolor':   GRID_COL,
        'axes.labelcolor':  TEXT_COL,
        'xtick.color':      MUTED_COL,
        'ytick.color':      MUTED_COL,
        'axes.grid':        True,
        'grid.color':       GRID_COL,
        'grid.linewidth':   0.6,
        'axes.spines.top':  False,
        'axes.spines.right':False,
    })

    h        = history_dict
    epochs   = range(1, len(h.get('accuracy', h.get('acc', []))) + 1)
    n        = len(list(epochs))

    fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)
    fig.suptitle(
        "MobileNetV2 Transfer Learning — Training History",
        fontsize=16, fontweight='bold', color=TEXT_COL, y=0.97
    )

    # Phase boundary line helper
    def add_phase_line(ax):
        if phase1_end and phase1_end < n:
            ax.axvline(x=phase1_end, color=ORANGE, linewidth=1.2,
                       linestyle='--', alpha=0.6, label='Phase 1→2')

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)

    # ── Accuracy ──
    ax1 = fig.add_subplot(gs[0, 0])
    train_acc = h.get('accuracy', h.get('acc', []))
    val_acc   = h.get('val_accuracy', h.get('val_acc', []))
    ax1.plot(epochs, train_acc, color=BLUE,  linewidth=2,   label='Train Acc')
    ax1.plot(epochs, val_acc,   color=TEAL,  linewidth=2,   label='Val Acc',
             linestyle='--')
    ax1.fill_between(epochs, train_acc, val_acc,
                     alpha=0.08, color=TEAL)
    add_phase_line(ax1)
    ax1.set_title("Accuracy", color=TEXT_COL, fontsize=12, fontweight='bold', pad=8)
    ax1.set_xlabel("Epoch", fontsize=9)
    ax1.set_ylabel("Accuracy", fontsize=9)
    ax1.legend(fontsize=8, facecolor=CARD_BG, edgecolor=GRID_COL,
               labelcolor=TEXT_COL)
    ax1.set_ylim(0, 1.05)
    # Annotate best val accuracy
    best_val_acc = max(val_acc) if val_acc else 0
    best_ep      = val_acc.index(best_val_acc) + 1 if val_acc else 0
    ax1.annotate(
        f'Best: {best_val_acc:.3f}',
        xy=(best_ep, best_val_acc),
        xytext=(best_ep + max(1, n * 0.05), best_val_acc - 0.06),
        color=TEAL, fontsize=8, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.2)
    )

    # ── Loss ──
    ax2 = fig.add_subplot(gs[0, 1])
    train_loss = h.get('loss', [])
    val_loss   = h.get('val_loss', [])
    ax2.plot(epochs, train_loss, color=PURPLE, linewidth=2,   label='Train Loss')
    ax2.plot(epochs, val_loss,   color=RED,    linewidth=2,   label='Val Loss',
             linestyle='--')
    ax2.fill_between(epochs, train_loss, val_loss,
                     alpha=0.08, color=RED)
    add_phase_line(ax2)
    ax2.set_title("Focal Loss", color=TEXT_COL, fontsize=12, fontweight='bold', pad=8)
    ax2.set_xlabel("Epoch", fontsize=9)
    ax2.set_ylabel("Loss", fontsize=9)
    ax2.legend(fontsize=8, facecolor=CARD_BG, edgecolor=GRID_COL,
               labelcolor=TEXT_COL)
    best_val_loss = min(val_loss) if val_loss else 0
    best_ep2      = val_loss.index(best_val_loss) + 1 if val_loss else 0
    ax2.annotate(
        f'Best: {best_val_loss:.4f}',
        xy=(best_ep2, best_val_loss),
        xytext=(best_ep2 + max(1, n * 0.05), best_val_loss + 0.01),
        color=RED, fontsize=8, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=RED, lw=1.2)
    )

    # ── AUC ──
    ax3 = fig.add_subplot(gs[1, 0])
    train_auc = h.get('auc', [])
    val_auc   = h.get('val_auc', [])
    if train_auc:
        ax3.plot(epochs, train_auc, color=GREEN,  linewidth=2, label='Train AUC')
        ax3.plot(epochs, val_auc,   color=ORANGE, linewidth=2, label='Val AUC',
                 linestyle='--')
        ax3.fill_between(epochs, train_auc, val_auc, alpha=0.08, color=GREEN)
        add_phase_line(ax3)
        ax3.set_ylim(0, 1.05)
        best_vauc = max(val_auc) if val_auc else 0
        best_ep3  = val_auc.index(best_vauc) + 1 if val_auc else 0
        ax3.annotate(
            f'Best: {best_vauc:.4f}',
            xy=(best_ep3, best_vauc),
            xytext=(best_ep3 + max(1, n * 0.05), best_vauc - 0.06),
            color=ORANGE, fontsize=8, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.2)
        )
    ax3.set_title("AUC", color=TEXT_COL, fontsize=12, fontweight='bold', pad=8)
    ax3.set_xlabel("Epoch", fontsize=9)
    ax3.set_ylabel("AUC", fontsize=9)
    ax3.legend(fontsize=8, facecolor=CARD_BG, edgecolor=GRID_COL,
               labelcolor=TEXT_COL)

    # ── Precision & Recall ──
    ax4 = fig.add_subplot(gs[1, 1])
    val_prec = h.get('val_precision', [])
    val_rec  = h.get('val_recall',    [])
    if val_prec:
        ax4.plot(epochs, val_prec, color=BLUE,   linewidth=2, label='Val Precision')
        ax4.plot(epochs, val_rec,  color=RED,    linewidth=2, label='Val Recall',
                 linestyle='--')
        ax4.fill_between(epochs, val_prec, val_rec, alpha=0.07, color=PURPLE)
        add_phase_line(ax4)
        ax4.set_ylim(0, 1.05)
    ax4.set_title("Precision & Recall (Val)", color=TEXT_COL,
                  fontsize=12, fontweight='bold', pad=8)
    ax4.set_xlabel("Epoch", fontsize=9)
    ax4.set_ylabel("Score", fontsize=9)
    ax4.legend(fontsize=8, facecolor=CARD_BG, edgecolor=GRID_COL,
               labelcolor=TEXT_COL)

    # Phase label in bottom-right
    if phase1_end:
        fig.text(0.98, 0.01,
                 f'Phase 1: epochs 1–{phase1_end}  |  Phase 2: epochs {phase1_end+1}–{n}',
                 ha='right', va='bottom', color=ORANGE, fontsize=8, alpha=0.7)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=DARK_BG, edgecolor='none')
    plt.close(fig)
    print(f"[SAVED] Training history → {save_path}")


# ═══════════════════════════════════════════════════════════
# ── PLOT 2: Confusion Matrix ────────────────────────────
# ═══════════════════════════════════════════════════════════
def save_confusion_matrix(y_true, y_pred, save_path,
                          model_name="MobileNetV2"):
    plt.rcParams.update({'font.family': 'DejaVu Sans'})

    cm   = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)   # normalised

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)
    fig.suptitle(
        f"{model_name} — Confusion Matrix  |  Test Set",
        fontsize=14, fontweight='bold', color=TEXT_COL, y=1.01
    )

    titles   = ["Raw Counts", "Normalised (%)"]
    datasets = [cm, cm_n]
    fmts     = ["d", ".2%"]
    cmaps    = [
        sns.color_palette([DARK_BG, BLUE],   as_cmap=True),
        sns.color_palette([DARK_BG, GREEN],  as_cmap=True),
    ]

    for ax, data, title, fmt, cmap in zip(axes, datasets, titles, fmts, cmaps):
        ax.set_facecolor(CARD_BG)
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap=cmap,
            linewidths=2, linecolor=DARK_BG,
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
            annot_kws={"size": 14, "weight": "bold", "color": TEXT_COL},
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, color=TEXT_COL, fontsize=11,
                     fontweight='bold', pad=10)
        ax.set_xlabel("Predicted Label", color=TEXT_COL, fontsize=10, labelpad=8)
        ax.set_ylabel("True Label",      color=TEXT_COL, fontsize=10, labelpad=8)
        ax.tick_params(colors=TEXT_COL, labelsize=10)
        ax.figure.axes[-1].tick_params(colors=MUTED_COL, labelsize=8)

        # Colour each cell border by correctness
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                colour = GREEN if i == j else RED
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1,
                    fill=False, edgecolor=colour,
                    linewidth=2.5, clip_on=True
                ))

    # Summary stats box
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    ppv  = tp / (tp + fp + 1e-8)
    npv  = tn / (tn + fn + 1e-8)
    acc  = (tp + tn) / cm.sum()

    stats_txt = (
        f"Accuracy : {acc:.3f}   |   "
        f"Sensitivity (Recall) : {sens:.3f}   |   "
        f"Specificity : {spec:.3f}   |   "
        f"PPV (Precision) : {ppv:.3f}   |   "
        f"NPV : {npv:.3f}"
    )
    fig.text(0.5, -0.04, stats_txt,
             ha='center', va='top',
             color=MUTED_COL, fontsize=8.5,
             bbox=dict(boxstyle='round,pad=0.4',
                       facecolor=CARD_BG,
                       edgecolor=GRID_COL, linewidth=1))

    fig.tight_layout(pad=2.0)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=DARK_BG, edgecolor='none')
    plt.close(fig)
    print(f"[SAVED] Confusion matrix   → {save_path}")


# ═══════════════════════════════════════════════════════════
# ── PLOT 3: Classification Report ───────────────────────
# ═══════════════════════════════════════════════════════════
def save_classification_report(y_true, y_pred, y_prob,
                                save_path, model_name="MobileNetV2"):
    plt.rcParams.update({'font.family': 'DejaVu Sans'})

    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True
    )

    fig = plt.figure(figsize=(16, 7), facecolor=DARK_BG)
    fig.suptitle(
        f"{model_name} — Evaluation Report  |  Test Set",
        fontsize=14, fontweight='bold', color=TEXT_COL, y=0.98
    )
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── Bar chart: per-class metrics ──
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.set_facecolor(CARD_BG)

    metrics    = ['precision', 'recall', 'f1-score']
    bar_colors = [BLUE, GREEN, PURPLE]
    x          = np.arange(len(CLASS_NAMES))
    width      = 0.22
    offsets    = [-width, 0, width]

    for i, (metric, color) in enumerate(zip(metrics, bar_colors)):
        vals = [report[cls][metric] for cls in CLASS_NAMES]
        bars = ax1.bar(x + offsets[i], vals, width,
                       label=metric.capitalize(),
                       color=color, alpha=0.85,
                       edgecolor=DARK_BG, linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f'{val:.3f}',
                ha='center', va='bottom',
                fontsize=8.5, fontweight='bold', color=TEXT_COL
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(CLASS_NAMES, color=TEXT_COL, fontsize=11)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Score", color=TEXT_COL, fontsize=10)
    ax1.set_title("Per-Class Metrics", color=TEXT_COL,
                  fontsize=11, fontweight='bold', pad=8)
    ax1.tick_params(colors=TEXT_COL)
    ax1.set_facecolor(CARD_BG)
    ax1.spines['bottom'].set_color(GRID_COL)
    ax1.spines['left'].set_color(GRID_COL)
    ax1.grid(axis='y', color=GRID_COL, linewidth=0.6)
    ax1.legend(fontsize=9, facecolor=CARD_BG,
               edgecolor=GRID_COL, labelcolor=TEXT_COL)

    # ── ROC curve ──
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(CARD_BG)

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc     = sk_auc(fpr, tpr)
        ax2.plot(fpr, tpr, color=TEAL,   linewidth=2.5,
                 label=f'ROC (AUC = {roc_auc:.3f})')
        ax2.fill_between(fpr, tpr, alpha=0.12, color=TEAL)
    ax2.plot([0, 1], [0, 1], color=MUTED_COL,
             linewidth=1.2, linestyle='--', label='Random')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.05])
    ax2.set_xlabel("False Positive Rate", color=TEXT_COL, fontsize=9)
    ax2.set_ylabel("True Positive Rate",  color=TEXT_COL, fontsize=9)
    ax2.set_title("ROC Curve", color=TEXT_COL,
                  fontsize=11, fontweight='bold', pad=8)
    ax2.tick_params(colors=TEXT_COL)
    ax2.spines['bottom'].set_color(GRID_COL)
    ax2.spines['left'].set_color(GRID_COL)
    ax2.grid(color=GRID_COL, linewidth=0.5)
    ax2.legend(fontsize=9, facecolor=CARD_BG,
               edgecolor=GRID_COL, labelcolor=TEXT_COL)

    # Overall accuracy footer
    acc = report['accuracy']
    mac = report['macro avg']
    fig.text(
        0.5, 0.01,
        f"Overall Accuracy: {acc:.4f}   |   "
        f"Macro Avg  —  P: {mac['precision']:.4f}  "
        f"R: {mac['recall']:.4f}  F1: {mac['f1-score']:.4f}",
        ha='center', va='bottom',
        color=MUTED_COL, fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4',
                  facecolor=CARD_BG, edgecolor=GRID_COL, linewidth=1)
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=DARK_BG, edgecolor='none')
    plt.close(fig)
    print(f"[SAVED] Classification report → {save_path}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def main():

    # ── Dataset info ──────────────────────────────────────
    print("[INFO] Loading dataset info …")
    train_total, train_counts = get_dataset_info(DATASET_ROOT, "train")
    val_total,   val_counts   = get_dataset_info(DATASET_ROOT, "val")
    test_total,  test_counts  = get_dataset_info(DATASET_ROOT, "test")

    print(f"  Train : {train_total}  "
          f"(Benign={train_counts[0]}, Malignant={train_counts[1]})")
    print(f"  Val   : {val_total}  "
          f"(Benign={val_counts[0]},   Malignant={val_counts[1]})")
    print(f"  Test  : {test_total}  "
          f"(Benign={test_counts[0]},  Malignant={test_counts[1]})")

    total = train_counts[0] + train_counts[1]
    print(f"\n  Class ratio — Benign: {train_counts[0]/total*100:.1f}%  "
          f"Malignant: {train_counts[1]/total*100:.1f}%")

    # ── tf.data pipelines ─────────────────────────────────
    print("\n[INFO] Building tf.data pipelines …")
    train_ds = create_tf_dataset(
        DATASET_ROOT, "train", BATCH_SIZE, IMG_SIZE, augment=True
    )
    val_ds = create_tf_dataset(
        DATASET_ROOT, "val", BATCH_SIZE, IMG_SIZE, augment=False
    )

    steps_per_epoch  = max(1, train_total // BATCH_SIZE)
    validation_steps = max(1, val_total   // BATCH_SIZE)
    print(f"  steps_per_epoch  = {steps_per_epoch}")
    print(f"  validation_steps = {validation_steps}")

    # ── Build model ───────────────────────────────────────
    print("\n[INFO] Building MobileNetV2 model …")
    model, base_model = build_mobilenet_model(input_shape=(*IMG_SIZE, 3))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=focal_loss(gamma=2.0, alpha=0.6),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ],
    )
    model.summary()

    Path("models").mkdir(exist_ok=True)
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)

    # ── PHASE 1 ───────────────────────────────────────────
    print(f"\n[PHASE 1] Training head only — {EPOCHS_P1} epochs …")
    cb1 = [
        EarlyStopping(monitor='val_auc', patience=6,
                      restore_best_weights=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_auc', factor=0.5,
                          patience=3, min_lr=1e-7, mode='max', verbose=1),
        ModelCheckpoint(MODEL_SAVE, monitor='val_auc',
                        save_best_only=True, mode='max', verbose=1),
    ]
    h1 = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS_P1,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=cb1,
        verbose=1,
    )

    # ── PHASE 2 ───────────────────────────────────────────
    print(f"\n[PHASE 2] Fine-tuning top 50 layers — {EPOCHS_P2} epochs …")
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=focal_loss(gamma=2.0, alpha=0.6),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ],
    )
    cb2 = [
        EarlyStopping(monitor='val_auc', patience=8,
                      restore_best_weights=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_auc', factor=0.5,
                          patience=4, min_lr=1e-8, mode='max', verbose=1),
        ModelCheckpoint(MODEL_SAVE, monitor='val_auc',
                        save_best_only=True, mode='max', verbose=1),
    ]

    train_ds = create_tf_dataset(
        DATASET_ROOT, "train", BATCH_SIZE, IMG_SIZE, augment=True
    )
    val_ds = create_tf_dataset(
        DATASET_ROOT, "val", BATCH_SIZE, IMG_SIZE, augment=False
    )

    h2 = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS_P2,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=cb2,
        verbose=1,
    )

    # ── Merge histories ───────────────────────────────────
    merged = {
        k: h1.history[k] + h2.history.get(k, [])
        for k in h1.history
    }

    # ══════════════════════════════════════════════════════
    # SAVE PLOT 1: Training History
    # ══════════════════════════════════════════════════════
    save_training_history(
        merged,
        save_path=f"{PLOTS_DIR}/training_curves.png",
        phase1_end=len(h1.history['loss'])
    )

    # ── Test evaluation ───────────────────────────────────
    print("\n[INFO] Evaluating on test set …")
    X_list, y_list = [], []
    for imgs, lbls in load_dataset_from_voc(
        DATASET_ROOT, "test", IMG_SIZE, BATCH_SIZE
    ):
        X_list.append(imgs.astype(np.float32) / 255.0)
        y_list.append(lbls)
        if sum(len(a) for a in X_list) >= test_total:
            break

    X_test  = np.concatenate(X_list)[:test_total]
    y_test  = np.concatenate(y_list)[:test_total]

    y_prob  = model.predict(X_test, verbose=1).flatten()       # raw sigmoid probs
    y_pred  = (y_prob > 0.5).astype(int)

    results = model.evaluate(X_test, y_test, verbose=1)
    loss_val, acc, auc_val, precision, recall = results
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    print(f"\n{'═'*50}")
    print(f"  Test Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Test AUC       : {auc_val:.4f}")
    print(f"  Test Precision : {precision:.4f}")
    print(f"  Test Recall    : {recall:.4f}")
    print(f"  Test F1-Score  : {f1:.4f}")
    print(f"{'═'*50}")

    # ══════════════════════════════════════════════════════
    # SAVE PLOT 2: Confusion Matrix
    # ══════════════════════════════════════════════════════
    save_confusion_matrix(
        y_test, y_pred,
        save_path=f"{PLOTS_DIR}/confusion_matrix.png",
        model_name="MobileNetV2"
    )

    # ══════════════════════════════════════════════════════
    # SAVE PLOT 3: Classification Report + ROC Curve
    # ══════════════════════════════════════════════════════
    save_classification_report(
        y_test, y_pred, y_prob,
        save_path=f"{PLOTS_DIR}/classification_report.png",
        model_name="MobileNetV2"
    )

    print(f"\n{'═'*50}")
    print(f"  All plots saved to: {PLOTS_DIR}/")
    print(f"    ✅ training_curves.png")
    print(f"    ✅ confusion_matrix.png")
    print(f"    ✅ classification_report.png")
    print(f"  Model saved to: {MODEL_SAVE}")
    print(f"{'═'*50}")


if __name__ == "__main__":
    main()