"""
models/model_builder.py
-----------------------
Defines two model architectures:
  1. build_custom_cnn()   – scratch CNN
  2. build_transfer_model() – ResNet50 / EfficientNetB0 based
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


NUM_CLASSES = 2
IMG_SIZE    = (224, 224, 3)


# ═══════════════════════════════════════════════════════════
# Model 1 — Custom CNN
# ═══════════════════════════════════════════════════════════
def build_custom_cnn(num_classes: int = NUM_CLASSES,
                     input_shape: tuple = IMG_SIZE,
                     dropout_rate: float = 0.5) -> tf.keras.Model:
    """
    A moderately deep CNN with:
      Conv → BN → ReLU → MaxPool blocks
      Global Average Pooling
      Dropout + Dense head
    """
    inputs = layers.Input(shape=input_shape, name="input")

    # ── Block 1 ──
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4), name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4), name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(0.25)(x)

    # ── Block 2 ──
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4), name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4), name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(0.25)(x)

    # ── Block 3 ──
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4), name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4), name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.Dropout(0.25)(x)

    # ── Block 4 ──
    x = layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4), name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)
    x = layers.Dropout(0.3)(x)

    # ── Classification Head ──
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="fc1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate, name="drop_fc")(x)
    x = layers.Dense(128, activation="relu", name="fc2")(x)
    x = layers.Dropout(dropout_rate * 0.6)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="CustomCNN_Thyroid")
    return model


# ═══════════════════════════════════════════════════════════
# Model 2 — Transfer Learning (ResNet50 or EfficientNetB0)
# ═══════════════════════════════════════════════════════════
def build_transfer_model(
    backbone: str = "resnet50",
    num_classes: int = NUM_CLASSES,
    input_shape: tuple = IMG_SIZE,
    dropout_rate: float = 0.4,
    fine_tune_at: int = 100,        # freeze all layers before this index
) -> tf.keras.Model:
    """
    Transfer learning model.

    Parameters
    ----------
    backbone     : 'resnet50' | 'efficientnetb0'
    fine_tune_at : freeze base layers *before* this index (set 0 to freeze all)
    """
    backbone = backbone.lower()

    # ── Load pretrained backbone ──
    if backbone == "resnet50":
        base = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
        preprocess = tf.keras.applications.resnet50.preprocess_input
    elif backbone == "efficientnetb0":
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    else:
        raise ValueError(f"Unknown backbone: {backbone}. Choose 'resnet50' or 'efficientnetb0'.")

    # ── Freeze layers (initial phase) ──
    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    # ── Build model ──
    inputs = layers.Input(shape=input_shape, name="input")

    # NOTE: preprocess inside model for portability
    # For ResNet50, preprocess_input expects [0,255]-range input.
    # Since our loader normalises to [0,1], we scale back here.
    x = layers.Lambda(lambda t: t * 255.0, name="rescale_to_255")(inputs)
    x = layers.Lambda(preprocess,          name="backbone_preprocess")(x)

    x = base(x, training=False)           # keep BN layers in inference mode

    # ── Custom head ──
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="fc1")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(dropout_rate, name="drop1")(x)
    x = layers.Dense(256, activation="relu", name="fc2")(x)
    x = layers.Dropout(dropout_rate * 0.5, name="drop2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name=f"TransferLearning_{backbone.upper()}_Thyroid")
    return model, base        # return base so caller can unfreeze later
