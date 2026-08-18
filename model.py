import tensorflow as tf
from tensorflow.keras import layers, models


def build_baseline_cnn(input_shape=(96, 96, 3)):
    """
    Custom 4-layer CNN baseline for histopathology binary classification.

    Architecture:
        4x (Conv2D -> BatchNorm -> ReLU -> MaxPool)
        GlobalAveragePooling -> Dense(128) -> Dropout -> Dense(1, sigmoid)

    Total trainable parameters: ~290K
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # Block 4
        layers.Conv2D(256, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # Classifier head
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_resnet50(input_shape=(96, 96, 3)):
    """
    ResNet50 with ImageNet pretrained weights for transfer learning.

    Strategy:
        - Freeze base ResNet50 convolutional layers
        - Add custom classification head
        - Use a lower learning rate (1e-4) for stable fine-tuning

    Total trainable parameters: ~270K (head only)
    """
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base_model.trainable = False  # freeze pretrained layers

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_efficientnet_b0(input_shape=(96, 96, 3)):
    """
    EfficientNet-B0 with ImageNet pretrained weights.

    EfficientNet uses compound scaling (depth, width, resolution)
    for better parameter efficiency compared to ResNet.

    Strategy:
        - Freeze base EfficientNetB0 convolutional layers
        - Add custom classification head
        - Use a lower learning rate (1e-4)

    Total trainable parameters: ~270K (head only)
    """
    base_model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base_model.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Legacy compatibility wrapper
def build_model(input_shape=(96, 96, 3), model_name="resnet50"):
    """
    Unified model builder. Returns a compiled Keras model.

    Args:
        input_shape: tuple, input image dimensions
        model_name: one of 'baseline_cnn', 'resnet50', 'efficientnet_b0'
    """
    builders = {
        "baseline_cnn": build_baseline_cnn,
        "resnet50": build_resnet50,
        "efficientnet_b0": build_efficientnet_b0,
    }
    if model_name not in builders:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from {list(builders.keys())}"
        )
    return builders[model_name](input_shape=input_shape)