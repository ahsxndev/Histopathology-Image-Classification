"""
train.py - Multi-model training pipeline for PCam histopathology classification.

Trains three architectures (Baseline CNN, ResNet50, EfficientNet-B0) on the
PatchCamelyon dataset, evaluates each model, and generates result artifacts
(training curves, confusion matrices, Grad-CAM comparisons, metrics JSON).

Usage:
    python train.py

    Optional arguments (set via environment variables):
        NUM_TRAIN  - number of training samples (default: 20000)
        NUM_VAL    - number of validation samples (default: 5000)
        EPOCHS     - training epochs (default: 10)
        BATCH_SIZE - batch size (default: 64)

Author: Ahsan Zaman
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf

# Configuration via environment variables or defaults
NUM_TRAIN = int(os.environ.get("NUM_TRAIN", 20000))
NUM_VAL = int(os.environ.get("NUM_VAL", 5000))
EPOCHS = int(os.environ.get("EPOCHS", 10))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

TRAIN_X = os.path.join(DATA_DIR, "camelyonpatch_level_2_split_train_x.h5")
TRAIN_Y = os.path.join(DATA_DIR, "camelyonpatch_level_2_split_train_y.h5")
VAL_X = os.path.join(DATA_DIR, "camelyonpatch_level_2_split_valid_x.h5")
VAL_Y = os.path.join(DATA_DIR, "camelyonpatch_level_2_split_valid_y.h5")
TEST_X = os.path.join(DATA_DIR, "camelyonpatch_level_2_split_test_x.h5")
TEST_Y = os.path.join(DATA_DIR, "camelyonpatch_level_2_split_test_y.h5")

# Models to train
MODEL_NAMES = ["baseline_cnn", "resnet50", "efficientnet_b0"]


def check_data_files():
    """Verify all required dataset files exist."""
    required = [TRAIN_X, TRAIN_Y, VAL_X, VAL_Y, TEST_X, TEST_Y]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("ERROR: Missing dataset files:")
        for f in missing:
            print(f"  {f}")
        print()
        print("Please download the PCam .h5 files from:")
        print("  https://drive.google.com/drive/folders/"
              "1gHou49cA1s5vua2V5L98Lt8TiWA3FrKB")
        print(f"and place them in: {DATA_DIR}")
        sys.exit(1)


def main():
    print("=" * 60)
    print("PCam Histopathology Multi-Model Training Pipeline")
    print(f"Author: Ahsan Zaman")
    print("=" * 60)
    print(f"Config: {NUM_TRAIN} train, {NUM_VAL} val, "
          f"{EPOCHS} epochs, batch {BATCH_SIZE}")
    print()

    # Step 0: Check data availability
    check_data_files()

    # Step 1: Load data
    from data_loader import load_h5_data, load_test_data

    print("Loading training and validation data...")
    X_train, y_train, X_val, y_val = load_h5_data(
        TRAIN_X, TRAIN_Y, VAL_X, VAL_Y,
        num_train=NUM_TRAIN, num_val=NUM_VAL,
    )

    print("Loading test data...")
    X_test, y_test = load_test_data(TEST_X, TEST_Y, num_test=NUM_VAL)

    # Step 2: Create augmented dataset
    from data_loader import create_augmented_dataset

    train_ds = create_augmented_dataset(X_train, y_train,
                                        batch_size=BATCH_SIZE, augment=True)
    val_ds = create_augmented_dataset(X_val, y_val,
                                      batch_size=BATCH_SIZE, augment=False)

    # Step 3: Train each model
    from model import build_model

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "gradcam"), exist_ok=True)

    histories = {}
    trained_models = {}
    predictions = {}

    for name in MODEL_NAMES:
        print()
        print("=" * 60)
        print(f"Training: {name}")
        print("=" * 60)

        model = build_model(input_shape=(96, 96, 3), model_name=name)
        model.summary()

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6,
            ),
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1,
        )

        # Save model weights
        weights_path = os.path.join(MODELS_DIR, f"{name}.h5")
        model.save_weights(weights_path)
        print(f"Saved weights to {weights_path}")

        histories[name] = history
        trained_models[name] = model

        # Predict on validation set
        print(f"Evaluating {name} on validation set...")
        y_pred_prob = model.predict(X_val, verbose=0).flatten()
        predictions[name] = y_pred_prob

    # Step 4: Evaluate all models
    from evaluate import (
        compute_metrics,
        generate_classification_report,
        plot_confusion_matrices,
        plot_training_curves,
        plot_roc_curves,
        save_metrics_json,
        format_metrics_table,
    )

    all_metrics = {}
    for name in MODEL_NAMES:
        metrics = compute_metrics(y_val, predictions[name])
        all_metrics[name] = metrics
        generate_classification_report(y_val, predictions[name], name)

    # Save metrics
    save_metrics_json(all_metrics, os.path.join(RESULTS_DIR, "metrics.json"))
    format_metrics_table(all_metrics)

    # Plot training curves
    plot_training_curves(
        histories,
        save_path=os.path.join(RESULTS_DIR, "training_curves.png"),
    )

    # Plot confusion matrices
    plot_confusion_matrices(
        predictions, y_val,
        save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png"),
    )

    # Plot ROC curves
    plot_roc_curves(
        predictions, y_val,
        save_path=os.path.join(RESULTS_DIR, "roc_curves.png"),
    )

    # Step 5: Generate Grad-CAM comparisons
    from gradcam import generate_gradcam_comparison

    print("\nGenerating Grad-CAM comparisons...")

    # Pick sample images (one tumor, one normal) from validation set
    tumor_indices = np.where(y_val == 1)[0]
    normal_indices = np.where(y_val == 0)[0]

    for label, indices, label_name in [
        (1, tumor_indices, "tumor"),
        (0, normal_indices, "normal"),
    ]:
        if len(indices) > 0:
            # Pick a few samples
            for i, sample_idx in enumerate(indices[:3]):
                img = X_val[sample_idx:sample_idx + 1]
                save_path = os.path.join(
                    RESULTS_DIR, "gradcam",
                    f"gradcam_{label_name}_{i + 1}.png",
                )
                generate_gradcam_comparison(
                    img, trained_models, save_path=save_path,
                )

    print()
    print("=" * 60)
    print("Training complete. Results saved to:")
    print(f"  Models:           {MODELS_DIR}")
    print(f"  Training curves:  {RESULTS_DIR}/training_curves.png")
    print(f"  Confusion matrix: {RESULTS_DIR}/confusion_matrix.png")
    print(f"  ROC curves:       {RESULTS_DIR}/roc_curves.png")
    print(f"  Grad-CAM:         {RESULTS_DIR}/gradcam/")
    print(f"  Metrics JSON:     {RESULTS_DIR}/metrics.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
