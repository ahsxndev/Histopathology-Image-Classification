import numpy as np
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def compute_metrics(y_true, y_pred_prob, threshold=0.5):
    """
    Compute classification metrics for binary classification.

    Args:
        y_true: ground truth labels (0 or 1)
        y_pred_prob: predicted probabilities (float in [0, 1])
        threshold: classification threshold

    Returns:
        dict with accuracy, precision, recall, f1, auc_roc
    """
    y_pred = (y_pred_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_pred_prob)),
    }

    return metrics


def generate_classification_report(y_true, y_pred_prob, model_name,
                                   threshold=0.5):
    """
    Generate and print a full classification report.

    Args:
        y_true: ground truth labels
        y_pred_prob: predicted probabilities
        model_name: name of the model for display
        threshold: classification threshold

    Returns:
        report string
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    report = classification_report(
        y_true, y_pred,
        target_names=["Normal (0)", "Tumor (1)"],
        digits=4,
    )
    print(f"\n{'='*50}")
    print(f"Classification Report: {model_name}")
    print(f"{'='*50}")
    print(report)
    return report


def plot_confusion_matrices(results_dict, y_true, save_path=None):
    """
    Plot side-by-side confusion matrices for all models.

    Args:
        results_dict: dict of {model_name: y_pred_prob}
        y_true: ground truth labels
        save_path: path to save the figure
    """
    n = len(results_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))

    if n == 1:
        axes = [axes]

    display_names = {
        "baseline_cnn": "Baseline CNN",
        "resnet50": "ResNet50",
        "efficientnet_b0": "EfficientNet-B0",
    }

    for idx, (name, y_pred_prob) in enumerate(results_dict.items()):
        y_pred = (y_pred_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
            xticklabels=["Normal", "Tumor"],
            yticklabels=["Normal", "Tumor"],
            annot_kws={"size": 13},
        )
        display_name = display_names.get(name, name)
        axes[idx].set_title(display_name, fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Predicted", fontsize=10)
        axes[idx].set_ylabel("Actual", fontsize=10)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="white")
        print(f"Saved confusion matrices to {save_path}")
    plt.close()


def plot_training_curves(histories_dict, save_path=None):
    """
    Plot training and validation loss/accuracy curves for all models.

    Args:
        histories_dict: dict of {model_name: keras History object or dict}
        save_path: path to save the figure
    """
    n = len(histories_dict)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))

    if n == 1:
        axes = axes.reshape(-1, 1)

    display_names = {
        "baseline_cnn": "Baseline CNN",
        "resnet50": "ResNet50",
        "efficientnet_b0": "EfficientNet-B0",
    }

    colors = {
        "train": "#2563eb",
        "val": "#dc2626",
    }

    for idx, (name, history) in enumerate(histories_dict.items()):
        if hasattr(history, "history"):
            h = history.history
        else:
            h = history

        epochs = range(1, len(h["loss"]) + 1)
        display_name = display_names.get(name, name)

        # Loss
        axes[0][idx].plot(epochs, h["loss"], color=colors["train"],
                          linewidth=1.5, label="Train Loss")
        axes[0][idx].plot(epochs, h["val_loss"], color=colors["val"],
                          linewidth=1.5, label="Val Loss")
        axes[0][idx].set_title(f"{display_name}", fontsize=12,
                               fontweight="bold")
        axes[0][idx].set_xlabel("Epoch")
        axes[0][idx].set_ylabel("Loss")
        axes[0][idx].legend(fontsize=9)
        axes[0][idx].grid(True, alpha=0.3)

        # Accuracy
        axes[1][idx].plot(epochs, h["accuracy"], color=colors["train"],
                          linewidth=1.5, label="Train Acc")
        axes[1][idx].plot(epochs, h["val_accuracy"], color=colors["val"],
                          linewidth=1.5, label="Val Acc")
        axes[1][idx].set_xlabel("Epoch")
        axes[1][idx].set_ylabel("Accuracy")
        axes[1][idx].legend(fontsize=9)
        axes[1][idx].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="white")
        print(f"Saved training curves to {save_path}")
    plt.close()


def plot_roc_curves(results_dict, y_true, save_path=None):
    """
    Plot ROC curves for all models on the same axes.

    Args:
        results_dict: dict of {model_name: y_pred_prob}
        y_true: ground truth labels
        save_path: path to save the figure
    """
    display_names = {
        "baseline_cnn": "Baseline CNN",
        "resnet50": "ResNet50",
        "efficientnet_b0": "EfficientNet-B0",
    }

    colors = ["#2563eb", "#16a34a", "#dc2626"]

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    for idx, (name, y_pred_prob) in enumerate(results_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        auc = roc_auc_score(y_true, y_pred_prob)
        display_name = display_names.get(name, name)
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, color=color, linewidth=1.8,
                label=f"{display_name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="white")
        print(f"Saved ROC curves to {save_path}")
    plt.close()


def save_metrics_json(all_metrics, save_path):
    """
    Save all model metrics to a JSON file.

    Args:
        all_metrics: dict of {model_name: metrics_dict}
        save_path: path to save JSON file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics to {save_path}")


def format_metrics_table(all_metrics):
    """
    Format metrics as a Markdown table for README.

    Args:
        all_metrics: dict of {model_name: metrics_dict}

    Returns:
        Markdown table string
    """
    display_names = {
        "baseline_cnn": "Baseline CNN",
        "resnet50": "ResNet50",
        "efficientnet_b0": "EfficientNet-B0",
    }

    header = "| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |"
    sep = "|-------|----------|-----------|--------|----------|---------|"
    rows = [header, sep]

    for name, metrics in all_metrics.items():
        display_name = display_names.get(name, name)
        row = (
            f"| {display_name} "
            f"| {metrics['accuracy']:.4f} "
            f"| {metrics['precision']:.4f} "
            f"| {metrics['recall']:.4f} "
            f"| {metrics['f1_score']:.4f} "
            f"| {metrics['auc_roc']:.4f} |"
        )
        rows.append(row)

    table = "\n".join(rows)
    print("\nMetrics Table (Markdown):")
    print(table)
    return table
