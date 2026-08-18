import tensorflow as tf
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


# Map model names to their last convolutional layer names
LAST_CONV_LAYERS = {
    "baseline_cnn": "activation_3",       # last ReLU in block 4
    "resnet50": "conv5_block3_out",       # standard ResNet50 final conv
    "efficientnet_b0": "top_conv",        # EfficientNet top conv layer
}


def get_gradcam_heatmap(img_array, model, last_conv_layer_name=None,
                        model_name=None):
    """
    Generate Grad-CAM heatmap for a single image array.

    Args:
        img_array: preprocessed image tensor, shape (1, 96, 96, 3)
        model: compiled Keras model
        last_conv_layer_name: name of last conv layer (auto-detected if
                              model_name is provided)
        model_name: one of 'baseline_cnn', 'resnet50', 'efficientnet_b0'

    Returns:
        heatmap: numpy array, normalized to [0, 1]
    """
    if last_conv_layer_name is None and model_name is not None:
        last_conv_layer_name = LAST_CONV_LAYERS.get(model_name)

    if last_conv_layer_name is None:
        raise ValueError("Provide either last_conv_layer_name or model_name")

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy()


def overlay_gradcam(img, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on the original image.

    Args:
        img: original image array, shape (H, W, 3), values in [0, 1]
        heatmap: Grad-CAM heatmap, arbitrary size
        alpha: overlay opacity

    Returns:
        superimposed_img: uint8 numpy array
    """
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    img_uint8 = np.uint8(255 * img) if img.max() <= 1.0 else np.uint8(img)
    superimposed = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_color,
                                   alpha, 0)
    return superimposed


def display_gradcam(img, heatmap, overlay_img, save_path=None):
    """
    Display original image, heatmap, and overlay side by side.

    Args:
        img: original image
        heatmap: raw Grad-CAM heatmap
        overlay_img: superimposed result
        save_path: if provided, save figure instead of showing
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay_img)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved Grad-CAM visualization to {save_path}")
    else:
        plt.show()
    plt.close()


def generate_gradcam_comparison(img_array, models_dict, save_path=None):
    """
    Generate side-by-side Grad-CAM comparison across multiple models.

    Args:
        img_array: single image, shape (1, 96, 96, 3), normalized to [0,1]
        models_dict: dict of {model_name: model_instance}
        save_path: path to save the comparison image

    Returns:
        matplotlib figure
    """
    original = img_array[0]  # remove batch dimension
    n_models = len(models_dict)

    fig, axes = plt.subplots(1, n_models + 1, figsize=(4 * (n_models + 1), 4))

    # Original image
    axes[0].imshow(original)
    axes[0].set_title("Original Patch", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    for idx, (name, model) in enumerate(models_dict.items()):
        try:
            heatmap = get_gradcam_heatmap(img_array, model, model_name=name)
            overlay = overlay_gradcam(original, heatmap, alpha=0.5)
            axes[idx + 1].imshow(overlay)
            # Format title nicely
            display_name = {
                "baseline_cnn": "Baseline CNN",
                "resnet50": "ResNet50",
                "efficientnet_b0": "EfficientNet-B0",
            }.get(name, name)
            axes[idx + 1].set_title(f"{display_name} CAM",
                                     fontsize=11, fontweight="bold")
        except Exception as e:
            axes[idx + 1].text(0.5, 0.5, f"Error:\n{str(e)[:40]}",
                               ha="center", va="center", fontsize=8)
            axes[idx + 1].set_title(name)
        axes[idx + 1].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="white")
        print(f"Saved Grad-CAM comparison to {save_path}")
    plt.close()
    return fig
