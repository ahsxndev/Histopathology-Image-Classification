import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import time
import os
from PIL import Image

from model import build_model
from gradcam import get_gradcam_heatmap, overlay_gradcam, LAST_CONV_LAYERS


# UI styling
def apply_custom_styles():
    IMAGE_URL = (
        "https://tse3.mm.bing.net/th/id/OIP.wOz-U9xMDdI1GPuEHA3MBgHaEo"
        "?cb=ucfimg2&pid=ImgDet&ucfimg=1&w=474&h=296&rs=1&o=7&rm=3"
    )
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("{IMAGE_URL}");
        background-attachment: fixed;
        background-size: 300% 300%;
        background-position: center center;
        animation: moveBackground 45s ease-in-out infinite;
    }}

    @keyframes moveBackground {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    .result-card {{
        background: rgba(0, 0, 0, 0.75);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-top: 15px;
        margin-bottom: 30px;
        width: 100%;
    }}

    h1, h2, h3, p, span, label,
    div[data-testid="stMarkdownContainer"] > p {{
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.9);
    }}

    [data-testid="stSidebar"] {{
        background-color: rgba(0, 0, 0, 0.85) !important;
        backdrop-filter: blur(15px);
    }}
    </style>
    """, unsafe_allow_html=True)


st.set_page_config(
    page_title="HistoPath AI | Tumor Detection",
    page_icon="🧬",
    layout="wide",
)
apply_custom_styles()


# Available models
AVAILABLE_MODELS = {
    "ResNet50": "resnet50",
    "EfficientNet-B0": "efficientnet_b0",
    "Baseline CNN": "baseline_cnn",
}


@st.cache_resource
def load_selected_model(model_name):
    """Load and cache a model by name."""
    model = build_model(input_shape=(96, 96, 3), model_name=model_name)
    weights_path = os.path.join("models", f"{model_name}.h5")

    # Fallback to legacy weights path
    if not os.path.exists(weights_path):
        weights_path = "histopath_model.h5"

    try:
        model.load_weights(weights_path)
        return model
    except Exception as e:
        st.error(f"Could not load weights for {model_name}: {e}")
        return None


# Grad-CAM function
def make_gradcam_heatmap(img_array, model, model_name):
    """Generate Grad-CAM heatmap using the gradcam module."""
    return get_gradcam_heatmap(img_array, model, model_name=model_name)


def overlay_gradcam_pil(image, heatmap, alpha=0.5):
    """Overlay Grad-CAM on a PIL image."""
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    image_np = np.array(image)
    superimposed_img = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img


# Sidebar
with st.sidebar:
    st.header("⚙️ Analysis Settings")

    selected_display = st.selectbox(
        "Select Model",
        list(AVAILABLE_MODELS.keys()),
        index=0,
    )
    selected_model_name = AVAILABLE_MODELS[selected_display]

    try:
        show_cam = st.toggle("Enable Grad-CAM Overlay", value=True)
    except AttributeError:
        show_cam = st.checkbox("Enable Grad-CAM Overlay", value=True)

    cam_opacity = st.slider("Heatmap Opacity", 0.1, 0.9, 0.5)

    st.divider()
    st.header("📌 Model Specs")
    st.info(
        f"**Architecture:** {selected_display}\n\n"
        "**Task:** Metastatic Tissue Detection\n\n"
        "**Input:** 96x96 PCam Patches"
    )
    st.warning("⚠️ Research Use Only.")
    st.caption("By Ahsan Zaman")


# Load model
model = load_selected_model(selected_model_name)

st.title("🧬 Histopathology Tumor Detection")
st.markdown("### AI-Powered Tissue Analysis")

if model is None:
    st.error("No model weights found. Please run `python train.py` first.")
    st.stop()

uploaded_files = st.file_uploader(
    "Upload tissue patches (PNG/JPG)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.subheader("🤖 Analyzing...")
    st.divider()
    results_list = []

    for file in uploaded_files:
        with st.container():
            col_img, col_cam = st.columns(2)

            image = Image.open(file).convert("RGB")
            img_resized = image.resize((96, 96))
            img_array = np.array(img_resized).astype("float32") / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            start_time = time.time()
            prediction_prob = model.predict(img_array, verbose=0)[0][0]
            inference_duration = time.time() - start_time

            is_malignant = prediction_prob >= 0.5
            label = "MALIGNANT" if is_malignant else "BENIGN"
            conf_score = prediction_prob if is_malignant else 1 - prediction_prob
            theme_color = "#ef4444" if is_malignant else "#22c55e"

            with col_img:
                st.image(image, caption="Original Patch",
                         use_container_width=True)

            with col_cam:
                if show_cam:
                    try:
                        heatmap_data = make_gradcam_heatmap(
                            img_array, model, selected_model_name,
                        )
                        cam_visual = overlay_gradcam_pil(
                            image, heatmap_data, alpha=cam_opacity,
                        )
                        st.image(cam_visual, caption="Grad-CAM Focus",
                                 use_container_width=True)
                    except Exception as e:
                        st.warning(f"Grad-CAM not available: {e}")
                else:
                    st.write("Visualization Disabled")

            st.markdown(f"""
            <div class="result-card" style="border-top: 6px solid {theme_color};">
                <h2 style="color:{theme_color} !important; margin:0;">
                    {label}
                </h2>
                <p style="margin:10px 0 5px 0; font-size: 1.2em;">
                    Confidence: <b>{conf_score * 100:.2f}%</b>
                </p>
                <p style="font-size: 0.9em; opacity: 0.7;">
                    Inference Time: {inference_duration:.3f}s |
                    File: {file.name} |
                    Model: {selected_display}
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(conf_score))

            results_list.append({
                "Filename": file.name,
                "Result": label,
                "Confidence": f"{conf_score * 100:.2f}%",
                "Model": selected_display,
            })
            st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("📊 View Batch Statistics"):
        st.dataframe(results_list)

# Educational footer
st.divider()
st.subheader("📘 Understanding the Technology")

tab1, tab2 = st.tabs(["🔍 What is Grad-CAM?", "🧠 The PCam Dataset"])

with tab1:
    st.markdown("""
    **Gradient-weighted Class Activation Mapping (Grad-CAM)** is a technique
    used to make Convolutional Neural Networks transparent.

    It uses the gradients of the target concept (e.g., "Malignant") flowing
    into the final convolutional layer to produce a coarse localization map
    highlighting important regions in the image.

    * 🔴 **Red/Hot Regions:** Areas heavily used by the model for its
      prediction (e.g., high cell density, irregular nuclei).
    * 🔵 **Blue/Cold Regions:** Areas that had little impact on the decision.
    """)

with tab2:
    st.markdown("""
    The **PatchCamelyon (PCam)** dataset is a clinically relevant benchmark
    for histopathology image classification.

    * It consists of **327,680 color images** (96x96px) extracted from
      histopathologic scans of lymph node sections.
    * Each image is annotated with a binary label indicating the presence
      of metastatic tissue.
    * The challenge is that the metastatic tissue might only occupy a small
      portion of the central region of the patch.
    """)
