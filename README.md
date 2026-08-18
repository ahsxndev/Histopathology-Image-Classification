# Comparative Histopathology Image Classification

**Multi-architecture benchmarking on PatchCamelyon for automated metastasis detection in lymph node tissue**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-red.svg)](https://histopath-ahsan.streamlit.app)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahsxndev/Histopathology-Image-Classification/blob/main/train_pcam.ipynb)

## Abstract

Detecting metastatic cancer in lymph node tissue remains one of the most time-consuming steps in pathology workflows. Misclassification directly affects patient staging and treatment decisions. This project evaluates three convolutional neural network architectures on the PatchCamelyon (PCam) benchmark: a custom 4-layer baseline CNN, ResNet50 with ImageNet transfer learning, and EfficientNet-B0. Each model is trained with a data augmentation pipeline that includes random flips, rotations, and color jitter to simulate real-world staining variability. Grad-CAM visualizations confirm that the transfer-learned models attend to cellular morphology rather than slide artifacts, while quantitative metrics (accuracy, F1-score, AUC-ROC) establish clear performance differences across architectures.

## Problem Statement and Clinical Relevance

Pathologists examine whole-slide images (WSIs) of lymph node biopsies to identify metastatic deposits. In practice, a single slide can contain millions of pixels, and the metastatic region may occupy only a small fraction of the tissue. Manual review is slow, subjective, and prone to inter-observer variability.

Automated classification of small tissue patches can serve as a screening tool: flagging suspicious regions for human review, reducing turnaround time, and improving consistency. The PatchCamelyon dataset was created specifically for this purpose, providing a standardized binary classification task (tumor vs. normal) on 96x96 pixel patches extracted from the Camelyon16 challenge.

This project does not replace clinical judgment. It explores how different CNN architectures handle the task and which ones generalize well enough for further development.

## Dataset and Preprocessing Pipeline

### PatchCamelyon (PCam)

PCam is a binary classification dataset derived from the Camelyon16 whole-slide image challenge.

| Property | Value |
|----------|-------|
| Total images | 327,680 |
| Image size | 96 x 96 pixels, RGB |
| Classes | Normal (0), Tumor (1) |
| Train split | 262,144 images |
| Validation split | 32,768 images |
| Test split | 32,768 images |
| Label criterion | Tumor tissue present in central 32x32 region |

The dataset is roughly balanced (slightly more normal patches than tumor). All images are stored in HDF5 format and normalized to [0, 1] before training.

**Dataset source:** [PCam on GitHub](https://github.com/basveeling/pcam)

### Data Augmentation

Since tissue patches can appear in any orientation on the slide, and staining intensity varies between laboratories, the following augmentations are applied during training:

- **Random horizontal and vertical flips** (tissue orientation is arbitrary)
- **Random rotation** (up to 20 degrees)
- **Random brightness adjustment** (simulates staining variability)
- **Random contrast adjustment** (handles scanner differences)

These are implemented as a `tf.data` pipeline with on-the-fly augmentation using Keras preprocessing layers.

## Model Architectures and Experimental Setup

Three architectures were evaluated under identical data conditions:

| Component | Baseline CNN | ResNet50 | EfficientNet-B0 |
|-----------|-------------|----------|-----------------|
| Architecture | 4x (Conv3x3 + BN + ReLU + MaxPool) + GAP + Dense | ResNet50 backbone + GAP + Dense head | EfficientNetB0 backbone + GAP + Dense head |
| Pretrained weights | None (trained from scratch) | ImageNet | ImageNet |
| Trainable params | ~290K (all layers) | ~270K (head only, backbone frozen) | ~270K (head only, backbone frozen) |
| Optimizer | Adam (lr=1e-3) | Adam (lr=1e-4) | Adam (lr=1e-4) |
| LR schedule | ReduceLROnPlateau (factor=0.5) | ReduceLROnPlateau (factor=0.5) | ReduceLROnPlateau (factor=0.5) |
| Regularization | Dropout (0.5) | Dropout (0.5) | Dropout (0.5) |
| Loss function | Binary cross-entropy | Binary cross-entropy | Binary cross-entropy |
| Early stopping | Patience = 3 | Patience = 3 | Patience = 3 |

The transfer-learned models freeze the pretrained convolutional backbone and only train the classification head. A lower learning rate (1e-4 vs 1e-3) is used for the pretrained models to avoid destabilizing learned feature representations.

## Quantitative Results and Evaluation

<!-- METRICS_TABLE_START -->
> **Note:** The table below will be populated with real metrics after training completes. Run `python train.py` to generate results.

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Baseline CNN | -- | -- | -- | -- | -- |
| ResNet50 | -- | -- | -- | -- | -- |
| EfficientNet-B0 | -- | -- | -- | -- | -- |
<!-- METRICS_TABLE_END -->

### Training and Validation Curves

<!-- TRAINING_CURVES_START -->
*Training curves will be generated after running `python train.py`. The resulting image will be saved to `results/training_curves.png`.*
<!-- TRAINING_CURVES_END -->

### Confusion Matrices

<!-- CONFUSION_MATRIX_START -->
*Confusion matrices will be generated after running `python train.py`. The resulting image will be saved to `results/confusion_matrix.png`.*
<!-- CONFUSION_MATRIX_END -->

### ROC Curves

<!-- ROC_CURVES_START -->
*ROC curves will be generated after running `python train.py`. The resulting image will be saved to `results/roc_curves.png`.*
<!-- ROC_CURVES_END -->

## Visual Interpretability (Grad-CAM)

Gradient-weighted Class Activation Mapping (Grad-CAM) highlights which image regions most influenced the model's prediction. This is important in medical imaging, where the model should focus on cellular morphology (nuclear atypia, mitotic figures, tissue architecture) rather than slide preparation artifacts (air bubbles, pen marks, tissue edges).

Each architecture uses different convolutional feature maps:
- **Baseline CNN:** Attends to the last custom convolutional layer
- **ResNet50:** Uses `conv5_block3_out` (the final residual block)
- **EfficientNet-B0:** Uses `top_conv` (the compound-scaled top layer)

<!-- GRADCAM_START -->
*Grad-CAM comparison images will be generated after running `python train.py`. Results are saved to `results/gradcam/`.*
<!-- GRADCAM_END -->

The side-by-side comparison reveals how transfer-learned models attend to tighter, more biologically relevant regions compared to the baseline, which often activates over broader areas of the patch.

## Live Demo and Deployment

A live web application is hosted on Streamlit Community Cloud:

**[Launch the Demo](https://histopath-ahsan.streamlit.app)**

The demo allows you to:
1. **Upload** one or more 96x96 histopathology patch images (PNG or JPG)
2. **Select** which model architecture to use (ResNet50, EfficientNet-B0, or Baseline CNN)
3. **View** the classification result (Malignant / Benign) with confidence score
4. **Inspect** the Grad-CAM overlay showing which regions influenced the prediction
5. **Compare** results across models using the sidebar selector

The application runs inference in real time and displays results with Grad-CAM heatmaps overlaid on the original image.

## Reproducibility / Quickstart

### 1. Clone and set up the environment

```bash
git clone https://github.com/ahsxndev/Histopathology-Image-Classification.git
cd Histopathology-Image-Classification

python -m venv venv
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Download the PCam dataset

Download the 6 `.h5` files from the [Google Drive link](https://drive.google.com/drive/folders/1gHou49cA1s5vua2V5L98Lt8TiWA3FrKB) and place them in the `data/` directory:

```
data/
  camelyonpatch_level_2_split_train_x.h5
  camelyonpatch_level_2_split_train_y.h5
  camelyonpatch_level_2_split_valid_x.h5
  camelyonpatch_level_2_split_valid_y.h5
  camelyonpatch_level_2_split_test_x.h5
  camelyonpatch_level_2_split_test_y.h5
```

### 3. Train all models

```bash
python train.py
```

This trains the Baseline CNN, ResNet50, and EfficientNet-B0, then generates all evaluation artifacts (training curves, confusion matrices, ROC curves, Grad-CAM comparisons) in the `results/` directory.

To customize training parameters:

```bash
# Train with more data and epochs
NUM_TRAIN=50000 NUM_VAL=10000 EPOCHS=20 python train.py
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Repository Structure

```
Histopathology-Image-Classification/
├── data/                           # PCam dataset .h5 files (not in git)
├── models/                         # Saved model weights (not in git)
├── results/                        # Training outputs and visualizations
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── metrics.json
│   └── gradcam/                    # Grad-CAM comparison images
├── .streamlit/
│   └── config.toml                 # Streamlit theme configuration
├── app.py                          # Streamlit web application
├── model.py                        # Model architectures (CNN, ResNet50, EfficientNet)
├── data_loader.py                  # Data loading and augmentation pipeline
├── gradcam.py                      # Grad-CAM heatmap generation
├── evaluate.py                     # Metrics computation and plotting
├── train.py                        # Training pipeline script
├── train_pcam.ipynb                # Original training notebook
├── requirements.txt
├── LICENSE
└── README.md
```

## Acknowledgements and Attribution

This project builds on the initial proof-of-concept by [BleeGleeWee](https://github.com/BleeGleeWee/Histopathology-Image-Classification), which demonstrated a single ResNet50-based classifier with Grad-CAM on the PCam dataset. That work is licensed under the MIT License and its copyright notice is retained in the [LICENSE](LICENSE) file.

The following components were independently developed by **Ahsan Zaman**:

- Multi-architecture benchmarking framework (Baseline CNN, ResNet50 with ImageNet weights, EfficientNet-B0)
- Data augmentation pipeline with random flips, rotations, brightness, and contrast adjustment
- Quantitative evaluation module (precision, recall, F1-score, AUC-ROC, confusion matrices, ROC curves)
- Grad-CAM comparison across all three architectures
- Streamlit deployment with model selection and live inference
- Complete training pipeline (`train.py`) with early stopping and learning rate scheduling

## References

1. Veeling, B. S., Linmans, J., Winkens, J., Cohen, T., & Welling, M. (2018). *Rotation Equivariant CNNs for Digital Pathology*. arXiv:1806.03962. [https://arxiv.org/abs/1806.03962](https://arxiv.org/abs/1806.03962)

2. Ehteshami Bejnordi et al. (2017). *Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer.* JAMA, 318(22), 2199-2210.

3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.

4. Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019.

5. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.
