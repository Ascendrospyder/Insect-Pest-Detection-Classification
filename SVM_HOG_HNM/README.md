# SVM–HOG–HNM Pest Classification & Detection

---

This approach relies on handcrafted feature extraction (**Color HOG**) and a linear SVM trained via Stochastic Gradient Descent. Crucially, it implements **Hard Negative Mining (HNM)** to iteratively reduce false positives by identifying difficult background regions during training.

## Pipeline Overview

### 1. Feature Extraction & Training (V1)
1.  **Patch Extraction**:
    - Positive samples are cropped from training images using ground truth labels.
    - Negative samples are random crops from background regions (no overlap with GT).
    - All patches resized to fixed dimensions.
2.  **HOG Features**:
    - **Color HOG**: Histogram of Oriented Gradients computed on each channel (B, G, R) and concatenated.
    - Configuration: `9` orientations, `6x6` pixels per cell, `2x2` cells per block.
3.  **Preprocessing**:
    - `StandardScaler` fitted on training data chunks (incremental learning to save RAM).
4.  **SVM Training (Linear)**:
    - `SGDClassifier` with `loss='hinge'` (Linear SVM).
    - Trained using `partial_fit` to handle large datasets in chunks.
    - Class weights computed to handle imbalance between background and pest classes.

### 2. Hard Negative Mining (HNM)
To improve detection performance and reduce False Positives (FPs):
1.  **Mining**: Run the trained V1 detector over the training dataset.
2.  **Identification**: Any detection with high confidence but low IoU ($< 0.1$) with ground truth is flagged as a **False Positive**.
3.  **Collection**: These "Hard Negatives" are cropped, converted to HOG features, and saved to disk.
4.  **Retraining (V2)**: The SVM is updated/retrained using the original positive samples mixed with the new hard negatives (labeled as background).

### 3. Inference & Detection
1.  **Image Pyramid**:
    - Input images are downscaled iteratively (Scale factor `1.5`) to detect objects at various sizes.
2.  **Sliding Window**:
    - A window slides across the image pyramid to extract HOG features at every step.
3.  **Classification**:
    - The retrained SVM scores each window.
4.  **NMS (Non-Maximum Suppression)**:
    - Overlapping boxes are pruned based on confidence scores and IoU threshold (`0.3`).
5.  **Robustness Evaluation**:
    - The pipeline is tested against synthetic corruptions: Gaussian Noise, Blur, Low Brightness, High Contrast, and Occlusion.

---

## Dataset Setup

The notebook expects the dataset (AgroPest12) to be organized in the following directory structure under an `archive/` folder:

/workspaces/comp9517/project/archive/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/


*Note: Input and Output paths can be configured in the "HYPERPARAMETERS" cell.*

---

## Hyperparameters

Key configurations defined in the notebook:

* **Patch Size**: `(64, 64)`
* **HOG**: 9 Orientations, 6 PPC, 2 CPB, Color=True
* **Training**: 20 Epochs, Batch Size 4096
* **HNM**: Mining on 100% of training data
* **Detection**: Pyramid Scale 1.5, Step Size 12, NMS Threshold 0.3

---

## Requirements

The project utilizes `scikit-learn` for the classifier and `scikit-image` for feature extraction.

- numpy
- pandas
- opencv-python
- scikit-learn
- scikit-image
- tqdm
- matplotlib
- seaborn
- pyyaml
- joblib

Install dependencies via:

```bash
pip install pandas matplotlib seaborn pyyaml opencv-python tqdm numpy scikit-learn scikit-image joblib
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx