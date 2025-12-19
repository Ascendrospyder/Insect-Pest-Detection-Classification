# SIFT + BoVW + SVM/kNN Pest Classification

This project implements a classical computer vision pipeline for crop pest classification using SIFT (Scale-Invariant Feature Transform) features, Bag-of-Visual-Words (BoVW) representation, and traditional machine learning classifiers (SVM and k-Nearest Neighbors).

## Overview

The pipeline consists of the following stages:

1. **Dataset Download & Preprocessing**: Downloads the Crop Pests Dataset from Kaggle and extracts bounding box crops from YOLO-format annotations
2. **Feature Extraction**: Extracts SIFT keypoints and descriptors from cropped pest images
3. **Visual Vocabulary Construction**: Builds a visual vocabulary using K-Means clustering (K=500)
4. **BoVW Feature Encoding**: Converts SIFT descriptors into histogram-based BoVW features
5. **Classification**: Trains and evaluates both Linear SVM and kNN classifiers
6. **Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, and macro AUC

## Requirements

```bash
pip install opencv-contrib-python scikit-learn tqdm joblib matplotlib seaborn
```

### Dependencies

- `opencv-contrib-python`: SIFT feature extraction
- `scikit-learn`: K-Means clustering, SVM, kNN, metrics
- `tqdm`: Progress bars
- `joblib`: Model serialization
- `matplotlib` & `seaborn`: Visualization
- `numpy`: Array operations

### Kaggle API Setup

This project downloads data from Kaggle, so you'll need:
1. A Kaggle account
2. API credentials (`kaggle.json` file)

## Dataset

**Crop Pests Dataset** from Kaggle: [rupankarmajumdar/crop-pests-dataset](https://www.kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset)

- **12 pest classes** (classes 0-11)
- **YOLO format** annotations
- Splits: train (11,502 images), validation (1,095 images), test (546 images)
- After cropping: 15,282 train crops, 1,341 valid crops, 689 test crops

## Usage

### Running on Google Colab (Recommended)

This notebook is designed for Google Colab. Simply:

1. Upload the notebook to Google Colab
2. Upload your `kaggle.json` file when prompted
3. Run all cells sequentially

### Running Locally

If running locally, modify these paths in the notebook:

```python
base = "/your/local/path/pest12"  # Change from "/content/pest12"
```

And comment out or modify the Colab-specific cells:
- Cell 2: `from google.colab import files` (for uploading kaggle.json)
- Cell 3-4: Kaggle authentication setup

## Pipeline Details

### 1. Data Preprocessing (Cells 0-8)

- Downloads dataset from Kaggle
- Converts YOLO format (normalized xc, yc, w, h) to pixel coordinates
- Crops bounding boxes from images
- Organizes crops by split and class: `crops/{split}/{class_id}/`

### 2. SIFT Feature Extraction (Cell 9)

- Extracts SIFT descriptors from grayscale crops
- Throttles to max 50 images per class and 50 descriptors per image (for training)
- Collects 28,988 descriptors for vocabulary construction

### 3. Visual Vocabulary Construction (Cell 10)

- **K-Means clustering** with K=500 visual words
- Uses MiniBatchKMeans for efficiency
- Saves vocabulary model: `kmeans_sift_K500.joblib`

### 4. BoVW Feature Encoding (Cell 11)

- Assigns SIFT descriptors to nearest visual words
- Creates L2-normalized histograms (500-dimensional vectors)
- Saves features: `sift_bovw_K500_features.npz`

### 5. Classifier Training

#### Linear SVM (Cells 13-17)

- **Hyperparameter tuning**: Tests C ∈ {0.1, 1.0, 10.0}
- **Best C**: 0.1 (validation accuracy: 41.3%)
- Pipeline: StandardScaler → LinearSVC

#### k-Nearest Neighbors (Cells 18-21)

- **Hyperparameter tuning**: Tests k ∈ {3, 5, 7}
- **Best k**: 7 (validation accuracy: 18.5%)
- Pipeline: StandardScaler → KNeighborsClassifier

## Results

### Test Set Performance


## Limitations & Future Work

1. **Feature representation**: SIFT + BoVW is a classical approach; deep learning (CNNs) would likely achieve higher accuracy
2. **Vocabulary size**: K=500 was chosen arbitrarily; grid search over K could improve results
3. **Class imbalance**: Some classes have far fewer samples than others
4. **Spatial information**: BoVW discards spatial relationships between features

## References

- SIFT: Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints
- Bag-of-Visual-Words: Csurka, G., et al. (2004). Visual categorization with bags of keypoints
- Dataset: [Crop Pests Dataset on Kaggle](https://www.kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset)

## License

Dataset is licensed under MIT License.
