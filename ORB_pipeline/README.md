# ORB–BoVW–SVM Pest Classification & Superpixel Detection (AgroPest12)


---
This repository contains a classical computer-vision pipeline for insect pest
classification and region-based detection on the **AgroPest12** dataset.
The approach is fully handcrafted (ORB descriptors + Bag-of-Visual-Words) and
uses a multi-class SVM for classification. Detection is performed by classifying
candidate regions obtained via SLIC superpixel proposals and refining results
with Non-Maximum Suppression (NMS).


## Pipeline Overview

### Training (classification)
1. **ORB preprocessing** (grayscale):
   - CLAHE local contrast enhancement  
   - mild unsharp mask (Gaussian blur + weighted sharpening)
2. **ORB keypoints + descriptors**
3. **BoVW codebook**:
   - MiniBatch K-Means clustering of pooled ORB descriptors  
   - Vocabulary size: `K = 400`
4. **BoVW encoding**:
   - Assign descriptors to nearest visual words  
   *L2 normalisation applied to histograms*
5. **SVM training**:
   - RBF kernel multi-class SVM  
   - Class-balanced weights
6. **Optional background class**:
   - Random negative boxes sampled from training images
   - Boxes must have low IoU with ground truth (background label `-1`)
   - Retrain SVM with pest classes + background

### Inference (detection)
1. **RGB preprocessing for SLIC**:
   - white balance
   - bilateral filter (reduce leaf texture)
   - gentle gamma correction
2. **SLIC superpixel segmentation** → bounding box proposals
3. **BoVW encoding for each proposal**
4. **SVM scoring + confidence threshold**
5. **NMS** to remove duplicates  
6. Final predicted bounding boxes + pest class labels

---

## Dataset Setup

The notebook expects the dataset under:
data/agropest12/
train/
images/
labels/
valid/
images/
labels/
test/
images/
labels/


### Option A: Manual download
Download from Kaggle:

`rupankarmajumdar/crop-pests-dataset`

Unzip into `data/agropest12/` so the structure matches above.

### Option B: Kaggle auto-download
If you have `~/.kaggle/kaggle.json` set up, the notebook can download the dataset
automatically from within a cell.

---

## Requirements

Non-stdlib packages used in the notebook:

- numpy  
- opencv-python  
- scikit-learn  
- scikit-image  
- tqdm  
- matplotlib  
- joblib  
- seaborn *(only for some plots)*  
- kaggle *(only if using auto-download)*  

Install:

```bash
pip install numpy opencv-python scikit-learn scikit-image tqdm matplotlib joblib seaborn kaggle






