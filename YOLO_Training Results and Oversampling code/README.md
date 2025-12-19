# YOLOv12 Project Structure and Documentation

This repository contains code, experiments, and result-processing scripts for evaluating three YOLOv12 models (**YOLOv12s**, **YOLOv12n**, and **YOLOv12m**), including oversampling techniques and post-processing of performance metrics.


## 📁 Project Structure

```
├── Final_Results_For_All_models/
│   ├── getting_required_metrics.ipynb
│   └── ... (exported results)
│
├── YOLO_V12_m_code and results/
│   ├── training code
│   ├── YOLO-native evaluation results
│   └── ...
│
├── YOLO_V12_n code and results/
│   ├── training code
│   ├── YOLO-native evaluation results
│   └── ...
│
├── YOLO_V12_s code and results/
│   ├── training code
│   ├── YOLO-native evaluation results
│   └── ...
│
├── oversampling.ipynb
└── README.md
```

---

## 📘 Folder Descriptions

### **1. `Final_Results_For_All_models/`**

This folder contains:

* **`getting_required_metrics.ipynb`** – a notebook used to convert YOLO's default evaluation metrics into classification-style metrics.

  * YOLO outputs precision, recall, and F1 scores at **IoU = 50%**, which follow detection-based definitions.
  * This notebook re-processes YOLO predictions into **sklearn-compatible classification metrics**, ensuring fair comparison across models.
tputs for all three models.

---

### **2. `oversampling.ipynb`**

This notebook contains:

* The **oversampling strategy** used to address class imbalance.
* **Augmentation logic** applied to minority classes.
* The procedure used to generate the oversampled datasets consumed by each YOLO model.

This notebook is essential for replicating the improved performance of YOLOv12m with oversampling.

---

### **3. YOLO Model Folders**

Each of the following folders contains **all training code and native YOLO results**:

* `YOLO_V12_s code and results/`
* `YOLO_V12_n code and results/`
* `YOLO_V12_m_code and results/`

Inside each folder, you will  find:

* Training scripts used to run the model.
* YOLO-native evaluation outputs (precision, recall, F1, mAP50, mAP50-95, etc.).

These results represent the raw outputs before conversion into sklearn-based metrics.



## 🔍 Metric Conversion Workflow

1. Train models normally using YOLO.
2. Export predictions and raw metrics.
3. Run `getting_required_metrics.ipynb` to compute the following on the test dataset:

   * Precision (classification definition)
   * Recall
   * F1 Score
   * Accuracy
   * AUC 
4. Aggregate results across all models into a comparable final table.



## 📝 Notes

* The oversampling notebook should be run **before** training the YOLOV12m model if you want to reproduce oversampled experiments.
* YOLO-native metrics are detection-based; the converted metrics follow sklearn classification definitions.

## 📬 Contact / Questions

For further help or clarification, feel free to reach out or create an issue in the repository.
