# 🦅 Aerial Object Classification & Detection

> Classify aerial images as **Bird** or **Drone** using deep learning — with optional YOLOv8 real-time detection and a Streamlit deployment.

---

## 📁 Project Structure

```
aerial_project/
├── configs/
│   └── data.yaml               # YOLOv8 dataset config
├── src/
│   ├── preprocess.py           # Data loading & augmentation
│   ├── custom_cnn.py           # Custom CNN architecture
│   ├── transfer_learning.py    # ResNet50 / MobileNet / EfficientNetB0
│   ├── train.py                # Training loop (both models)
│   ├── evaluate.py             # Evaluation, metrics, confusion matrix
│   ├── yolo_pipeline.py        # YOLOv8 training & inference
│   └── utils.py                # Helper functions
├── streamlit_app/
│   └── app.py                  # Streamlit deployment UI
├── scripts/
│   └── run_training.py         # End-to-end training script
├── notebooks/
│   └── aerial_eda.ipynb        # EDA & experimentation notebook
├── models/                     # Saved model weights (auto-created)
└── requirements.txt
```

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Set your dataset path
Edit `scripts/run_training.py` and set `DATASET_ROOT` to your dataset folder.

### 2. Train all models
```bash
python scripts/run_training.py
```

### 3. Launch Streamlit app
```bash
streamlit run streamlit_app/app.py
```

---

## 🧠 Models

| Model | Type | Notes |
|---|---|---|
| Custom CNN | From scratch | Conv → Pool → BN → Dropout → Dense |
| ResNet50 | Transfer Learning | Fine-tuned last layers |
| MobileNetV2 | Transfer Learning | Lightweight, fast |
| EfficientNetB0 | Transfer Learning | Best accuracy/size tradeoff |
| YOLOv8n | Object Detection | Optional, requires YOLO dataset |

---

## 📊 Metrics Tracked
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Training/Validation Loss & Accuracy Curves

---

## 🛠 Tech Stack
`Python` `TensorFlow/Keras` `YOLOv8 (Ultralytics)` `Streamlit` `OpenCV` `scikit-learn` `Matplotlib`
