# traffic-sign-classifier-gtsrb-cnn
A deep learning model to classify German traffic signs using CNN and GTSRB dataset.
# 🛑 Traffic Sign Classifier using CNN | GTSRB Dataset

A complete deep learning project to classify German traffic signs using Convolutional Neural Networks (CNNs). This project utilizes the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset and includes data preprocessing, augmentation, training, and evaluation with metrics like accuracy, loss curves, and confusion matrix.

---

## 🚀 Project Overview

**Goal:**  
Build a high-accuracy deep learning model using TensorFlow/Keras (or PyTorch) that can recognize traffic signs from the GTSRB dataset.

---

## 📦 Dataset

The dataset used is the official GTSRB dataset available on Kaggle:

**Kaggle Dataset:** [`ibrahimkaratas/gtsrb-german-traffic-sign-recognition-benchmark`](https://www.kaggle.com/datasets/ibrahimkaratas/gtsrb-german-traffic-sign-recognition-benchmark)

Use the following to download directly in Colab:
```python
import kagglehub
path = kagglehub.dataset_download("ibrahimkaratas/gtsrb-german-traffic-sign-recognition-benchmark")
🧠 Model Features
✅ CNN built from scratch or using Transfer Learning (MobileNetV2, ResNet)

✅ Data preprocessing: resizing, normalization

✅ Data augmentation for robustness

✅ Model trained with train-validation-test split

✅ Visualizations of:

Accuracy & Loss over epochs

Confusion matrix

Classification report (Precision, Recall, F1-Score)
