# 🐾 Animal Eye Classifier using Transfer Learning (EfficientNetB0)

## 📌 Overview

The **Animal Eye Classifier** is a deep learning-based computer vision project that classifies animal eye images into three different categories using **Transfer Learning**. Instead of training a convolutional neural network from scratch, this project leverages **EfficientNetB0**, a pretrained model trained on the ImageNet dataset, allowing faster convergence and higher accuracy even with a relatively small dataset.

The project demonstrates the complete deep learning workflow, including dataset loading, preprocessing, data augmentation, model building, training, evaluation, visualization, and model serialization.

---

# 🎯 Objectives

* Build an image classification model for animal eye images.
* Apply Transfer Learning using EfficientNetB0.
* Improve model generalization through data augmentation.
* Evaluate training and validation performance.
* Save the trained model for future inference and deployment.

---

# 🧠 Concepts Implemented

### Deep Learning

* Artificial Neural Networks
* Convolutional Neural Networks (CNN)
* Transfer Learning
* Feature Extraction
* Image Classification

### Computer Vision

* Image preprocessing
* Image resizing
* Pixel normalization
* Dataset batching

### Data Augmentation

* Horizontal Flip
* Random Rotation
* Random Zoom
* Random Contrast

### Machine Learning Concepts

* Training & Validation Split
* Model Generalization
* Overfitting Prevention
* Loss Optimization
* Performance Evaluation

---

# 🛠 Technologies Used

* Python
* TensorFlow
* Keras
* EfficientNetB0
* NumPy
* Matplotlib

---

# 📂 Project Workflow

```
Dataset
    │
    ▼
Load Images
    │
    ▼
Train / Validation Split
    │
    ▼
Data Augmentation
    │
    ▼
EfficientNetB0 (Pretrained)
    │
    ▼
Global Average Pooling
    │
    ▼
Dropout Layer
    │
    ▼
Dense Layer (ReLU)
    │
    ▼
Softmax Output Layer
    │
    ▼
Training
    │
    ▼
Evaluation
    │
    ▼
Save Trained Model
```

---

# 📊 Dataset Preparation

The dataset is loaded directly using TensorFlow's `image_dataset_from_directory()` utility.

Features of the dataset pipeline:

* Automatic batching
* Automatic image resizing
* 80–20 train-validation split
* Random seed for reproducibility
* Efficient prefetching using TensorFlow's data pipeline

Input image size:

```
224 × 224 × 3
```

---

# 🔄 Data Augmentation

To improve model robustness and reduce overfitting, multiple augmentation techniques were applied during training.

* Random Horizontal Flip
* Random Rotation
* Random Zoom
* Random Contrast Adjustment

These transformations generate new variations of training images without modifying the original dataset.

---

# 🏗 Model Architecture

The classifier is built using Transfer Learning.

### Base Model

* EfficientNetB0
* Pretrained on ImageNet
* Top classification layers removed
* Pretrained weights frozen during training

### Custom Classification Head

```
Input Image
        │
Data Augmentation
        │
EfficientNet Preprocessing
        │
EfficientNetB0
        │
GlobalAveragePooling2D
        │
Dropout (0.3)
        │
Dense (128, ReLU)
        │
Dense (3, Softmax)
```

---

# ⚙ Training Configuration

| Parameter     | Value                           |
| ------------- | ------------------------------- |
| Image Size    | 224 × 224                       |
| Batch Size    | 32                              |
| Epochs        | 15                              |
| Optimizer     | Adam                            |
| Loss Function | Sparse Categorical Crossentropy |
| Metric        | Accuracy                        |

---

# 📈 Model Evaluation

The model performance is monitored using:

* Training Accuracy
* Validation Accuracy
* Training Loss
* Validation Loss

Accuracy and loss curves are plotted after training to visualize the learning process and detect possible overfitting or underfitting.

---

# 💾 Model Saving

After successful training, the model is exported in TensorFlow's native format.

```
model/
└── animal_eye_classifier.keras
```

This saved model can later be loaded for inference or deployment.

---

# 🚀 Features

* Transfer Learning with EfficientNetB0
* Automatic dataset loading
* Image augmentation pipeline
* High-performance TensorFlow data pipeline
* Frozen pretrained feature extractor
* Accuracy and loss visualization
* Model serialization for deployment

---

# 📚 Key Learnings

Through this project, the following concepts were explored:

* Deep Learning fundamentals
* Image Classification
* Transfer Learning
* Feature Extraction
* TensorFlow Dataset API
* Data Augmentation
* Model Optimization
* Performance Evaluation
* Computer Vision workflows
* Model Persistence

---

# 🔮 Future Improvements

* Fine-tune EfficientNetB0 by unfreezing upper layers.
* Increase dataset size for improved generalization.
* Add confusion matrix and classification report.
* Implement prediction on custom images.
* Build a web interface using Streamlit or Flask.
* Deploy the model on a cloud platform.

---

# 📁 Project Structure

```
Animal_Eye_Classifier/
│
├── eye_dataset/
│   └── train/
│
├── model/
│   └── animal_eye_classifier.keras
│
├── train.py
│
├── requirements.txt
│
└── README.md
```

---

# 👨‍💻 Conclusion

This project demonstrates a complete deep learning image classification pipeline using TensorFlow and EfficientNetB0. By combining transfer learning, data augmentation, and a custom classification head, the model efficiently learns visual features from animal eye images while maintaining computational efficiency. The project serves as a practical implementation of modern computer vision techniques and establishes a strong foundation for deploying real-world image classification systems.
