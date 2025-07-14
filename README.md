# 🧠 Brain Tumor Detection using CNN

This project uses a Convolutional Neural Network (CNN) to detect the presence of brain tumors in MRI images. It is trained on the Brain MRI dataset sourced from Kaggle.

---

## 📌 Project Overview

- **Goal**: Classify MRI brain scans into `Tumor` or `No Tumor`
- **Model**: Convolutional Neural Network (built with TensorFlow/Keras)
- **Dataset**: Brain MRI dataset (organized into `/yes` and `/no` image folders)
- **Input**: MRI images
- **Output**: Tumor or No Tumor (binary classification)

---

## 🧠 Technologies Used

- Python 3.x
- TensorFlow / Keras
- NumPy, Matplotlib, OS
- OpenCV (for image processing)
- Google Colab / Jupyter Notebook
- Git, GitHub

---

## 📁 Project Structure
'''

brain_tumor_app/
├── data/
│ ├── yes/ # Tumor images
│ └── no/ # Non-tumor images
├── model/ # Place your .h5 model here after download
├── train.py # Training script for CNN
├── utils.py # Utility functions
├── requirements.txt
└── README.md


'''

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/ashwi1n/brain_tumor_app.git
cd brain_tumor_app
