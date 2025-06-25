# 🎙️ Emotion Classification from Speech using Deep Learning

## 📌 Project Overview
This project is an end-to-end pipeline for **Emotion Classification** from **Speech Audio Data** using deep learning. We leverage signal processing (MFCCs) and a Convolutional Neural Network (CNN) to detect emotional states such as **happy**, **sad**, **angry**, etc., directly from voice recordings.

## 🧠 Objective
To build a robust and accurate system that can:
- Extract relevant features from audio signals.
- Train a deep learning model for emotion classification.
- Deploy a web application using **Streamlit** to allow real-time emotion prediction from user-uploaded audio.

---

## 🗂️ Dataset
We use the **RAVDESS** dataset (Ryerson Audio-Visual Database of Emotional Speech and Song), available at:
🔗 [Zenodo Link](https://zenodo.org/record/1188976)

The dataset contains 8 different emotions expressed by actors through speech:
- **Neutral**
- **Calm**
- **Happy**
- **Sad**
- **Angry**
- **Fearful**
- **Disgust**
- **Surprised**

> Only selected audio files are used as per project instructions.

---

## 🛠️ Technologies and Libraries
- Python
- Librosa (audio processing)
- NumPy, Pandas
- TensorFlow / Keras
- Matplotlib, Seaborn
- Streamlit (for web app)

---

## 🔄 Preprocessing Pipeline
- Load `.wav` files and extract **MFCC features** (Mel Frequency Cepstral Coefficients)
- Normalize and reshape the feature vectors
- Label encoding of emotions

---

## 🤖 Model Architecture
- A **CNN-based model** was built using Keras.
- Input shape: `(40, 1)` MFCC vector
- Activation: ReLU
- Optimizer: Adam
- Output: Softmax layer with 8 classes

---

## 📈 Evaluation Metrics
- **F1-Score**: > 80%
- **Class-wise Accuracy**: > 75%
- **Confusion Matrix**: Included
- Model performs well on both validation and test splits.

---

## 🚀 How to Run the Code

### Jupyter Notebook
```bash
git clone https://github.com/your-username/emotion-classifier-speech.git
cd emotion-classifier-speech
open Mars.ipynb
