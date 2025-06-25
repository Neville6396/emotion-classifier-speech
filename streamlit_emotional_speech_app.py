import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# --- Page Setup ---
st.set_page_config(page_title="Emotion Detection", page_icon="🎙️", layout="centered")

# --- Custom Background ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #2c3e50, #4ca1af);
        color: white;
    }
    .stApp {
        background: linear-gradient(135deg, #2c3e50, #4ca1af);
    }
    h1, h2, h3, h4, h5 {
        color: #F5F5F5;
    }
    .css-18e3th9 {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Trained Model ---
model = load_model("emotion_classifier (1).h5")
emotion_mapping = {
    0: "angry",
    1: "calm",
    2: "disgust",
    3: "fearful",
    4: "happy",
    5: "neutral",
    6: "sad",
    7: "surprised"
}

# --- Correct MFCC-Based Feature Extraction ---
def extract_features(audio, sr, n_mfcc=40, max_len=300):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.T[np.newaxis, ...]

# --- App Layout ---
st.markdown("<h1 style='text-align: center;'>🎙️ Emotion Detection from Speech</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a <b>.wav</b> file to detect the underlying emotion in the audio.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload a `.wav` file", type=["wav"])

if uploaded_file is not None:
    st.markdown("---")
    st.subheader("🎧 Audio Playback")
    st.audio(uploaded_file, format="audio/wav")

    y, sr = librosa.load(uploaded_file, sr=16000)

    st.subheader("📊 Waveform")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform", color='white')
    plt.tight_layout()
    st.pyplot(fig)

    with st.spinner("🔍 Analyzing Emotion..."):
        features = extract_features(y, sr)  # shape: (1, 300, 40)
        prediction = model.predict(features)
        predicted_label = emotion_mapping[np.argmax(prediction)]

    st.markdown("---")
    st.markdown(f"<h2 style='color:#00e676; text-align:center;'>😄 Detected Emotion: <b>{predicted_label.upper()}</b></h2>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #ccc;'>Made by Priyanshu | Powered by Streamlit</p>", unsafe_allow_html=True)
