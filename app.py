# app.py

import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import io
import time
import pandas as pd

from tensorflow.keras.models import load_model

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")

MODEL_PATH = "model.h5"   # Your Keras model
EMO_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Audio feature extraction settings (ensure these match your training)
SR = 22050          # Sampling rate
N_MFCC = 40         # Number of MFCCs
MAX_LEN = 173       # Max number of time frames to pad/truncate MFCC


# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------
@st.cache_resource
def load_emotion_model():
    return load_model(MODEL_PATH)


# ----------------------------------------------------
# FEATURE EXTRACTION
# ----------------------------------------------------
def extract_mfcc(audio_bytes, sr=SR):
    """Reads audio bytes, converts to mono, resamples, extracts MFCCs, pads/truncates."""
    
    # Load audio
    with io.BytesIO(audio_bytes) as f:
        signal, file_sr = sf.read(f)

    # Convert to mono if stereo
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    # Resample if needed
    if file_sr != sr:
        signal = librosa.resample(signal.astype(np.float32), orig_sr=file_sr, target_sr=sr)

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)

    # Normalize
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)

    # Pad / truncate
    if mfcc.shape[1] < MAX_LEN:
        padding = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0), (0, padding)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_LEN]

    # Model expects: (1, n_mfcc, time, 1)
    mfcc = mfcc[..., np.newaxis]
    mfcc = mfcc[np.newaxis, ...]

    return mfcc


# ----------------------------------------------------
# PREDICTION
# ----------------------------------------------------
def predict_emotion(model, mfcc_input):
    probs = model.predict(mfcc_input)[0]
    top_idx = np.argmax(probs)
    return EMO_LABELS[top_idx], float(probs[top_idx]), probs


# ----------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------
def main():

    st.title("🎧 Speech Emotion Recognition")

    st.markdown("Upload an audio file to detect the **emotion** from the voice.")

    st.markdown("### Upload Audio File")
    audio_file = st.file_uploader("Upload audio (wav, mp3, m4a)", type=["wav", "mp3", "m4a"])

    # load model once
    model = load_emotion_model()

    if audio_file is not None:

        audio_bytes = audio_file.read()

        # Show audio player
        st.markdown("### Audio Player")
        st.audio(audio_bytes, format=audio_file.type)

        with st.spinner("Processing and predicting..."):
            mfcc_input = extract_mfcc(audio_bytes)
            label, conf, probs = predict_emotion(model, mfcc_input)
            time.sleep(0.3)

        # Result box
        st.markdown("### Predicted Emotion Box")
        st.success(f"Predicted Emotion: **{label.upper()}**")
        st.write(f"Confidence: **{conf:.2f}**")

        # Probability graph
        st.markdown("### Probability Chart")
        prob_dict = {EMO_LABELS[i]: float(probs[i]) for i in range(len(EMO_LABELS))}
        st.bar_chart(list(prob_dict.values()))

        # Probability table
        st.markdown("### Probability Table")
        st.table([
            {"Emotion": emo, "Confidence": f"{prob:.3f}"}
            for emo, prob in prob_dict.items()
        ])

    else:
        st.info("Please upload an audio file to get predictions.")


if __name__ == "__main__":
    main()

