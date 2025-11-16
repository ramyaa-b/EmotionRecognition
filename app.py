# app.py (minimal — shows only final prediction + audio)
import os
import tempfile
import traceback
import numpy as np
import streamlit as st
import soundfile as sf
import librosa
import tensorflow as tf

st.set_page_config(page_title="Emotion Recognition", layout="centered")

# --------- Configuration (EDIT IF NEEDED) ----------
model_path = "model.h5"         # path to your saved Keras model
n_mfcc = 40                     # MFCC coefficients used in training
target_sr = 22050               # sample rate used in training
# class names must be in the same order used during training
class_names = [
    "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"
]
# ---------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model(path):
    return tf.keras.models.load_model(path)

def compute_mfcc_from_file(path, sr=target_sr, n_mfcc=n_mfcc):
    try:
        y, fs = librosa.load(path, sr=sr, mono=True)
    except Exception:
        data, fs = sf.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        y = librosa.resample(data.astype(np.float32), orig_sr=fs, target_sr=sr)
        fs = sr
    mfcc = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=n_mfcc)
    return mfcc  # shape: (n_mfcc, frames)

def prepare_input_for_model_from_mfcc(mfcc):
    """
    The model expects shape (1, 40, 1). We collapse time axis by mean to match that.
    """
    # mfcc: (n_mfcc, frames)
    collapsed = np.mean(mfcc, axis=1, keepdims=True)   # -> (n_mfcc, 1)
    x_input = np.expand_dims(collapsed, axis=0)         # -> (1, n_mfcc, 1)
    return x_input

def predict_label_from_audio_file(model, audio_path):
    mfcc = compute_mfcc_from_file(audio_path, sr=target_sr, n_mfcc=n_mfcc)
    x = prepare_input_for_model_from_mfcc(mfcc)
    preds = model.predict(x)
    probs = np.array(preds[0]).flatten()
    return probs

# ---- UI ----
st.title("Emotion Recognition")

# load model (show friendly message on failure)
try:
    model = load_model(model_path)
except Exception as e:
    st.error("Failed to load model. Check model_path and model file.")
    st.stop()

uploaded = st.file_uploader("Upload an audio file (wav/mp3/ogg/flac/m4a)", type=["wav", "mp3", "ogg", "flac", "m4a"])
use_local_file = st.checkbox("Use local file path instead of upload", value=False)
local_path = ""
if use_local_file:
    local_path = st.text_input("Enter local audio file path (absolute or relative):", value="")

audio_path = None
if uploaded is not None:
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(uploaded.getbuffer())
        tmp.flush()
        tmp.close()
        audio_path = tmp.name
        st.audio(uploaded, format="audio/*")
    except Exception:
        st.error("Failed to read uploaded file.")
        st.stop()
elif use_local_file and local_path:
    if os.path.exists(local_path):
        audio_path = local_path
        # attempt to play local file if possible
        try:
            with open(local_path, "rb") as f:
                st.audio(f.read(), format="audio/*")
        except Exception:
            pass
else:
    st.info("Upload an audio file above or choose a local file path to predict.")
    st.stop()

# Predict button (so user intentionally triggers)
if st.button("Predict Emotion"):
    try:
        probs = predict_label_from_audio_file(model, audio_path)
        probs = probs.flatten()
        # If output length doesn't match class_names, show top label from probs anyway
        if probs.size == 0:
            st.error("Model returned no probabilities.")
        else:
            idx = int(np.argmax(probs))
            confidence = float(probs[idx])
            label = class_names[idx] if idx < len(class_names) else str(idx)
            # Display final prediction only
            st.markdown(f"### Predicted emotion: **{label}**")
            st.markdown(f"**Confidence:** {confidence:.4f}")
    except Exception:
        # Friendly error without raw debug dump
        st.error("Prediction failed. Check audio file format and model compatibility.")
        # Also print traceback to terminal for your debugging
        print("Prediction error:", traceback.format_exc())

# cleanup temp file
if uploaded is not None and 'audio_path' in locals() and audio_path and os.path.exists(audio_path):
    try:
        os.remove(audio_path)
    except Exception:
        pass





