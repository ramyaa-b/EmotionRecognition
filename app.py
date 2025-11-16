# app.py
import os
import tempfile
import traceback
import numpy as np
import streamlit as st
import soundfile as sf
import librosa
import tensorflow as tf

st.set_page_config(page_title="Emotion Recognition (debug dashboard)", layout="centered")

# -------------------------
# Configuration - EDIT THESE
# -------------------------
model_path = "model.h5"         # <-- change to your model path if different
n_mfcc = 40                     # number of MFCC coefficients expected (change if your model used different)
target_sr = 22050               # sampling rate used for MFCC extraction
# Provide the class names in the same order used during training
class_names = [
    "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"
]

# -------------------------
# Helper functions
# -------------------------
def safe_load_model(path):
    try:
        model = tf.keras.models.load_model(path)
        return model, None
    except Exception as e:
        return None, traceback.format_exc()

def compute_mfcc_from_file(path, sr=target_sr, n_mfcc=n_mfcc):
    """
    Returns MFCC matrix with shape (n_mfcc, frames)
    """
    try:
        y, fs = librosa.load(path, sr=sr, mono=True)
    except Exception:
        data, fs = sf.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        y = librosa.resample(data.astype(np.float32), orig_sr=fs, target_sr=sr)
        fs = sr
    mfcc = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=n_mfcc)
    return mfcc

def pad_or_truncate_frames(mat, desired_frames):
    # mat is expected (frames, features)
    frames, feats = mat.shape
    if frames == desired_frames:
        return mat
    if frames < desired_frames:
        pad_width = desired_frames - frames
        pad = np.zeros((pad_width, feats), dtype=mat.dtype)
        return np.vstack([mat, pad])
    else:
        return mat[:desired_frames, :]

def try_reshape_and_predict(model, arr):
    """
    Try multiple reshape permutations to match model.input_shape (Conv1D-style).
    Also tries collapsing time-axis via mean so shape becomes (1, 40, 1) if needed.
    Returns: (probs, used_shape, diagnostics)
    """
    x = np.array(arr)
    diagnostics = []
    expected = model.input_shape
    diagnostics.append(f"model.input_shape: {expected}")
    diagnostics.append(f"original array shape: {x.shape}, ndim={x.ndim}")

    candidates = []

    # 2D input common cases: (n_mfcc, frames) or (frames, n_mfcc)
    if x.ndim == 2:
        candidates.append(np.expand_dims(x.T, axis=0))        # (1, frames, n_mfcc)
        candidates.append(np.expand_dims(x, axis=0))          # (1, n_mfcc, frames)
        candidates.append(np.expand_dims(x[..., np.newaxis], axis=0))    # (1, n_mfcc, frames,1)
        candidates.append(np.expand_dims(x.T[..., np.newaxis], axis=0))  # (1, frames, n_mfcc,1)

    elif x.ndim == 3:
        candidates.append(x)                                  # as-is
        candidates.append(x.squeeze())
        try:
            candidates.append(np.transpose(x, (0, 2, 1)))
        except Exception:
            pass
        candidates.append(x[..., np.newaxis])

    elif x.ndim == 4:
        candidates.append(np.squeeze(x, axis=-1))
        candidates.append(np.squeeze(x, axis=0))
        candidates.append(np.squeeze(x))
        try:
            candidates.append(np.transpose(np.squeeze(x), (0,2,1)))
        except Exception:
            pass

    # generic attempt
    try:
        candidates.append(np.expand_dims(x, axis=0))
    except Exception as e:
        diagnostics.append(f"expand_dims failed: {e}")

    # Deduplicate candidates by shape
    seen = set()
    final_candidates = []
    for c in candidates:
        if not isinstance(c, np.ndarray):
            continue
        if c.shape not in seen:
            seen.add(c.shape)
            final_candidates.append(c)

    # Try direct candidates
    for cand in final_candidates:
        diagnostics.append(f"trying candidate shape: {cand.shape}")
        try:
            preds = model.predict(cand)
            diagnostics.append(f"SUCCESS with shape {cand.shape}")
            return preds[0], cand.shape, diagnostics
        except Exception as e:
            diagnostics.append(f"FAILED for shape {cand.shape}: {type(e).__name__}: {e}")

    # NEW: Try collapsed-time candidate (mean across time axis) -> (n_mfcc, 1) -> (1, n_mfcc, 1)
    try:
        if x.ndim == 2:
            collapsed = np.mean(x, axis=1, keepdims=True)    # (n_mfcc, 1)
            cand_collapsed = np.expand_dims(collapsed, axis=0)  # (1, n_mfcc, 1)
            diagnostics.append(f"trying collapsed (mean-over-time) candidate: {cand_collapsed.shape}")
            preds = model.predict(cand_collapsed)
            diagnostics.append(f"SUCCESS with collapsed candidate {cand_collapsed.shape}")
            return preds[0], cand_collapsed.shape, diagnostics
    except Exception as e:
        diagnostics.append(f"FAILED collapsed candidate: {type(e).__name__}: {e}")

    # Try aligning to model's expected (steps, features) by pad/truncate if possible
    try:
        expected_shapes = tuple(s for s in expected if s is not None)
        if len(expected_shapes) >= 2:
            exp_steps = expected_shapes[0]
            exp_features = expected_shapes[1]
            diagnostics.append(f"Trying align-to expected steps/features: {exp_steps}/{exp_features}")
            if x.ndim == 2:
                # try detect orientation
                if x.shape[0] == exp_features:
                    mat = x.T  # (frames, features)
                elif x.shape[1] == exp_features:
                    mat = x    # (frames, features) if already oriented that way
                else:
                    # fallback: assume transpose to get (frames, features)
                    mat = x.T
                mat2 = pad_or_truncate_frames(mat, exp_steps)
                cand = np.expand_dims(mat2, axis=0)  # (1, steps, features)
                diagnostics.append(f"Trying padded/truncated candidate shape: {cand.shape}")
                try:
                    preds = model.predict(cand)
                    diagnostics.append(f"SUCCESS with padded/truncated shape {cand.shape}")
                    return preds[0], cand.shape, diagnostics
                except Exception as e:
                    diagnostics.append(f"FAILED padded/truncated attempt: {type(e).__name__}: {e}")
    except Exception as e:
        diagnostics.append("Error while attempting align-to-expected: " + str(e))

    # nothing worked
    raise ValueError("All candidate shapes failed. Diagnostics:\n" + "\n".join(diagnostics))

# -------------------------
# Streamlit UI
# -------------------------
st.title("Emotion Recognition — Streamlit Dashboard (debug friendly)")

# environment info
st.write("TensorFlow version:", tf.__version__)
st.write("Working dir:", os.getcwd())
st.write("Expected model path:", model_path)

# Load model
model, load_err = safe_load_model(model_path)
if load_err:
    st.error("Failed to load model. See traceback below.")
    st.text(load_err)
    st.stop()
else:
    st.success("Model loaded successfully.")
    st.write("Model input shape:", model.input_shape)
    # Print model summary to terminal for debugging
    print("Loaded model summary:")
    model.summary()

st.write("---")

# File upload / local path
uploaded = st.file_uploader("Upload an audio file (wav/mp3/ogg/flac/m4a) — or use local path", type=["wav", "mp3", "ogg", "flac", "m4a"])
use_local_file = st.checkbox("Use local file path (instead of upload)", value=False)
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
    except Exception as e:
        st.error(f"Failed to write uploaded file to disk: {e}")
        st.stop()
elif use_local_file and local_path:
    if os.path.exists(local_path):
        audio_path = local_path
        # attempt to play local file (works if browser can access it)
        try:
            with open(local_path, "rb") as f:
                audio_bytes = f.read()
                st.audio(audio_bytes, format="audio/*")
        except Exception:
            pass
else:
    st.info("Upload an audio file above or choose a local file path to test prediction.")
    st.stop()

# compute MFCC
with st.spinner("Computing MFCC..."):
    try:
        mfcc = compute_mfcc_from_file(audio_path, sr=target_sr, n_mfcc=n_mfcc)
        st.write("Raw mfcc.shape (n_mfcc, frames):", mfcc.shape)
    except Exception:
        st.error("Failed to compute MFCC. See traceback in terminal.")
        st.text(traceback.format_exc())
        st.stop()

# Try prediction (multiple attempts)
with st.spinner("Trying model prediction (multiple reshape attempts)..."):
    try:
        probs, used_shape, diagnostics = try_reshape_and_predict(model, mfcc)
        st.success("Prediction successful.")
        st.write("Used input shape for model:", used_shape)
        st.write("Diagnostics (last attempts):")
        for line in diagnostics[-16:]:
            st.text(line)

        probs = np.array(probs).flatten()
        if len(probs) != len(class_names):
            st.warning(f"Model output length ({len(probs)}) != class_names length ({len(class_names)}). Showing top probabilities.")
            top_k = min(len(probs), 10)
            idxs = np.argsort(probs)[::-1][:top_k]
            for i in idxs:
                st.write(f"{i}: prob={probs[i]:.4f}")
        else:
            idx = int(np.argmax(probs))
            st.write("Predicted label:", class_names[idx])
            st.write(f"Confidence: {probs[idx]:.4f}")
            st.write("All class probabilities:")
            for cname, p in zip(class_names, probs):
                st.write(f"{cname}: {p:.4f}")

    except Exception as e:
        st.error("Prediction failed. See diagnostics below.")
        st.text(str(e))
        st.text(traceback.format_exc())
        st.stop()

# cleanup temp file if created
if uploaded is not None and audio_path and os.path.exists(audio_path):
    try:
        os.remove(audio_path)
    except Exception:
        pass

st.write("Done. If prediction seems incorrect, confirm `n_mfcc`, `target_sr`, and `class_names` match how the model was trained.")




