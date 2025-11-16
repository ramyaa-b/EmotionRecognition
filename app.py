# app.py
import streamlit as st
import numpy as np
import traceback
import tempfile
import os
import soundfile as sf
import librosa
import tensorflow as tf

st.set_page_config(page_title="Emotion Recognition (debug dashboard)", layout="centered")

# -------------------------
# Configuration - edit these
# -------------------------
model_path = "model.h5"         # <- change to your model file if different
n_mfcc = 40                     # number of MFCC coefficients you expect (common: 13 or 40)
target_sr = 22050               # sampling rate used for mfcc extraction (common default)
class_names = [                 # default class list - change to your dataset's classes if different
    "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"
]

# -------------------------
# Helpers
# -------------------------
def safe_load_model(path):
    try:
        model = tf.keras.models.load_model(path)
        return model, None
    except Exception as e:
        return None, traceback.format_exc()

def compute_mfcc_from_file(path, sr=target_sr, n_mfcc=n_mfcc):
    # Load audio, preserve mono
    try:
        y, fs = librosa.load(path, sr=sr, mono=True)
    except Exception:
        # fallback using soundfile then resample if needed
        data, fs = sf.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        y = librosa.resample(data.astype(np.float32), orig_sr=fs, target_sr=sr)
        fs = sr
    # compute MFCC (shape: (n_mfcc, frames))
    mfcc = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=n_mfcc)
    return mfcc

def pad_or_truncate_frames(mat, desired_frames):
    """
    mat: numpy array shape (frames, features) OR (features, frames)
    We'll try to detect orientation outside. This function expects (frames, features).
    """
    frames, feats = mat.shape
    if frames == desired_frames:
        return mat
    if frames < desired_frames:
        # pad with zeros at end
        pad_width = desired_frames - frames
        pad = np.zeros((pad_width, feats), dtype=mat.dtype)
        return np.vstack([mat, pad])
    else:
        # truncate
        return mat[:desired_frames, :]

def try_reshape_and_predict(model, arr):
    """
    Try multiple common reshape permutations to match model.input_shape for Conv1D style models.
    arr: numpy array of mfcc or preprocessed feature (e.g. shape (40,173) or (173,40) or (1,40,173,1))
    Returns: (probs, used_shape, diagnostics) on success, or raises ValueError with diagnostics.
    """
    x = np.array(arr)
    diagnostics = []
    expected = model.input_shape
    diagnostics.append(f"model.input_shape: {expected}")
    diagnostics.append(f"original array shape: {x.shape}, ndim={x.ndim}")

    candidates = []

    # If arr is 2D: (n_mfcc, frames) or (frames, n_mfcc)
    if x.ndim == 2:
        # Treat x as (n_mfcc, frames) -> transpose to (frames, n_mfcc)
        candidates.append(np.expand_dims(x.T, axis=0))        # (1, frames, n_mfcc)
        candidates.append(np.expand_dims(x, axis=0))          # (1, n_mfcc, frames)
        # add channel axis at end (4D) (likely wrong for Conv1D but try)
        candidates.append(np.expand_dims(x[..., np.newaxis], axis=0))       # (1, n_mfcc, frames, 1)
        candidates.append(np.expand_dims(x.T[..., np.newaxis], axis=0))     # (1, frames, n_mfcc, 1)

    elif x.ndim == 3:
        candidates.append(x)                                   # as-is
        candidates.append(x.squeeze())
        candidates.append(np.transpose(x, (0, 2, 1)))
        candidates.append(x[..., np.newaxis])

    elif x.ndim == 4:
        candidates.append(np.squeeze(x, axis=-1))
        candidates.append(np.squeeze(x, axis=0))
        candidates.append(np.squeeze(x))
        candidates.append(np.transpose(np.squeeze(x), (0,2,1)))

    # generic attempt
    try:
        candidates.append(np.expand_dims(x, axis=0))
    except Exception as e:
        diagnostics.append(f"expand_dims failed: {e}")

    # Deduplicate candidates by shape
    seen = {}
    final_candidates = []
    for c in candidates:
        if not isinstance(c, np.ndarray):
            continue
        if c.shape not in seen:
            seen[c.shape] = True
            final_candidates.append(c)

    # Try predicting
    for cand in final_candidates:
        diagnostics.append(f"trying candidate shape: {cand.shape}")
        try:
            preds = model.predict(cand)
            diagnostics.append(f"SUCCESS with shape {cand.shape}")
            return preds[0], cand.shape, diagnostics
        except Exception as e:
            diagnostics.append(f"FAILED for shape {cand.shape}: {type(e).__name__}: {e}")

    # If nothing worked, try intelligent reshape based on model.input_shape (pad/truncate)
    # Inspect model expectation ignoring batch dim
    try:
        expected_shapes = tuple(s for s in expected if s is not None)
        # expected_shapes usually (steps, features) or (steps, features, ...)
        if len(expected_shapes) >= 2:
            exp_steps = expected_shapes[0]
            exp_features = expected_shapes[1]
            diagnostics.append(f"Trying align-to expected steps/features: {exp_steps}/{exp_features}")
            # Make sure arr is 2D as (frames, features)
            if x.ndim == 2:
                # detect orientation: if first dim equals n_mfcc, assume (n_mfcc, frames)
                if x.shape[0] == exp_features:
                    # x currently (features, frames) -> transpose to (frames, features)
                    mat = x.T
                elif x.shape[1] == exp_features:
                    mat = x
                else:
                    # fallback: try both orientations, prefer (frames, features) by transposing if necessary
                    mat = x.T
                # pad or truncate frames
                mat2 = pad_or_truncate_frames(mat, exp_steps)
                cand = np.expand_dims(mat2, axis=0)  # (1, steps, features)
                diagnostics.append(f"Trying padded/truncated candidate shape: {cand.shape}")
                try:
                    preds = model.predict(cand)
                    diagnostics.append(f"SUCCESS with padded/truncated shape {cand.shape}")
                    return preds[0], cand.shape, diagnostics
                except Exception as e:
                    diagnostics.append(f"FAILED padded/truncated attempt: {type(e).__name__}: {e}")
            else:
                diagnostics.append("Data not 2D, skipping padded/truncate attempt.")
    except Exception as e:
        diagnostics.append("Error while attempting align-to-expected: " + str(e))

    raise ValueError("All candidate shapes failed. Diagnostics:\n" + "\n".join(diagnostics))


# -------------------------
# Streamlit UI
# -------------------------
st.title("Emotion Recognition — Streamlit Dashboard (debug friendly)")

# show environment info
st.write("TensorFlow version:", tf.__version__)
st.write("Working dir:", os.getcwd())
st.write("Expected model path:", model_path)

# load model
model, load_err = safe_load_model(model_path)
if load_err:
    st.error("Failed to load model. See traceback below.")
    st.text(load_err)
    st.stop()
else:
    st.success("Model loaded successfully.")
    st.write("Model input shape:", model.input_shape)
    # print a short model summary to terminal (not to the UI, it can be long)
    print("Loaded model summary:")
    model.summary()

st.write("---")

uploaded = st.file_uploader("Upload an audio file (wav/mp3/ogg...) — or use sample below", type=["wav", "mp3", "ogg", "flac", "m4a"])
col1, col2 = st.columns(2)
with col1:
    if uploaded is not None:
        st.audio(uploaded, format="audio/*")
with col2:
    st.write("Tip: Upload the same dataset audio that worked in Colab to compare outputs.")

# provide an example/browse local file selector (optional)
use_local_file = st.checkbox("Use local file path (instead of upload)", value=False)
local_path = None
if use_local_file:
    local_path = st.text_input("Enter local audio file path (absolute or relative):", value="")

# If user uploaded file, write to a temp file
audio_path = None
if uploaded is not None:
    try:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tfile.write(uploaded.getbuffer())
        tfile.flush()
        tfile.close()
        audio_path = tfile.name
    except Exception as e:
        st.error(f"Failed to write uploaded file to disk: {e}")
        st.stop()
elif use_local_file and local_path:
    if os.path.exists(local_path):
        audio_path = local_path
    else:
        st.error("Local path does not exist.")
        st.stop()
else:
    st.info("Upload an audio file above or choose a local file path to test prediction.")
    st.stop()

# compute MFCC and show shapes
with st.spinner("Computing MFCC..."):
    try:
        mfcc = compute_mfcc_from_file(audio_path, sr=target_sr, n_mfcc=n_mfcc)
        st.write("Raw mfcc.shape (n_mfcc, frames):", mfcc.shape)
    except Exception as e:
        st.error("Failed to compute MFCC. See traceback in terminal.")
        st.text(traceback.format_exc())
        st.stop()

# Try predicting
with st.spinner("Trying model prediction (multiple reshape attempts)..."):
    try:
        probs, used_shape, diagnostics = try_reshape_and_predict(model, mfcc)
        st.success("Prediction successful.")
        st.write("Used input shape for model:", used_shape)
        st.write("Diagnostics (last attempts):")
        # show only last ~8 entries for brevity
        for line in diagnostics[-12:]:
            st.text(line)

        # If probs is not 1D, try to flatten
        probs = np.array(probs).flatten()
        # if class_names mismatch length, show both
        if len(probs) != len(class_names):
            st.warning(f"Model output length ({len(probs)}) != class_names length ({len(class_names)}). Showing top probs only.")
            # show top-k
            top_k = min(len(probs), 10)
            idxs = np.argsort(probs)[::-1][:top_k]
            for i in idxs:
                st.write(f"{i}: prob={probs[i]:.4f}")
        else:
            # show label + probs
            idx = int(np.argmax(probs))
            st.write("Predicted label:", class_names[idx])
            st.write(f"Confidence: {probs[idx]:.4f}")
            st.write("All class probabilities:")
            for cname, p in zip(class_names, probs):
                st.write(f"{cname}: {p:.4f}")

    except Exception as e:
        st.error("Prediction failed. See diagnostics below.")
        st.text(str(e))
        # If the exception string is long, show last part
        tb = traceback.format_exc()
        st.text(tb)
        st.stop()

# cleanup temp file if we created one
if uploaded is not None and audio_path and os.path.exists(audio_path):
    try:
        os.remove(audio_path)
    except Exception:
        pass

st.write("Done. If prediction is incorrect, check that `n_mfcc`, `target_sr`, and `class_names` match how the model was trained.")



