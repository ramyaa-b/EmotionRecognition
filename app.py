# app.py
import streamlit as st
import numpy as np
from PIL import Image
import io
import time

# If Keras model:
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Emotion Recognition", layout="centered")

@st.cache_resource
def load_my_model(path="model.h5"):
    model = load_model(path)
    return model

# TODO: put your exact labels here in the correct order used by your model
EMO_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def preprocess_image(pil_img, target_size=(48,48), grayscale=True):
    """
    Convert PIL image to array ready for your model.
    - Adjust target_size and grayscale depending on your model input.
    - Return shape should match model input: e.g. (1, h, w, 1) or (1, h, w, 3)
    """
    if grayscale:
        pil_img = pil_img.convert("L")  # single channel
    else:
        pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(target_size)
    arr = np.array(pil_img).astype(np.float32)
    # Example normalization (change if your training used a different scheme)
    arr = arr / 255.0
    if grayscale:
        arr = np.expand_dims(arr, -1)  # (h,w,1)
    arr = np.expand_dims(arr, 0)      # (1,h,w,channels)
    return arr

def predict_image(model, pil_img):
    x = preprocess_image(pil_img, target_size=(48,48), grayscale=True)  # adjust if needed
    preds = model.predict(x)  # shape e.g. (1, num_classes)
    probs = preds[0]
    top_idx = np.argmax(probs)
    return EMO_LABELS[top_idx], float(probs[top_idx]), probs

def main():
    st.title("🎭 Emotion Recognition — Simple Streamlit Dashboard")
    st.write("Upload an image or take a picture with your camera.")

    model = load_my_model("model.h5")

    col1, col2 = st.columns([2,1])

    with col1:
        uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
        # camera input (works in browsers; mobile support varies)
        cam_pic = st.camera_input("Or take a picture")

    with col2:
        st.markdown("**Example usage**")
        st.write("Upload a face image or use camera.")
        st.write("Model: Keras `.h5` — adapt `preprocess_image` if your training used different input.")

    image = None
    if uploaded is not None:
        image = Image.open(uploaded)
    elif cam_pic is not None:
        image = Image.open(io.BytesIO(cam_pic.getvalue()))

    if image is not None:
        st.image(image, caption="Input image", use_column_width=True)
        with st.spinner("Predicting..."):
            label, conf, probs = predict_image(model, image)
            time.sleep(0.2)
        st.success(f"Prediction: **{label}**  — confidence {conf:.2f}")
        # Show probability bar chart
        prob_dict = {EMO_LABELS[i]: float(probs[i]) for i in range(len(EMO_LABELS))}
        st.bar_chart(list(prob_dict.values()), x=None, height=200)
        # Also show table
        st.table([{ "Emotion": k, "Confidence": f"{v:.3f}" } for k, v in prob_dict.items()])

if __name__ == "__main__":
    main()
