import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model (cache for performance)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("outputs/models/pneumonia_model.h5")

model = load_model()

# UI Title
st.title("🩺 AI Medical Image Analysis System")
st.write("Upload a Chest X-ray to detect Pneumonia")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # ✅ Convert to RGB (FIX)
    img = Image.open(uploaded_file).convert("RGB")
    
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    # Show confidence
    confidence = prediction if prediction > 0.5 else (1 - prediction)

    st.write(f"🔍 Confidence: {confidence:.2f}")

    # Output result
    if prediction > 0.5:
        st.error("❌ Pneumonia Detected")
    else:
        st.success("✅ Normal")