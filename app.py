import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Page config
st.set_page_config(page_title="Maize Disease Classifier", layout="centered")

# 2. Load model once and cache it
@st.cache_resource
def load_my_model():
    return load_model("maize_disease.keras")

model = load_my_model()

# 3. Config
IMG_SIZE = 256
CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# --- 🎨 Branding ---
brand_css = """
<style>
.brand-topright {
    position: fixed;
    top: 10px;
    right: 20px;
    font-size: 22px;
    font-weight: bold;
    color: #2E8B57;
    z-index: 1000;
}
.brand-center {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 48px;
    font-weight: bold;
    color: #2E8B57;
    opacity: 0.15;
    z-index: 500;
    pointer-events: none;
}
/* Sidebar watermark */
[data-testid="stSidebar"]::before {
    content: "👁️ FARMEYE";
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%) rotate(-90deg);
    font-size: 36px;
    font-weight: bold;
    color: #2E8B57;
    opacity: 0.1;
    white-space: nowrap;
    pointer-events: none;
}
</style>
<div class="brand-topright">👁️ FARMEYE</div>
<div class="brand-center">👁️ FARMEYE</div>
"""
st.markdown(brand_css, unsafe_allow_html=True)

# Sidebar burger menu
st.sidebar.title("📖 Menu")
st.sidebar.markdown("### PROJECT INTRO")
st.sidebar.info("This project detects maize leaf diseases using deep learning.")
st.sidebar.markdown("### DATASET")
st.sidebar.info("Corn/Maize Leaf Disease Dataset (Kaggle).")
st.sidebar.markdown("### INFO")
st.sidebar.info("Upload a maize leaf image to classify into Blight, Common Rust, Gray Leaf Spot, or Healthy.")

# --- Main UI ---
st.markdown("<h2 style='text-align:center; color:#2E8B57;'>👁️ FARMEYE</h2>", unsafe_allow_html=True)
st.title("🌽 Maize Leaf Disease Detection")
st.write("Upload a maize leaf image and the model will predict the disease.")

# Preprocessing
def preprocess_image(image: Image.Image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# File uploader
uploaded_file = st.file_uploader("🌽 Choose a maize leaf image... 🥬", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=500)

    if st.button("Predict"):
        with st.spinner("Classifying..."):
            processed_img = preprocess_image(image)
            preds = model.predict(processed_img, verbose=0)[0]
            class_idx = np.argmax(preds)
            confidence = np.max(preds)

            st.success(f"**Prediction: {CLASS_NAMES[class_idx]}**")
            st.metric("Confidence", f"{confidence*100:.2f}%")

            # --- Probabilities as sorted horizontal bar chart ---
            st.write("### Probabilities")
            prob_df = pd.DataFrame({
                "Class": CLASS_NAMES,
                "Probability": preds * 100
            })

            # Sort descending
            prob_df = prob_df.sort_values("Probability", ascending=False).reset_index(drop=True)

            # Assign colors after sorting
            colors = ["seagreen" if c == CLASS_NAMES[class_idx] else "gold" for c in prob_df["Class"]]

            fig, ax = plt.subplots(figsize=(6,3))  # compact chart
            ax.barh(prob_df["Class"], prob_df["Probability"], color=colors)
            ax.set_xlabel("Probability (%)")
            ax.set_xlim(0, 100)

            # Add percentage labels beside bars
            for i, (cls, v) in enumerate(zip(prob_df["Class"], prob_df["Probability"])):
                ax.text(v + 1, i, f"{v:.2f}%", va="center", color="black")

            st.pyplot(fig)
else:
    st.info("👆 Upload an image to get started")

# Footer TM
st.markdown(
    "<hr style='border:1px solid gray'>"
    "<p style='text-align:center; color:gray;'>Powered by TechCrush Cohort 6 Group36 ™</p>",
    unsafe_allow_html=True
)
