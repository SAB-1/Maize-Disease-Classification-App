import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# ════════════════════════════════════════════════════════════════
# 1. Page config
# ════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Maize Disease Classifier", layout="wide")

# ════════════════════════════════════════════════════════════════
# 2. Load model once and cache it
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def load_my_model():
    m = load_model("maize_disease.keras")
    m(tf.zeros((1, 256, 256, 3)))  # Force build so Grad-CAM works
    return m


def get_last_conv_layer(model):
    """
    Find the last Conv2D that has a usable output tensor by scanning
    every layer at every nesting level, but ONLY accepting layers whose
    .output tensor is already defined (i.e. was called during the main
    model forward pass and is reachable from model.inputs).
    """
    candidates = []

    def scan(m):
        for layer in m.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                try:
                    out = layer.output
                    if out is not None:
                        candidates.append(layer)
                except Exception:
                    pass
            if hasattr(layer, "layers"):
                scan(layer)

    scan(model)
    return candidates[-1] if candidates else None


model = load_my_model()
last_conv_layer = get_last_conv_layer(model)   # computed once globally

# ════════════════════════════════════════════════════════════════
# 3. Config
# ════════════════════════════════════════════════════════════════
IMG_SIZE = 256
CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# ════════════════════════════════════════════════════════════════
# 4. CSS — Farm-green theme inspired by the e-commerce UI layout
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=Space+Grotesk:wght@400;600;700&display=swap');

/* ── Root ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #f0faf2 !important;
}
[data-testid="stAppViewContainer"] > .main {
    background: #f0faf2;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1.5px solid #d1f0da !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] * {
    color: #1a3d2b !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ── Header bar ── */
.farm-header {
    background: linear-gradient(135deg, #16a34a 0%, #15803d 60%, #14532d 100%);
    border-radius: 18px;
    padding: 28px 36px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 8px 32px rgba(22,163,74,0.18);
}
.farm-header-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
    margin: 0;
}
.farm-header-sub {
    font-size: 0.95rem;
    color: #bbf7d0;
    margin-top: 4px;
    font-weight: 400;
}
.farm-header-badge {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 50px;
    padding: 8px 18px;
    color: #ffffff;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Stat tiles ── */
.stat-tile {
    background: #ffffff;
    border: 1.5px solid #d1f0da;
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(22,163,74,0.06);
}
.stat-tile-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #16a34a;
    line-height: 1;
}
.stat-tile-label {
    font-size: 0.72rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 4px;
    font-weight: 600;
}

/* ── Cards ── */
.farm-card {
    background: #ffffff;
    border: 1.5px solid #d1f0da;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 18px;
    box-shadow: 0 2px 16px rgba(22,163,74,0.06);
}
.farm-card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    color: #16a34a;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 14px;
}

/* ── Prediction result ── */
.result-pill {
    display: inline-block;
    background: linear-gradient(135deg, #16a34a, #22c55e);
    color: white;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    padding: 10px 28px;
    border-radius: 50px;
    box-shadow: 0 4px 16px rgba(22,163,74,0.3);
    margin-bottom: 12px;
}
.confidence-big {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: #16a34a;
    line-height: 1;
}
.confidence-sub {
    font-size: 0.78rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
}

/* ── Sidebar category pills ── */
.cat-pill {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 14px;
    border-radius: 10px;
    margin-bottom: 6px;
    font-size: 0.85rem;
    font-weight: 500;
    background: #f0faf2;
    color: #1a3d2b;
    border: 1px solid #d1f0da;
}
.cat-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── Uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #86efac !important;
    border-radius: 14px !important;
    background: #f0faf2 !important;
    padding: 8px !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    background: linear-gradient(135deg, #16a34a, #22c55e) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 32px !important;
    box-shadow: 0 4px 16px rgba(22,163,74,0.3) !important;
    width: 100% !important;
    letter-spacing: 0.04em !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 24px rgba(22,163,74,0.45) !important;
    transform: translateY(-1px) !important;
}

/* ── GradCAM section ── */
.gradcam-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    color: #16a34a;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 8px;
    text-align: center;
}

/* ── Footer ── */
.farm-footer {
    text-align: center;
    padding: 16px;
    color: #9ca3af;
    font-size: 0.78rem;
    border-top: 1px solid #d1f0da;
    margin-top: 32px;
    letter-spacing: 0.06em;
}
.farm-footer span { color: #16a34a; font-weight: 700; }

/* ── Metrics override ── */
[data-testid="stMetricValue"] { color: #16a34a !important; }
[data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 0.72rem !important; }

/* ── Tab bar ── */
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    color: #6b7280 !important;
}
.stTabs [aria-selected="true"] {
    color: #16a34a !important;
    border-bottom-color: #16a34a !important;
}

/* ── Watermarks ── */
.brand-topright {
    position: fixed; top: 10px; right: 20px;
    font-size: 15px; font-weight: 700;
    color: #16a34a; z-index: 1000;
    font-family: 'Space Grotesk', sans-serif;
    background: #f0faf2; padding: 4px 12px;
    border-radius: 20px; border: 1px solid #d1f0da;
}
.brand-center {
    position: fixed; top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    font-size: 48px; font-weight: 800;
    color: #16a34a; opacity: 0.04;
    z-index: 500; pointer-events: none;
    font-family: 'Space Grotesk', sans-serif;
}
[data-testid="stSidebar"]::before {
    content: "👁️ FARMEYE";
    position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -50%) rotate(-90deg);
    font-size: 28px; font-weight: 800;
    color: #16a34a; opacity: 0.07;
    white-space: nowrap; pointer-events: none;
    font-family: 'Space Grotesk', sans-serif;
}

/* bottom padding */
.main .block-container { padding-bottom: 80px !important; }
</style>

<div class="brand-topright">👁️ FARMEYE</div>
<div class="brand-center">👁️ FARMEYE</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# 5. Preprocessing (original, untouched)
# ════════════════════════════════════════════════════════════════
def preprocess_image(image: Image.Image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ════════════════════════════════════════════════════════════════
# 6. Grad-CAM helpers
# ════════════════════════════════════════════════════════════════
def make_gradcam_heatmap(img_array, model, conv_layer):
    """
    Build a sub-model from model.inputs → (last_conv output, softmax output).
    Uses the layer object directly and model.layers[-1].output to avoid
    any graph-access issues with nested Sequential models.
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.layers[-1].output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_img: Image.Image, heatmap: np.ndarray, alpha=0.45):
    """Overlay Grad-CAM heatmap using PIL + matplotlib — no cv2 needed."""
    img = np.array(original_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32)
    heatmap_pil = Image.fromarray(np.uint8(255 * heatmap)).resize(
        (IMG_SIZE, IMG_SIZE), Image.BILINEAR
    )
    heatmap_resized = np.array(heatmap_pil) / 255.0
    cmap = plt.get_cmap("jet")
    colormap = np.uint8(cmap(heatmap_resized)[:, :, :3] * 255).astype(np.float32)
    superimposed = (1 - alpha) * img + alpha * colormap
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    return Image.fromarray(superimposed)

# ════════════════════════════════════════════════════════════════
# 7. Sidebar
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding: 20px 8px 12px 8px;'>
        <div style='font-family:"Space Grotesk",sans-serif;font-size:1.25rem;
                    font-weight:800;color:#16a34a;'>👁️ FARMEYE</div>
        <div style='font-size:0.75rem;color:#6b7280;margin-top:2px;'>
            Maize Disease Classifier
        </div>
    </div>
    <hr style='border:1px solid #d1f0da;margin:0 0 16px 0'>
    """, unsafe_allow_html=True)

    st.markdown("**SUPPORTED CLASSES**", help="4 disease categories the model can detect")

    cat_colors = {"Blight": "#ef4444", "Common_Rust": "#f97316",
                  "Gray_Leaf_Spot": "#eab308", "Healthy": "#22c55e"}
    for cls in CLASS_NAMES:
        dot_color = cat_colors.get(cls, "#16a34a")
        st.markdown(f"""
        <div class='cat-pill'>
            <div class='cat-dot' style='background:{dot_color};'></div>
            {cls.replace('_', ' ')}
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**PROJECT INTRO**")
    st.info("This project detects maize leaf diseases using deep learning.")
    st.markdown("**DATASET**")
    st.info("Corn/Maize Leaf Disease Dataset (Kaggle).")
    st.markdown("**INFO**")
    st.info("Upload a maize leaf image to classify into Blight, Common Rust, Gray Leaf Spot, or Healthy.")

    # Debug panel — shows every layer so we can find the right one
    st.markdown("---")
    if last_conv_layer is not None:
        st.success(f"✅ Grad-CAM layer: `{last_conv_layer.name}`")
    else:
        st.error("❌ No Conv2D found — Grad-CAM unavailable.")
    with st.expander("🔍 All model layers (debug)"):
        def list_layers(m, indent=0):
            rows = []
            for layer in m.layers:
                prefix = "  " * indent
                has_out = "✅" if hasattr(layer, "_inbound_nodes") and layer._inbound_nodes else "❌"
                rows.append(f"{prefix}{has_out} [{layer.__class__.__name__}] {layer.name}")
                if hasattr(layer, "layers"):
                    rows.extend(list_layers(layer, indent + 1))
            return rows
        st.code("\n".join(list_layers(model)))

# ════════════════════════════════════════════════════════════════
# 8. Header
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div class='farm-header'>
    <div>
        <div class='farm-header-title'>👁️ FARMEYE</div>
        <div class='farm-header-sub'>Upload a maize leaf photo — the model will predict the disease instantly.</div>
    </div>
    <div class='farm-header-badge'>🌽 Maize Disease Detection</div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# 9. Stat tiles
# ════════════════════════════════════════════════════════════════
t1, t2, t3 = st.columns(3)
with t1:
    st.markdown("""<div class='stat-tile'>
        <div class='stat-tile-value'>4</div>
        <div class='stat-tile-label'>Categories</div>
    </div>""", unsafe_allow_html=True)
with t2:
    st.markdown("""<div class='stat-tile'>
        <div class='stat-tile-value'>256px</div>
        <div class='stat-tile-label'>Input Size</div>
    </div>""", unsafe_allow_html=True)
with t3:
    predictions_made = st.session_state.get("predictions_made", 0)
    st.markdown(f"""<div class='stat-tile'>
        <div class='stat-tile-value'>{predictions_made}</div>
        <div class='stat-tile-label'>Predictions Made</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# 10. Main layout — Upload left | Results right
# ════════════════════════════════════════════════════════════════
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("<div class='farm-card'><div class='farm-card-title'>📤 Upload Image</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("🌽 Choose a maize leaf image... 🥬", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 Predict")
    else:
        st.info("👆 Upload an image to get started")
        predict_btn = False

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='farm-card'><div class='farm-card-title'>🔬 Prediction Results</div>", unsafe_allow_html=True)

    if uploaded_file is not None and predict_btn:
        with st.spinner("Classifying..."):
            processed_img = preprocess_image(image)
            preds = model.predict(processed_img, verbose=0)[0]
            class_idx = np.argmax(preds)
            confidence = np.max(preds)

            st.session_state["predictions_made"] = st.session_state.get("predictions_made", 0) + 1
            st.session_state["last_result"] = {
                "preds": preds, "class_idx": class_idx,
                "confidence": confidence, "image": image
            }

    result = st.session_state.get("last_result", None)

    if result is None:
        st.markdown("""
        <div style='text-align:center;padding:48px 0;color:#9ca3af;'>
            <div style='font-size:3rem;margin-bottom:12px;'>🌿</div>
            <div style='font-size:0.9rem;'>Upload an image and click Predict</div>
        </div>""", unsafe_allow_html=True)
    else:
        preds      = result["preds"]
        class_idx  = result["class_idx"]
        confidence = result["confidence"]

        st.markdown(f"""
        <div style='text-align:center;margin-bottom:20px;'>
            <div style='font-size:0.75rem;color:#6b7280;text-transform:uppercase;
                        letter-spacing:0.1em;margin-bottom:8px;font-weight:600;'>
                DETECTED CLASS
            </div>
            <div class='result-pill'>{CLASS_NAMES[class_idx].replace('_', ' ')}</div>
            <br>
            <div class='confidence-big'>{confidence*100:.2f}%</div>
            <div class='confidence-sub'>Model Confidence</div>
        </div>
        """, unsafe_allow_html=True)

        st.write("### Probabilities")
        prob_df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": preds * 100
        })
        prob_df = prob_df.sort_values("Probability", ascending=False).reset_index(drop=True)
        colors = ["seagreen" if c == CLASS_NAMES[class_idx] else "gold" for c in prob_df["Class"]]

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(prob_df["Class"], prob_df["Probability"], color=colors)
        ax.set_xlabel("Probability (%)")
        ax.set_xlim(0, 100)
        for i, (cls, v) in enumerate(zip(prob_df["Class"], prob_df["Probability"])):
            ax.text(v + 1, i, f"{v:.2f}%", va="center", color="black")
        fig.patch.set_facecolor("#f0faf2")
        ax.set_facecolor("#f0faf2")
        st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# 11. Grad-CAM section (below, full width)
# ════════════════════════════════════════════════════════════════
result = st.session_state.get("last_result", None)
if result is not None:
    st.markdown("<div class='farm-card'>", unsafe_allow_html=True)
    st.markdown("<div class='farm-card-title'>🧠 Grad-CAM — Model Attention Heatmap</div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.85rem;color:#6b7280;margin-bottom:18px;'>"
        "Grad-CAM highlights the leaf regions the model focused on when making its prediction. "
        "<b style='color:#16a34a;'>Red/yellow areas</b> = high attention zones."
        "</p>", unsafe_allow_html=True
    )

    if last_conv_layer is None:
        st.warning("No Conv2D layer was found in this model. Grad-CAM is unavailable.")
    else:
        with st.spinner("Generating Grad-CAM..."):
            try:
                img_array_tf = tf.cast(preprocess_image(result["image"]), tf.float32)
                heatmap = make_gradcam_heatmap(img_array_tf, model, last_conv_layer)
                gradcam_img = overlay_gradcam(result["image"], heatmap)

                g1, g2 = st.columns(2)
                with g1:
                    st.markdown("<div class='gradcam-label'>Original Image</div>", unsafe_allow_html=True)
                    st.image(result["image"], use_container_width=True)
                with g2:
                    st.markdown("<div class='gradcam-label'>Grad-CAM Overlay</div>", unsafe_allow_html=True)
                    st.image(gradcam_img, use_container_width=True)
            except Exception as e:
                st.error(f"Grad-CAM error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# 12. Footer TM (original, untouched)
# ════════════════════════════════════════════════════════════════
st.markdown(
    "<hr style='border:1px solid #d1f0da'>"
    "<p style='text-align:center; color:gray;'>Powered by TechCrush Cohort 6 Group36 ™</p>",
    unsafe_allow_html=True
)