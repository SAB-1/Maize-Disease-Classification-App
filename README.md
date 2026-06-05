<!-- PROJECT LOGO -->
<div align="center">
  <h1>👁️ FARMEYE</h1>
  <p><b>Maize Disease Classification with AI-Powered Visual Analysis</b></p>
  <p>
    <a href="#about-the-project">About</a> •
    <a href="#features">Features</a> •
    <a href="#getting-started">Getting Started</a> •
    <a href="#usage">Usage</a> •
    <a href="#model-details">Model Details</a> •
    <a href="#technology-stack">Tech Stack</a>
  </p>
</div>

---

## About The Project

**FARMEYE** is an intelligent agricultural solution that uses deep learning to detect and classify maize leaf diseases in real-time. By uploading a leaf image, farmers can instantly receive:
- **Disease classification** (Blight, Common Rust, Gray Leaf Spot, or Healthy)
- **Confidence scores** for each prediction
- **Visual explainability** via Grad-CAM heatmaps showing which leaf regions influenced the diagnosis
- **Actionable insights** for crop management

This tool empowers farmers to make data-driven decisions, prevent crop loss, and optimize yield through early disease detection.

---

## Features

✨ **Core Capabilities**
- 🔍 **Real-time Disease Detection** – Classify maize leaf diseases instantly
- 📊 **High Precision (99.2%)** – Reliable model trained on extensive datasets
- 🧠 **Explainable AI (Grad-CAM)** – Visualize which leaf areas triggered the diagnosis
- 📱 **User-Friendly Interface** – Built with Streamlit for seamless interaction
- 🌾 **Farm-Themed UX** – Intuitive design tailored for agricultural users

🎯 **Supported Disease Classes**
1. **Blight** (🔴 Critical) – Fungal infection causing rapid leaf damage
2. **Common Rust** (🟠 Moderate) – Orange pustule formations on leaves
3. **Gray Leaf Spot** (🟡 Moderate) – Grayish lesions with dark borders
4. **Healthy** (🟢 Good) – No visible disease symptoms

📈 **Real-Time Metrics**
- Diseases Detected: 4 types
- Model Precision: 99.2%
- Crops Analyzed: Live counter of total scans

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip or conda
- 1.3 MB disk space for model

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SAB-1/Maize-Disease-Classification-App.git
   cd Maize-Disease-Classification-App
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the app**
   - Open your browser to `http://localhost:8501`

---

## Usage

### Step-by-Step Guide

1. **Upload a Maize Leaf Image**
   - Click the upload area or drag & drop a `.jpg`, `.jpeg`, or `.png` file
   - The image will be displayed for verification

2. **Click "Predict"**
   - The model processes your image (takes ~2-3 seconds)
   - Displays the detected disease class and confidence percentage

3. **Review Probabilities**
   - A bar chart shows the model's confidence for all 4 disease categories
   - Highest probability is highlighted in green

4. **Analyze Grad-CAM Heatmap**
   - View side-by-side comparison: Original image vs. Model Attention
   - **Red/Yellow regions** = High attention zones (areas the model focused on)
   - **Blue regions** = Low attention zones

5. **Track Your Progress**
   - "Crops Analyzed" counter increments with each prediction
   - Useful for batch-monitoring farm sections

### Example Workflow

```
📤 Upload Leaf Image
        ↓
🔍 Click "Predict"
        ↓
📊 View Disease Classification
        ↓
📈 Check Confidence Scores
        ↓
🧠 Analyze Grad-CAM Heatmap
        ↓
✅ Make Informed Decision
```

---

## Model Details

### Architecture

- **Type:** Convolutional Neural Network (CNN)
- **Base Model:** Transfer learning with deep architecture
- **Input:** 256×256 RGB images
- **Output:** 4-class probability distribution

### Training Data

- **Dataset:** Corn/Maize Leaf Disease Dataset (Kaggle)
- **Images:** Thousands of diverse leaf samples
- **Augmentation:** Rotations, flips, brightness adjustments
- **Train/Test Split:** Standard ML practices

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | ~99% |
| **Precision** | 99.2% |
| **Inference Time** | <3 seconds per image |
| **Model Size** | 1.3 MB |

### Grad-CAM Explainability

The app uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to show:
- Which convolutional features the model prioritized
- Spatial regions most influential to the prediction
- Why the model made its decision (transparency)

---

## Technology Stack

### Backend
- **TensorFlow/Keras** – Deep learning model & inference
- **NumPy/Pandas** – Data processing
- **Pillow** – Image manipulation
- **Matplotlib** – Visualization & Grad-CAM overlay

### Frontend
- **Streamlit** – Web UI framework
- **Custom CSS** – Farm-green theme styling
- **Real-time session state** – Dynamic metric tracking

### Deployment
- **Streamlit** – Lightweight, production-ready deployment
- **Compatible with:** Cloud platforms (Heroku, AWS, GCP, Streamlit Cloud)

---

## Project Structure

```
Maize-Disease-Classification-App/
├── app.py                      # Main Streamlit application
├── maize_disease.keras         # Pre-trained TensorFlow model
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## File Explanations

### `app.py`
Main application file containing:
- Model loading & caching (`@st.cache_resource`)
- Image preprocessing pipeline
- Grad-CAM heatmap generation
- Streamlit UI layout & styling
- Session state management for real-time metrics

### `maize_disease.keras`
Pre-trained Keras model (1.3 MB):
- Ready-to-use for inference
- Optimized for 256×256 RGB input
- No retraining required

### `requirements.txt`
```
streamlit
tensorflow
pillow
numpy
pandas
matplotlib
```

---

## How Grad-CAM Works

1. **Forward Pass:** Image → Model → Predictions
2. **Gradient Computation:** Calculate gradients of predicted class w.r.t. final conv layer
3. **Weighted Activation:** Multiply conv feature maps by gradient weights
4. **Heatmap Generation:** Average across channels to produce 2D attention map
5. **Overlay:** Blend heatmap with original image using "jet" colormap

**Result:** Visual explanation of model reasoning with **red/yellow = high importance** regions highlighted.

---

## Contributing

We welcome contributions! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Future Enhancements

🚀 **Planned Features:**
- Batch image processing (analyze multiple leaves at once)
- Disease severity scoring
- Treatment recommendations based on disease type
- Mobile app version
- Model fine-tuning with user feedback
- Multi-crop support (wheat, rice, etc.)
- Historical analysis & trend tracking

---

## Troubleshooting

### Common Issues

**Issue:** Model fails to load
- **Solution:** Ensure `maize_disease.keras` is in the same directory as `app.py`

**Issue:** Grad-CAM error "Cannot compute gradients"
- **Solution:** This is handled automatically with fallback to global average pooling

**Issue:** Slow inference
- **Solution:** Ensure TensorFlow GPU support is installed for faster processing

**Issue:** Image upload fails
- **Solution:** Use `.jpg`, `.jpeg`, or `.png` files. Max size: typically 25-100 MB (platform-dependent)

---

## Resources & References

- 📚 [Keras Documentation](https://keras.io)
- 📚 [Streamlit Docs](https://docs.streamlit.io)
- 📚 [Grad-CAM Paper](https://arxiv.org/abs/1610.02055)
- 📊 [Kaggle Maize Disease Dataset](https://www.kaggle.com/datasets/sprojeshamhat/corn-or-maize-leaf-disease-dataset)

---

## License

This project is open source and available under the MIT License.

---

## Authors & Acknowledgments

**Developed by:** TechCrush Cohort 6 Group36

**Special Thanks:**
- 🌾 Agricultural domain experts for insights
- 📊 Dataset contributors on Kaggle
- 🧠 Deep learning community for model architectures

---

## Contact & Support

For questions, issues, or suggestions:
- 📧 Open an Issue on GitHub
- 🤝 Submit a Pull Request
- 💬 Discuss in Discussions

---

<div align="center">
  <p><b>Made with ❤️ for farmers & agriculture</b></p>
  <p>👁️ FARMEYE – See the disease, save the crop.</p>
</div>
