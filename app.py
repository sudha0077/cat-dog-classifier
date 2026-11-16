from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "mobilenetv2_cat_dog.keras"
IMAGE_SIZE = (128, 128)
CLASS_NAMES = {0: "Cat", 1: "Dog"}

st.set_page_config(
    page_title="PawVision | Cat vs Dog AI",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { color-scheme: dark; }
.stApp {
  background:
    radial-gradient(circle at 12% 8%, rgba(99,102,241,.19), transparent 30%),
    radial-gradient(circle at 88% 18%, rgba(14,165,233,.14), transparent 28%),
    linear-gradient(135deg,#080d1a 0%,#10172a 55%,#0a1020 100%);
  color: #edf2ff;
  font-family: 'DM Sans', sans-serif;
}
.block-container {max-width: 1100px; padding-top: 2.4rem; padding-bottom: 3rem;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif !important; color:#f8faff;}
.hero {
  padding: 2rem 2.1rem; border: 1px solid rgba(148,163,184,.22);
  border-radius: 26px; background: rgba(15,23,42,.72);
  box-shadow: 0 18px 60px rgba(0,0,0,.25); margin-bottom: 1.5rem;
}
.eyebrow {color:#93c5fd; letter-spacing:.16em; text-transform:uppercase; font-size:.76rem; font-weight:700;}
.hero h1 {font-size: clamp(2.2rem,5vw,3.5rem); margin:.45rem 0;}
.hero p {color:#aab8d1; font-size:1.05rem; max-width:700px;}
.panel {
  background:rgba(15,23,42,.72); border:1px solid rgba(148,163,184,.2);
  border-radius:22px; padding:1.25rem; min-height:160px;
}
.metric {
  background:linear-gradient(135deg,rgba(99,102,241,.18),rgba(14,165,233,.12));
  border:1px solid rgba(129,140,248,.28); border-radius:20px;
  padding:1.25rem; text-align:center;
}
.prediction {font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:700;color:#e0e7ff;}
.muted {color:#9caec9;}
.stButton>button {
  border:0; border-radius:12px; padding:.65rem 1.1rem;
  background:linear-gradient(90deg,#6366f1,#0ea5e9); color:white; font-weight:700;
}
[data-testid="stFileUploader"] {
  background:rgba(15,23,42,.55); border:1px dashed rgba(147,197,253,.45);
  border-radius:18px; padding:1rem;
}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading the trained MobileNetV2 model...")
def load_classifier():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

st.markdown("""
<div class="hero">
  <div class="eyebrow">Computer Vision · Deep Learning</div>
  <h1>🐾 PawVision AI</h1>
  <p>Upload a picture and let the trained MobileNetV2 model predict whether it contains a cat or a dog.</p>
</div>
""", unsafe_allow_html=True)

model = load_classifier()
if model is None:
    st.error(
        "The trained model file is missing. Add "
        "`model/mobilenetv2_cat_dog.keras` to this Space repository, "
        "then restart the Space."
    )
    st.stop()

left, right = st.columns([1, 1], gap="large")
with left:
    st.markdown("### Upload an image")
    uploaded = st.file_uploader(
        "Choose a JPG, JPEG or PNG image",
        type=["jpg", "jpeg", "png"],
        help="For best results, use a clear image containing one main animal.",
    )
    st.markdown(
        '<p class="muted">Accepted formats: JPG · JPEG · PNG</p>',
        unsafe_allow_html=True,
    )

with right:
    st.markdown("### Prediction")
    if uploaded is None:
        st.markdown(
            '<div class="panel"><div class="prediction">Ready when you are ✨</div>'
            '<p class="muted">Your uploaded image and AI prediction will appear here.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded image", use_container_width=True)

            resized = image.resize(IMAGE_SIZE)
            array = np.asarray(resized, dtype=np.float32)
            batch = np.expand_dims(array, axis=0)

            # Match notebook preprocessing:
            # images were scaled to [0, 1], then MobileNetV2 preprocess_input(images * 255)
            processed = tf.keras.applications.mobilenet_v2.preprocess_input(batch)

            with st.spinner("Analyzing image..."):
                raw = model.predict(processed, verbose=0)
            dog_probability = float(np.asarray(raw).reshape(-1)[0])
            dog_probability = min(max(dog_probability, 0.0), 1.0)

            predicted_index = 1 if dog_probability >= 0.5 else 0
            predicted_label = CLASS_NAMES[predicted_index]
            confidence = dog_probability if predicted_index == 1 else 1.0 - dog_probability

            st.markdown(
                f'<div class="metric"><div class="muted">PREDICTED CLASS</div>'
                f'<div class="prediction">{ "🐶" if predicted_index == 1 else "🐱" } {predicted_label}</div>'
                f'<div class="muted">Confidence: {confidence:.1%}</div></div>',
                unsafe_allow_html=True,
            )
            st.progress(confidence, text=f"Model confidence · {confidence:.1%}")
            c1, c2 = st.columns(2)
            c1.metric("Cat probability", f"{1-dog_probability:.1%}")
            c2.metric("Dog probability", f"{dog_probability:.1%}")
            st.caption("Confidence is the model's predicted probability for the selected class, not a guarantee of correctness.")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            st.error(f"Could not read this image. Please upload a valid JPG or PNG. Details: {exc}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

st.markdown("---")
st.markdown(
    '<p class="muted" style="text-align:center">PawVision AI · Cat vs Dog Image Classification · MobileNetV2</p>',
    unsafe_allow_html=True,
)
