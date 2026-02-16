import sys
import subprocess
import importlib.util
from pathlib import Path
import logging
import time

# ------------------------------------------
# ✅ ROBUST CLIP INSTALLATION
# ------------------------------------------
def ensure_clip_installed():
    """Ensure CLIP is installed with proper dependencies"""
    # First ensure setuptools is available
    try:
        import pkg_resources
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools==75.8.0"])
    
    # Check if clip is installed
    clip_spec = importlib.util.find_spec("clip")
    if clip_spec is None:
        import streamlit as st
        with st.spinner("Installing CLIP (this may take a minute)..."):
            try:
                # Try installing with pre-built dependencies first
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "ftfy", "regex", "tqdm"
                ])
                # Then install CLIP from GitHub
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "git+https://github.com/openai/CLIP.git"
                ])
            except Exception as e:
                st.error(f"Failed to install CLIP: {e}")
                st.stop()

# Run the installation check
ensure_clip_installed()

# Now import CLIP
import clip
import torch
import joblib
import streamlit as st
from PIL import Image, UnidentifiedImageError

st.set_page_config(page_title="Multimodal Misinformation Detector")

# ------------------------------------------
# ✅ Robust paths (works on Streamlit Cloud)
# ------------------------------------------
BASE_DIR = Path(__file__).resolve().parent          # .../demo
PROJECT_ROOT = BASE_DIR.parent                      # .../(project root)
MODEL_PATH = BASE_DIR / "model.pkl"                 # .../demo/model.pkl (simpler!)

DEVICE = "cpu"

logger = logging.getLogger("mbd_app")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def confidence_band(prob: float) -> str:
    if prob < 0.33:
        return "Low"
    if prob < 0.66:
        return "Medium"
    return "High"


@st.cache_resource(show_spinner=False)
def load_clip_model(device: str):
    """Load CLIP model with proper error handling"""
    try:
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        logger.info("CLIP loaded successfully")
        return model, preprocess
    except Exception as e:
        logger.error(f"Failed to load CLIP: {e}")
        st.error(f"Failed to load CLIP model: {e}")
        st.stop()


@st.cache_resource(show_spinner=False)
def load_classifier(model_path: str):
    """Load trained classifier"""
    try:
        model = joblib.load(model_path)
        logger.info("Classifier loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load classifier: {e}")
        st.error(f"Failed to load classifier: {e}")
        st.stop()


def prepare_features(image: Image.Image, text: str, clip_model, preprocess):
    """Prepare features exactly like training"""
    image_input = preprocess(image).unsqueeze(0)
    text_input = clip.tokenize([text])

    with torch.no_grad():
        img_emb = clip_model.encode_image(image_input)
        txt_emb = clip_model.encode_text(text_input)

    # Normalize embeddings (important!)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    
    # Concatenate exactly like training
    features = torch.cat([img_emb, txt_emb], dim=1).cpu().numpy()
    return features


def predict_label(features, classifier):
    """Get prediction and probability"""
    pred = int(classifier.predict(features)[0])

    prob = None
    if hasattr(classifier, "predict_proba"):
        prob = float(classifier.predict_proba(features)[0][1])

    return pred, prob


# ------------------------------------------
# UI
# ------------------------------------------
st.title("Multimodal Misinformation Detector")
st.caption("Upload an image and related text to estimate misinformation risk.")
show_debug = st.sidebar.checkbox("Show debug panel", value=False)

if show_debug:
    with st.sidebar.expander("Debug", expanded=True):
        st.write(f"Device: `{DEVICE}`")
        st.write(f"Project root: `{PROJECT_ROOT}`")
        st.write(f"Model path: `{MODEL_PATH}`")
        st.write(f"Model exists: `{MODEL_PATH.exists()}`")

if not MODEL_PATH.exists():
    logger.error("Missing model file at %s", MODEL_PATH)
    st.error(f"Missing model file: {MODEL_PATH}")
    st.stop()

try:
    with st.spinner("Loading CLIP and classifier..."):
        clip_model, preprocess = load_clip_model(DEVICE)
        clf = load_classifier(str(MODEL_PATH))
    logger.info("Models loaded successfully. device=%s model_path=%s", DEVICE, MODEL_PATH)
except Exception as exc:
    logger.exception("Model loading failed")
    st.error(f"Failed to load models: {exc}")
    st.stop()

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
text_input = st.text_area("Enter accompanying text", placeholder="Write the caption/post text here...")
run_clicked = st.button("Run prediction", type="primary")

if run_clicked:
    cleaned_text = text_input.strip()

    if uploaded_image is None:
        st.warning("Please upload an image.")
        st.stop()

    if not cleaned_text:
        st.warning("Please enter non-empty text.")
        st.stop()

    try:
        image = Image.open(uploaded_image).convert("RGB")
    except UnidentifiedImageError:
        logger.warning("Rejected invalid image upload")
        st.error("The uploaded file is not a valid image.")
        st.stop()
    except Exception as exc:
        logger.exception("Image read failure")
        st.error(f"Could not read image: {exc}")
        st.stop()

    try:
        start_time = time.perf_counter()
        features = prepare_features(image, cleaned_text, clip_model, preprocess)
        pred, prob = predict_label(features, clf)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "Inference complete. pred=%s prob=%s text_chars=%s elapsed_ms=%.2f",
            pred,
            "N/A" if prob is None else f"{prob:.4f}",
            len(cleaned_text),
            elapsed_ms,
        )
    except Exception as exc:
        logger.exception("Inference failed")
        st.error(f"Inference failed: {exc}")
        st.stop()

    st.subheader("Prediction")
    
    # Custom styling for prediction
    if pred == 1:
        st.markdown("### ⚠️ **Likely Misinformation**")
    else:
        st.markdown("### ✅ **Likely Consistent**")

    st.subheader("Misinformation Probability")
    if prob is None:
        st.info("Probability is unavailable for this classifier.")
    else:
        bounded_prob = max(0.0, min(1.0, prob))
        band = confidence_band(bounded_prob)
        
        # Color-coded progress bar
        if bounded_prob > 0.66:
            st.progress(bounded_prob, text="High risk")
        elif bounded_prob > 0.33:
            st.progress(bounded_prob, text="Medium risk")
        else:
            st.progress(bounded_prob, text="Low risk")
            
        st.caption(f"Estimated probability: {bounded_prob * 100:.1f}%")
        st.caption(f"Confidence band: {band}")

    if show_debug:
        with st.sidebar.expander("Last run", expanded=True):
            st.write(f"Uploaded file: `{uploaded_image.name}`")
            st.write(f"Text length: `{len(cleaned_text)}` chars")
            st.write(f"Feature shape: `{features.shape}`")
            st.write(f"Prediction: `{pred}`")
            st.write(f"Probability: `{'N/A' if prob is None else f'{prob:.4f}'}`")
            st.write(f"Inference time: `{elapsed_ms:.2f} ms`")

    st.subheader("Why this decision?")
    st.write(
        "The prediction comes from a classifier trained on multimodal examples, "
        "using both visual and textual CLIP embeddings. The model learns patterns "
        "from real misinformation data, where even semantically aligned image-text "
        "pairs can indicate misinformation if the text lacks context or uses "
        "sensational framing."
    )
    
    if prob is not None:
        st.caption(f"Model confidence: {abs(prob - 0.5) * 2:.1f} / 1.0")