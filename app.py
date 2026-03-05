import torch
import streamlit as st
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification

st.set_page_config(
    page_title="HandNet — ASL Detector",
    page_icon="🤟",
    layout="centered",
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    h1 {
        text-align: center;
        font-size: 2.8rem !important;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #a0aec0;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    .stCameraInput > div {
        border: 2px dashed #a78bfa;
        border-radius: 16px;
        padding: 10px;
    }
    .result-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(167, 139, 250, 0.35);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 1.5rem;
        backdrop-filter: blur(10px);
        text-align: center;
    }
    .big-letter { font-size: 6rem; line-height: 1; margin-bottom: 0.3rem; }
    .label-text { font-size: 1.3rem; color: #a78bfa; font-weight: 700; }
    .confidence-text { font-size: 0.95rem; margin-top: 0.4rem; }
    hr { border-color: rgba(167, 139, 250, 0.3); }
</style>
""", unsafe_allow_html=True)

MODEL_ID = "prithivMLmods/Alphabet-Sign-Language-Detection"

LABELS = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G",
    7: "H", 8: "I", 9: "J", 10: "K", 11: "L", 12: "M", 13: "N",
    14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
    20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z",
}


@st.cache_resource(show_spinner="Loading model from HuggingFace…")
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = SiglipForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


def predict(image: Image.Image):
    processor, model = load_model()
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    top_idx = int(torch.argmax(torch.tensor(probs)).item())
    label = LABELS[top_idx]
    confidence = probs[top_idx]

    sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    top5 = {LABELS[i]: probs[i] for i in sorted_indices[:5]}

    return label, confidence, top5


import torch
import streamlit as st
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification

st.set_page_config(
    page_title="HandNet — ASL Detector",
    page_icon="🤟",
    layout="centered",
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    h1 {
        text-align: center;
        font-size: 2.8rem !important;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #a0aec0;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    .stCameraInput > div {
        border: 2px dashed #a78bfa;
        border-radius: 16px;
        padding: 10px;
    }
    .result-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(167, 139, 250, 0.35);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 1.5rem;
        backdrop-filter: blur(10px);
        text-align: center;
    }
    .big-letter { font-size: 6rem; line-height: 1; margin-bottom: 0.3rem; }
    .label-text { font-size: 1.3rem; color: #a78bfa; font-weight: 700; }
    .confidence-text { font-size: 0.95rem; margin-top: 0.4rem; }
    hr { border-color: rgba(167, 139, 250, 0.3); }
</style>
""", unsafe_allow_html=True)

MODEL_ID = "prithivMLmods/Alphabet-Sign-Language-Detection"

LABELS = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G",
    7: "H", 8: "I", 9: "J", 10: "K", 11: "L", 12: "M", 13: "N",
    14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
    20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z",
}


@st.cache_resource(show_spinner="Loading model from HuggingFace…")
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = SiglipForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


def predict(image: Image.Image):
    processor, model = load_model()
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    top_idx = int(torch.argmax(torch.tensor(probs)).item())
    label = LABELS[top_idx]
    confidence = probs[top_idx]

    sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    top5 = {LABELS[i]: probs[i] for i in sorted_indices[:5]}

    return label, confidence, top5


st.title("🤟 HandNet")

st.markdown('<div class="subtitle">Show a hand sign to detect the ASL letter</div>', unsafe_allow_html=True)

camera_image = st.camera_input("Take a picture")

if camera_image is not None:
    image = Image.open(camera_image)

    st.image(image, caption="Captured image", use_container_width=True)

    with st.spinner("Analyzing sign..."):
        label, confidence, top5 = predict(image)

    st.markdown(
        f"""
        <div class="result-box">
            <div class="big-letter">{label}</div>
            <div class="label-text">Detected Letter</div>
            <div class="confidence-text">Confidence: {confidence:.2%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.subheader("Top predictions")

    for letter, score in top5.items():
        st.write(f"{letter} — {score:.2%}")
        st.progress(score)



