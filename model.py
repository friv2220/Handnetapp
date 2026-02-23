import cv2
from transformers import pipeline
from PIL import Image

pipe = pipeline(
    "image-classification",
    model="prithivMLmods/Alphabet-Sign-Language-Detection",
    use_fast=True
)

def analyze_sign(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)

    result = pipe(img)

    best_prediction = result[0]
    label = best_prediction["label"]
    score = best_prediction["score"]

    return f"Prediction: {label} ({score:.2%})"

