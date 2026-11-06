import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps
import pandas as pd

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="😊")
st.title("😊 Facial Emotion Recognition")
st.write("Upload a face photo and let the trained CNN estimate its facial-expression category.")

MODEL_PATH = "best_fer_model.h5"
IMG_SIZE = 48
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

@st.cache_resource
def load_detector():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(path)
    if detector.empty():
        raise RuntimeError("Could not load OpenCV face detector.")
    return detector

uploaded = st.file_uploader("Choose a face photo", type=["jpg", "jpeg", "png"])

if uploaded:
    image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
    image_rgb = np.array(image)
    st.image(image, caption="Uploaded photo", use_container_width=True)

    if st.button("Predict Emotion", type="primary"):
        with st.spinner("Loading model and analysing image..."):
            model = load_model()
            detector = load_detector()
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
            )
            marked = image_rgb.copy()
            if len(faces):
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                face = gray[y:y+h, x:x+w]
                cv2.rectangle(marked, (x, y), (x+w, y+h), (0, 220, 80), 3)
                st.caption("Largest detected face selected.")
            else:
                face = gray
                st.warning("No face detected; using the full image instead.")

            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = (face.astype("float32") / 255.0).reshape(1, IMG_SIZE, IMG_SIZE, 1)
            probabilities = model.predict(face, verbose=0)[0]
            index = int(np.argmax(probabilities))

        st.subheader(f"Estimated expression: {EMOTION_LABELS[index]}")
        st.metric("Model confidence", f"{probabilities[index] * 100:.1f}%")
        st.image(marked, caption="Face detection result", use_container_width=True)
        st.subheader("Class probabilities")
        st.bar_chart(pd.DataFrame({
            "Confidence": probabilities.astype(float)
        }, index=EMOTION_LABELS))
        st.caption("This is a model estimate of visible facial-expression patterns, not a reliable reading of someone's actual feelings.")
else:
    st.info("Upload a clear, front-facing face photo to begin.")
