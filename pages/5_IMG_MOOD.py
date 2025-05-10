import streamlit as st
import numpy as np
import cv2
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import joblib
import pywt

# Load model, scaler, PCA
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        pca = joblib.load("pca.pkl")
    except:
        # Dummy fallback for local testing
        dummy_data = np.random.rand(35, 100)
        scaler = StandardScaler().fit(dummy_data)
        pca = PCA(n_components=20).fit(scaler.transform(dummy_data))
        model = SVC().fit(pca.transform(scaler.transform(dummy_data)), np.random.randint(0, 7, 35))
    return model, scaler, pca

def detect_faces(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    cropped_faces = [gray[y:y+h, x:x+w] for (x, y, w, h) in faces]
    return cropped_faces

def apply_wavelet_transform(images):
    LL_images = []
    for img in images:
        img_resized = cv2.resize(img, (48, 48))
        coeffs2 = pywt.dwt2(img_resized, 'haar')
        LL, (_, _, _) = coeffs2
        LL_images.append(LL)
    return np.array(LL_images)

def from_2d_to_1d(images, expected_length=100):
    flat_images = [img.flatten() for img in images]
    processed = []
    for vec in flat_images:
        if len(vec) > expected_length:
            vec = vec[:expected_length]
        elif len(vec) < expected_length:
            vec = np.pad(vec, (0, expected_length - len(vec)))
        processed.append(vec)
    return np.array(processed)

# Streamlit App
st.set_page_config(page_title="IMG Mood AI", layout="centered")
st.title("🖼️ IMG Mood AI - Emotion Detector")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

model, scaler, pca = load_model()

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    faces = detect_faces(img)

    if not faces:
        st.warning("😕 No face detected. Please try another image.")
    else:
        st.success(f"✅ {len(faces)} face(s) detected")

        try:
            cropped_faces = faces[:5]
            LL_images = apply_wavelet_transform(cropped_faces)
            X_flat = from_2d_to_1d(LL_images, expected_length=scaler.mean_.shape[0])
            X_scaled = scaler.transform(X_flat)
            X_pca = pca.transform(X_scaled)

            predictions = model.predict(X_pca)
            emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

            for i, (face_img, pred) in enumerate(zip(cropped_faces, predictions)):
                st.image(cv2.resize(face_img, (96, 96)), caption=f"Face {i+1} - {emotion_labels[pred]}", use_container_width=True)

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")



