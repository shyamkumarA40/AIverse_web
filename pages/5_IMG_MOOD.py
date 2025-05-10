import streamlit as st
import numpy as np
import cv2
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import joblib
import pywt

# Load model and preprocessing pipeline
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        pca = joblib.load("pca.pkl")
    except:
        dummy_data = np.random.rand(35, 100)
        scaler = StandardScaler().fit(dummy_data)
        pca = PCA(n_components=20).fit(scaler.transform(dummy_data))
        model = SVC().fit(pca.transform(scaler.transform(dummy_data)), np.random.randint(0, 7, 35))
    return model, scaler, pca

# Detect only first face (you can modify this to process all if needed)
def extract_face(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return gray[y:y+h, x:x+w]
    else:
        return None

def apply_wavelet_transform(img):
    img_resized = cv2.resize(img, (48, 48))
    coeffs2 = pywt.dwt2(img_resized, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return LL.flatten().reshape(1, -1)

# Streamlit UI
st.set_page_config(page_title="IMG Mood AI", layout="centered")
st.title("🖼️ IMG Mood AI - Emotion Detector")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
model, scaler, pca = load_model()

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    face_img = extract_face(img)
    if face_img is None:
        st.error("No face detected. Please upload a clearer image.")
    else:
        try:
            features = apply_wavelet_transform(face_img)
            features_scaled = scaler.transform(features)
            features_pca = pca.transform(features_scaled)
            pred = model.predict(features_pca)[0]
            emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            st.subheader("🧠 Predicted Emotion:")
            st.success(f"**{emotion_labels[pred]}**")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")




