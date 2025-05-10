import streamlit as st
import cv2
import numpy as np
import pywt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Paths
JAFFE_DIR_PATH = "jaffedbase/jaffe/"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Expressions
expres_code = ['NE', 'HA', 'AN', 'DI', 'FE', 'SA', 'SU']
expres_label = ['Neutral', 'Happy', 'Angry', 'Disgust', 'Fear', 'Sad', 'Surprise']

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Load JAFFE dataset
def read_data(path):
    images, labels = [], []
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            label = expres_code.index(file[3:5])
            images.append(img)
            labels.append(label)
    return np.array(images), labels

# Detect and preprocess faces using OpenCV
def preprocess(images):
    processed = []
    for img in images:
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (96, 128))
            face = cv2.equalizeHist(face)
            processed.append(face)
            break  # Only take first face
    return processed

def apply_wavelet_transform(images):
    return [pywt.dwt2(img, 'haar')[0] for img in images]

def from_2d_to_1d(images):
    return np.array([img.flatten() for img in images])

@st.cache_resource
def load_model():
    X_raw, y = read_data(JAFFE_DIR_PATH)
    faces = preprocess(X_raw)
    LL_faces = apply_wavelet_transform(faces)
    X_flat = from_2d_to_1d(LL_faces)

    scaler = StandardScaler().fit(X_flat)
    X_scaled = scaler.transform(X_flat)
    pca = PCA(n_components=35).fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    X_train, _, y_train, _ = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    model = SVC(kernel='linear').fit(X_train, y_train)
    return model, scaler, pca

# UI
st.title("🧠 IMG Mood AI - Emotion Detection")
uploaded_file = st.file_uploader("Upload a facial image", type=["jpg", "jpeg", "png"])

model, scaler, pca = load_model()

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    st.image(gray, caption="Uploaded Image", use_container_width=True, channels="GRAY")

    faces = preprocess([gray])
    if not faces:
        st.warning("No face detected. Try another image.")
    else:
        face = faces[0]
        LL = apply_wavelet_transform([face])[0]
        flat = from_2d_to_1d([LL])
        scaled = scaler.transform(flat)
        reduced = pca.transform(scaled)
        pred = model.predict(reduced)
        st.success(f"🎭 Predicted Emotion: **{expres_label[pred[0]]}**")






