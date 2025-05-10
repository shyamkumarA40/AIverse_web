import streamlit as st
import cv2
import numpy as np
import pywt
import os
import math
import mediapipe as mp
import torch
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Dataset path
JAFFE_DIR_PATH = "jaffedbase/jaffe/"

# Expression codes
expres_code = ['NE', 'HA', 'AN', 'DI', 'FE', 'SA', 'SU']
expres_label = ['Neutral', 'Happy', 'Angry', 'Disgust', 'Fear', 'Sad', 'Surprise']

# MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def read_data(dir_path):
    img_data_list = []
    labels = []
    for img in os.listdir(dir_path):
        input_img = cv2.imread(os.path.join(dir_path, img), cv2.IMREAD_GRAYSCALE)
        if input_img is None:
            continue
        label = img[3:5]
        if label in expres_code:
            labels.append(expres_code.index(label))
            resized_img = cv2.resize(input_img, (256, 256))
            img_data_list.append(resized_img)
    return np.array(img_data_list), labels

def angle_line_x_axis(point1, point2):
    angle_r = math.atan2(point1[1] - point2[1], point1[0] - point2[0])
    return angle_r * 180 / math.pi

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

def detect_eyes_mediapipe(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return None, None
    face_landmarks = results.multi_face_landmarks[0]
    left_eye = (face_landmarks.landmark[33].x * image.shape[1], face_landmarks.landmark[33].y * image.shape[0])
    right_eye = (face_landmarks.landmark[263].x * image.shape[1], face_landmarks.landmark[263].y * image.shape[0])
    return left_eye, right_eye

def preprocess(images):
    normalized_faces = []
    for gray in images:
        color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        left_eye, right_eye = detect_eyes_mediapipe(color_img)
        if left_eye is None or right_eye is None:
            continue
        angle = angle_line_x_axis(left_eye, right_eye)
        rotated_img = rotate_image(gray, angle)
        D = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
        center = [(left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2]
        x, y = int(center[0] - (0.9 * D)), int(center[1] - (0.6 * D))
        w, h = int(1.8 * D), int(2.2 * D)
        face_roi = rotated_img[y:y + h, x:x + w]
        if face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
            continue
        face_roi = cv2.resize(face_roi, (96, 128))
        face_roi = cv2.equalizeHist(face_roi)
        normalized_faces.append(face_roi)
    return normalized_faces

def apply_wavelet_transform(images):
    return [pywt.dwt2(img, 'bior1.3')[0] for img in images]

def from_2d_to_1d(images):
    return np.array([img.reshape(-1) for img in images])

@st.cache_resource

def load_model():
    X, Y = read_data(JAFFE_DIR_PATH)
    cropped_X = preprocess(X)
    if not cropped_X:
        raise ValueError("No valid faces detected. Please check the preprocessing steps.")
    LL_images = apply_wavelet_transform(cropped_X)
    X_flat = from_2d_to_1d(LL_images)
    scaler = StandardScaler().fit(X_flat)
    X_scaled = scaler.transform(X_flat)
    pca = PCA(n_components=35).fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(X_pca, Y):
        X_tr, X_ts = X_pca[train_idx], X_pca[test_idx]
        y_tr, y_ts = np.array(Y)[train_idx], np.array(Y)[test_idx]
    model = SVC(C=1, gamma=0.01, kernel='linear')
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_ts)
    report = classification_report(y_ts, y_pred, target_names=expres_label)
    print("\nClassification Report:\n", report)

    return model, scaler, pca

# Streamlit UI
st.title("Facial Expression Recognition (JAFFE Dataset) - MediaPipe Version")
st.sidebar.header("Upload a Facial Image")
uploaded_file = st.sidebar.file_uploader("Choose a face image", type=["jpg", "jpeg", "png"])

try:
    model, scaler, pca = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (256, 256))
    st.image(resized_img, caption="Uploaded Image (Grayscale)", channels="GRAY")

    processed_faces = preprocess([resized_img])
    if not processed_faces:
        st.warning("Face not detected properly. Try another image.")
    else:
        face = processed_faces[0]
        st.image(face, caption="Preprocessed Face", channels="GRAY")
        LL = apply_wavelet_transform([face])[0]
        flat = from_2d_to_1d([LL])
        flat_scaled = scaler.transform(flat)
        flat_pca = pca.transform(flat_scaled)
        pred = model.predict(flat_pca)
        st.success(f"Predicted Expression: {expres_label[pred[0]]}")




























