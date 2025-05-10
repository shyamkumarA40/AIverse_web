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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# JAFFE dataset directory (should be in root or provide full path)
JAFFE_DIR_PATH = "jaffedbase/jaffe/"

# Expression labels and mappings
expres_code = ['NE', 'HA', 'AN', 'DI', 'FE', 'SA', 'SU']
expres_label = ['Neutral', 'Happy', 'Angry', 'Disgust', 'Fear', 'Sad', 'Surprise']

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def read_data(dir_path):
    img_data_list = []
    labels = []
    img_list = [f for f in os.listdir(dir_path) if f.endswith(".tiff")]
    for img in img_list:
        input_img = cv2.imread(os.path.join(dir_path, img), cv2.IMREAD_GRAYSCALE)
        if input_img is None:
            continue
        img_data_list.append(input_img)
        label = img[3:5]
        labels.append(expres_code.index(label))
    return np.array(img_data_list), labels

def angle_line_x_axis(point1, point2):
    angle_r = math.atan2(point1[1] - point2[1], point1[0] - point2[0])
    return angle_r * 180 / math.pi

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

def detect_eyes_mediapipe(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None, None
    face_landmarks = results.multi_face_landmarks[0]
    left_eye = (int(face_landmarks.landmark[33].x * image.shape[1]),
                int(face_landmarks.landmark[33].y * image.shape[0]))
    right_eye = (int(face_landmarks.landmark[263].x * image.shape[1]),
                 int(face_landmarks.landmark[263].y * image.shape[0]))
    return left_eye, right_eye

def preprocess(images):
    normalized_faces = []
    for gray in images:
        left_eye, right_eye = detect_eyes_mediapipe(gray)
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

@st.cache_data(show_spinner=True)
def load_model():
    X, Y = read_data(JAFFE_DIR_PATH)
    cropped_X = preprocess(X)
    if not cropped_X:
        raise ValueError("No valid faces detected. Please check the preprocessing steps.")
    LL_images = apply_wavelet_transform(cropped_X)
    X_flat = from_2d_to_1d(LL_images)
    scaler = StandardScaler().fit(X_flat)
    X_scaled = scaler.transform(X_flat)
    pca = PCA(n_components=50).fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    X_tr, X_ts, y_tr, y_ts = train_test_split(X_pca, Y, test_size=0.2, random_state=42)
    model = SVC(C=5, gamma='scale', kernel='rbf', probability=True)
    model.fit(X_tr, y_tr)
    return model, scaler, pca

# ---------------------- Streamlit UI ----------------------
st.title("Facial Expression Recognition (JAFFE + MediaPipe)")
st.sidebar.header("Upload a Facial Image")
uploaded_file = st.sidebar.file_uploader("Choose a grayscale face image", type=["jpg", "png", "jpeg", "tiff"])

try:
    model, scaler, pca = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    gray_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    processed_faces = preprocess([gray_img])
    if not processed_faces:
        st.warning("Face not detected properly. Try another image.")
    else:
        face = processed_faces[0]
        LL = apply_wavelet_transform([face])[0]
        flat = from_2d_to_1d([LL])
        flat_scaled = scaler.transform(flat)
        flat_pca = pca.transform(flat_scaled)
        pred = model.predict(flat_pca)
        pred_prob = model.predict_proba(flat_pca)
        confidence = np.max(pred_prob) * 100
        st.success(f"Predicted Expression: **{expres_label[pred[0]]}** ({confidence:.2f}% confidence)")





















