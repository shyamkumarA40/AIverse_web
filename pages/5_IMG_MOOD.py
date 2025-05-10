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

# JAFFE dataset directory
JAFFE_DIR_PATH = "jaffedbase/jaffe/"

# Expression labels
expres_code = ['NE', 'HA', 'AN', 'DI', 'FE', 'SA', 'SU']
expres_label = ['Neutral', 'Happy', 'Angry', 'Disgust', 'Fear', 'Sad', 'Surprise']

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def read_data(dir_path):
    img_data_list = []
    labels = []
    img_list = os.listdir(dir_path)
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
    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return None, None
    face_landmarks = results.multi_face_landmarks[0]
    left_eye = (face_landmarks.landmark[33].x * image.shape[1], face_landmarks.landmark[133].y * image.shape[0])
    right_eye = (face_landmarks.landmark[362].x * image.shape[1], face_landmarks.landmark[263].y * image.shape[0])
    return left_eye, right_eye

def preprocess(images):
    normalized_faces = []
    for gray in images:
        print(f"[DEBUG] Original shape: {gray.shape}")
        if len(gray.shape) == 2:  # convert grayscale to BGR
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        left_eye, right_eye = detect_eyes_mediapipe(gray)
        if left_eye is None or right_eye is None:
            print("No eyes detected.")
            continue

        angle = angle_line_x_axis(left_eye, right_eye)
        rotated_img = rotate_image(gray, angle)
        D = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
        center = [(left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2]
        x, y = int(center[0] - (0.9 * D)), int(center[1] - (0.6 * D))
        w, h = int(1.8 * D), int(2.2 * D)

        face_roi = rotated_img[y:y + h, x:x + w]
        if face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
            print("Invalid face region.")
            continue

        face_roi = cv2.resize(face_roi, (96, 128))
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi = cv2.equalizeHist(face_roi)
        normalized_faces.append(face_roi)

    print(f"[DEBUG] Valid faces detected: {len(normalized_faces)}")
    return normalized_faces

def apply_wavelet_transform(images):
    return [pywt.dwt2(img, 'bior1.3')[0] for img in images]

def from_2d_to_1d(images):
    return np.array([img.reshape(-1) for img in images])

# Streamlit UI
st.title("Facial Expression Recognition (JAFFE Dataset) - MediaPipe Version")
st.sidebar.header("Upload a Facial Image")
uploaded_file = st.sidebar.file_uploader("Choose a grayscale face image", type=["jpg", "png", "jpeg", "tiff"])

@st.cache_resource
def load_model():
    try:
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
        X_tr, X_ts, y_tr, y_ts = train_test_split(X_pca, Y, test_size=0.2, random_state=42)
        model = SVC(C=1, gamma=0.01, kernel='linear')
        model.fit(X_tr, y_tr)
        return model, scaler, pca
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, scaler, pca = load_model()

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    gray_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    st.image(gray_img, caption="Uploaded Image", use_container_width=True, channels="GRAY")

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
        st.success(f"Predicted Expression: {expres_label[pred[0]]}")


















