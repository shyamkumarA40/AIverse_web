import streamlit as st
import cv2
import dlib
import numpy as np
import pywt
import math
from imutils import face_utils
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# File Paths
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"
JAFFE_DIR_PATH = "jaffedbase/jaffe/"

# Load Dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)

# Expression Labels
expres_code = ['NE', 'HA', 'AN', 'DI', 'FE', 'SA', 'SU']
expres_label = ['Neutral', 'Happy', 'Angry', 'Disgust', 'Fear', 'Sad', 'Surprise']

# Data Loader
def read_data(dir_path):
    img_data_list, labels = [], []
    for img in os.listdir(dir_path):
        path = os.path.join(dir_path, img)
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            img_data_list.append(gray)
            label = img[3:5]
            labels.append(expres_code.index(label))
    return np.array(img_data_list), labels

# Rotation and Alignment Helpers
def rotate_image(image, angle):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

def angle_line_x_axis(p1, p2):
    return math.atan2(p1[1] - p2[1], p1[0] - p2[0]) * 180 / math.pi

def detect_eyes(gray):
    faces = detector(gray, 1)
    for rect in faces:
        shape = face_utils.shape_to_np(predictor(gray, rect))
        left_eye = np.mean(shape[42:48], axis=0).astype("int")
        right_eye = np.mean(shape[36:42], axis=0).astype("int")
        return left_eye, right_eye
    return None, None

# Preprocessing
def preprocess(images):
    faces = []
    for gray in images:
        left_eye, right_eye = detect_eyes(gray)
        if left_eye is None or right_eye is None:
            continue
        angle = angle_line_x_axis(left_eye, right_eye)
        rotated = rotate_image(gray, angle)
        dist = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
        center = [(left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2]
        x, y = int(center[0] - 0.9 * dist), int(center[1] - 0.6 * dist)
        w, h = int(1.8 * dist), int(2.2 * dist)
        face = rotated[y:y+h, x:x+w]
        if face.shape[0] > 0 and face.shape[1] > 0:
            face = cv2.resize(face, (96, 128))
            face = cv2.equalizeHist(face)
            faces.append(face)
    return faces

# Wavelet Transform
def apply_wavelet_transform(images):
    return [pywt.dwt2(img, 'bior1.3')[0] for img in images]

def from_2d_to_1d(images):
    return np.array([img.flatten() for img in images])

# Model Loader and Trainer
@st.cache_resource
def load_model():
    X, Y = read_data(JAFFE_DIR_PATH)
    processed_faces = preprocess(X)
    LL_faces = apply_wavelet_transform(processed_faces)
    X_flat = from_2d_to_1d(LL_faces)

    scaler = StandardScaler().fit(X_flat)
    X_scaled = scaler.transform(X_flat)
    pca = PCA(n_components=35).fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    X_train, _, y_train, _ = train_test_split(X_pca, Y, test_size=0.2, random_state=42)
    model = SVC(C=1, gamma=0.01, kernel='linear').fit(X_train, y_train)
    return model, scaler, pca

# Streamlit UI
st.set_page_config(page_title="IMG Mood AI", layout="centered")
st.title("🧠 IMG Mood AI: Emotion Detection")
st.markdown("Upload a facial image to detect the expressed emotion using JAFFE-trained AI.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

model, scaler, pca = load_model()

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    gray_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    st.image(gray_img, caption="Uploaded Image", use_column_width=True, channels="GRAY")

    processed_faces = preprocess([gray_img])
    if not processed_faces:
        st.warning("⚠️ Face not detected. Please try another image.")
    else:
        face = processed_faces[0]
        LL = apply_wavelet_transform([face])[0]
        flat = from_2d_to_1d([LL])
        flat_scaled = scaler.transform(flat)
        flat_pca = pca.transform(flat_scaled)
        pred = model.predict(flat_pca)
        st.success(f"🎭 Predicted Emotion: **{expres_label[pred[0]]}**")





