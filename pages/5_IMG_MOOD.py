import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pywt
import os
import math
from imutils import face_utils
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Paths
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"  # No longer needed as we are using mediapipe
JAFFE_DIR_PATH = "jaffedbase/jaffe/"

# Expressions
expres_code = ['NE', 'HA', 'AN', 'DI', 'FE', 'SA', 'SU']
expres_label = ['Neutral', 'Happy', 'Angry', 'Disgust', 'Fear', 'Sad', 'Surprise']

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize mediapipe Face Mesh model
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,  # You can adjust this if you want to detect multiple faces
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Utility Functions
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

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

def angle_line_x_axis(point1, point2):
    angle_r = math.atan2(point1[1] - point2[1], point1[0] - point2[0])
    return angle_r * 180 / math.pi

def detect_eyes(gray):
    # Convert to RGB for mediapipe processing
    rgb_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    results = face_mesh.process(rgb_image)
    
    left_eye_center = None
    right_eye_center = None
    
    if results.multi_face_landmarks:
        # For the first face, get the landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract the eye points (indices 33 to 133 for left eye, 133 to 233 for right eye)
        left_eye = face_landmarks.landmark[33:133]  # Left eye landmarks
        right_eye = face_landmarks.landmark[133:233]  # Right eye landmarks
        
        # Get the mean positions of the eyes
        left_eye_center = np.mean([(point.x, point.y) for point in left_eye], axis=0)
        right_eye_center = np.mean([(point.x, point.y) for point in right_eye], axis=0)

    return left_eye_center, right_eye_center

def preprocess(images):
    normalized_faces = []
    for gray in images:
        left_eye, right_eye = detect_eyes(gray)
        if left_eye is None or right_eye is None:
            continue  # Skip if eyes are not detected
        angle = angle_line_x_axis(left_eye, right_eye)
        rotated_img = rotate_image(gray, angle)
        D = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
        D_point = [(left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2]
        x, y = int(D_point[0] - (0.9 * D)), int(D_point[1] - (0.6 * D))
        w, h = int(1.8 * D), int(2.2 * D)
        face_roi = rotated_img[y:y + h, x:x + w]
        if face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
            continue  # Skip if face region is invalid
        face_roi = cv2.resize(face_roi, (96, 128))
        face_roi = cv2.equalizeHist(face_roi)
        normalized_faces.append(face_roi)
    return normalized_faces

def apply_wavelet_transform(images):
    return [pywt.dwt2(img, 'bior1.3')[0] for img in images]

def from_2d_to_1d(images):
    return np.array([img.reshape(-1) for img in images])

# Streamlit UI
st.title("Facial Expression Recognition (JAFFE Dataset)")

st.sidebar.header("Upload a Facial Image")
uploaded_file = st.sidebar.file_uploader("Choose a grayscale face image", type=["jpg", "png", "jpeg"])

# Load and prepare model
@st.cache_resource
def load_model():
    X, Y = read_data(JAFFE_DIR_PATH)
    cropped_X = preprocess(X)
    
    if not cropped_X:  # Check if no faces are detected after preprocessing
        raise ValueError("No valid faces detected in the dataset. Ensure that the images are properly formatted.")

    LL_images = apply_wavelet_transform(cropped_X)
    X_flat = from_2d_to_1d(LL_images)

    if X_flat.shape[0] == 0:
        raise ValueError("No valid data available after preprocessing.")  # Ensure that data is not empty

    scaler = StandardScaler().fit(X_flat)
    X_scaled = scaler.transform(X_flat)
    pca = PCA(n_components=35).fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    X_tr, X_ts, y_tr, y_ts = train_test_split(X_pca, Y, test_size=0.2, random_state=42)
    model = SVC(C=1, gamma=0.01, kernel='linear')
    model.fit(X_tr, y_tr)
    return model, scaler, pca

try:
    model, scaler, pca = load_model()
except ValueError as e:
    st.error(f"Error: {e}")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    gray_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.image(gray_img, caption="Uploaded Image", use_column_width=True, channels="GRAY")

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


