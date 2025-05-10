import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Paths
JAFFE_DIR_PATH = "jaffedbase/jaffe/"

# Expressions
expres_code = ['NE', 'HA', 'AN', 'DI', 'FE', 'SA', 'SU']
expres_label = ['Neutral', 'Happy', 'Angry', 'Disgust', 'Fear', 'Sad', 'Surprise']

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize mediapipe Face Mesh model
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,  # Set to True for single image mode
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

def detect_eyes(gray):
    # Convert to RGB for mediapipe processing
    rgb_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    results = face_mesh.process(rgb_image)
    
    left_eye_center = None
    right_eye_center = None
    
    if results.multi_face_landmarks:
        # For the first face, get the landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Draw landmarks on the image for debugging
        annotated_image = rgb_image.copy()
        mp_drawing.draw_landmarks(annotated_image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

        # Extract the eye points (indices 33 to 133 for left eye, 133 to 233 for right eye)
        left_eye = face_landmarks.landmark[33:133]  # Left eye landmarks
        right_eye = face_landmarks.landmark[133:233]  # Right eye landmarks
        
        # Get the mean positions of the eyes
        left_eye_center = np.mean([(point.x, point.y) for point in left_eye], axis=0)
        right_eye_center = np.mean([(point.x, point.y) for point in right_eye], axis=0)

        # Visualize the annotated image for debugging purposes
        st.image(annotated_image, caption="Face Landmarks Detected", use_column_width=True)
    
    return left_eye_center, right_eye_center

def preprocess(images):
    normalized_faces = []
    for gray in images:
        left_eye, right_eye = detect_eyes(gray)
        if left_eye is None or right_eye is None:
            continue  # Skip if eyes are not detected
        normalized_faces.append(gray)  # For debugging, skip rotations for now
    return normalized_faces

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

    # Feature extraction code (use without transformations for debugging)
    X_flat = np.array([img.reshape(-1) for img in cropped_X])

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
        st.success("Face detected successfully.")



