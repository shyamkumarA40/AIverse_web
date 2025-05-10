import os
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils.feature_extraction import extract_features
from collections import Counter

def parse_label(filename):
    filename = filename.lower()

    # Very important: check for "_female_" before "_male_"
    if "_female_" in filename:
        gender = "female"
    elif "_male_" in filename:
        gender = "male"
    else:
        gender = "unknown"

    # Extract emotion label from filename
    emotion_match = re.search(r'(angry|happy|sad|neutral)', filename)
    emotion = emotion_match.group(1) if emotion_match else "neutral"

    return gender, emotion

def load_dataset(data_folder):
    features, genders, emotions = [], [], []
    for file in os.listdir(data_folder):
        if file.endswith(".wav"):
            path = os.path.join(data_folder, file)
            feat = extract_features(path)
            if feat is not None:
                gender, emotion = parse_label(file)
                features.append(feat)
                genders.append(gender)
                emotions.append(emotion)
    print("Loaded Gender Labels:", Counter(genders))
    print("Loaded Emotion Labels:", Counter(emotions))
    return np.array(features), np.array(genders), np.array(emotions)

def train_models(X, y_gender, y_emotion):
    print("Before split - Gender:", Counter(y_gender))
    print("Before split - Emotion:", Counter(y_emotion))

    # Stratified split for gender
    X_train, X_test, yg_train, yg_test, ye_train, ye_test = train_test_split(
        X, y_gender, y_emotion, test_size=0.2, random_state=42, stratify=y_gender
    )

    print("Train Gender:", Counter(yg_train))
    print("Test Gender:", Counter(yg_test))

    gender_model = RandomForestClassifier()
    emotion_model = RandomForestClassifier()

    gender_model.fit(X_train, yg_train)
    emotion_model.fit(X_train, ye_train)

    print("Gender Classification Report:")
    print(classification_report(yg_test, gender_model.predict(X_test)))

    print("Emotion Classification Report:")
    print(classification_report(ye_test, emotion_model.predict(X_test)))

    return gender_model, emotion_model

