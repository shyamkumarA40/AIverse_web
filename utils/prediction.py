from utils.feature_extraction import extract_features

def predict(file_path, gender_model, emotion_model):
    # Extract features from the audio file
    features = extract_features(file_path)

    # Check if feature extraction succeeded
    if features is None:
        print(f"[ERROR] Could not extract features from {file_path}")
        return None, None

    try:
        # Reshape if necessary (some models may expect 2D input)
        features = features.reshape(1, -1)

        # Predict gender and emotion
        gender = gender_model.predict(features)[0]
        emotion = emotion_model.predict(features)[0]

        return gender, emotion

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return None, None

