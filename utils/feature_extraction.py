import librosa
import numpy as np

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
        return np.hstack([mfcc, chroma, zcr])
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None