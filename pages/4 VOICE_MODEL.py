import streamlit as st
import os
import base64
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from utils.feature_extraction import extract_features
from utils.model_training import load_dataset, train_models
from utils.prediction import predict

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="AI Voice Insight System", layout="wide", page_icon="🎧")

# --- Combined Global and Custom CSS Styling ---
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
            font-size: 16px;
            background-color: #F5F5F5;
        }
        .stApp {
            background-color: #0a0f1c;
            color: #e0f7fa;
        }
        .stFileUploader, .stAudio, .stMarkdown {
            font-size: 18px;
        }
        .stButton, .stCheckbox {
            font-size: 14px;
        }
        .custom-box {
            font-size: 20px;
            padding: 1px;
            background-color: #212121;
            border-radius: 5px;
        }
        .card-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
            margin-top: 30px;
        }
        .card {
            background-color: #ffffff;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            padding: 20px;
            width: 300px;
            transition: transform 0.2s;
        }
        .card:hover {
            transform: scale(1.02);
        }
        .card-title {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .card-content {
            font-size: 16px;
            color: #666;
        }
        .icon {
            width: 24px;
            height: 24px;
        }
        @media screen and (max-width: 600px) {
            .card-container {
                flex-direction: column;
                align-items: center;
            }
        }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🎧 AI VOICE INSIGHT SYSTEM")
st.markdown("### Analyze Gender, Emotion, and More From Voice Using AI")


# --- Load dataset and train models ---
with st.spinner("Training models..."):
    if not os.path.exists("data"):
        st.error("Please add audio files in the 'data/' folder.")
        st.stop()
    X, y_gender, y_emotion = load_dataset("data")
    gender_model, emotion_model = train_models(X, y_gender, y_emotion)

# --- Upload Section ---
st.subheader("Upload Audio")
col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader("Drag and drop a WAV file", type=["wav"])

col_opt1, col_opt2 = st.columns(2)
noise_reduction = col_opt1.checkbox("Noise reduction")
skip_silence = col_opt2.checkbox("Skip silence")

if uploaded_file:
    file_path = "temp.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.audio(file_path)

    st.markdown("#### 📈 Audio Waveform")
    y, sr = librosa.load(file_path, sr=None)
    fig, ax = plt.subplots(figsize=(5, 0.5))  # Smaller waveform plot
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform", fontsize=8)
    ax.tick_params(labelsize=6)
    st.pyplot(fig)

    # --- Prediction ---
    with st.spinner("Analyzing audio..."):
        gender, emotion = predict(file_path, gender_model, emotion_model)
        duration = librosa.get_duration(y=y, sr=sr)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### AI Prediction")
            st.markdown(f"""
            <div class='custom-box'>
                👤 GENDER: {gender.capitalize()}<br>
                😊 EMOTION: {emotion.capitalize()}<br>
                🌐 LANGUAGE: English (default)<br>
                📊 Pitch Range: Medium<br>
                🧠 Consonance: Moderate<br>
                ⏱️ Duration: {round(duration, 2)} sec
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("#### Emotion & Pitch Chart")
            fig, ax = plt.subplots(figsize=(5,2))  # Smaller pitch chart
            pitch = librosa.yin(y, fmin=50, fmax=300, sr=sr)
            ax.plot(pitch, label='Pitch', color='magenta')
            ax.set_title("Pitch over Time", fontsize=14)
            ax.set_xlabel("Frames", fontsize=12)
            ax.set_ylabel("Frequency (Hz)", fontsize=6)
            ax.tick_params(labelsize=6)
            st.pyplot(fig)

            import speech_recognition as sr
            from pydub import AudioSegment

            # --- Transcript Section ---
            st.subheader("📝 Transcript")

            try:
                # Convert WAV to compatible format using pydub
                audio = AudioSegment.from_wav(file_path)
                audio.export("converted.wav", format="wav")

                recognizer = sr.Recognizer()
                with sr.AudioFile("converted.wav") as source:
                    audio_data = recognizer.record(source)

                    with st.spinner("Transcribing audio..."):
                        transcript = recognizer.recognize_google(audio_data)

                st.success("✅ Transcript generated:")
                st.text_area("Transcript", transcript, height=200)

            except sr.UnknownValueError:
                st.warning("Speech was unclear. Try another audio sample.")
            except sr.RequestError:
                st.error("Could not reach the speech recognition service.")
            except Exception as e:
                st.error(f"An error occurred during transcription: {e}")
