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
st.set_page_config(page_title="AI Voice Insight System", page_icon="🎧")
st.markdown("""
<style>
/* Import cyberpunk font */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap');

/* === MAIN PAGE LIGHT THEME === */
html, body, .stApp {
    background: linear-gradient(to right, #fddb92, #d1fdff);
    background-attachment: fixed;
    background-size: cover;
    color: #2c3e50 !important;
    font-family: 'Segoe UI', sans-serif !important;
}

/* Highlighted Header */
.highlight {
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white;
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 2rem;
}
.highlight h1 {
    font-size: 2.6rem;
    margin: 0;
}

/* Data Info Card */
.data-box {
    background-color: #ffffff;
    padding: 1.5rem;
    border: 2px dashed #1abc9c;
    border-radius: 12px;
    text-align: center;
    color: #2c3e50;
    margin-top: 1rem;
}
/* === CYBERPUNK BORDER FRAME === */
.glow-frame {
    position: relative;
    padding: 2rem;
    border-radius: 12px;
    background: rgba(0,0,0,0.7);
    overflow: hidden;
    z-index: 1;
    margin-top: 2rem;
}

.glow-frame::before {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    background: linear-gradient(45deg, #f500ff, #00f7ff, #f500ff, #00f7ff);
    z-index: -1;
    background-size: 400% 400%;
    animation: glitchBorder 6s linear infinite;
    border-radius: 14px;
}

@keyframes glitchBorder {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}


/* === SIDEBAR CYBERPUNK OVERRIDES === */
section[data-testid="stSidebar"] {
    background-color: #0a0a12 !important;
    border-right: 2px solid #222;
    box-shadow: 4px 0 25px rgba(245, 0, 255, 0.5);
    padding: 2rem 1rem 2rem 1rem;
    font-family: 'Orbitron', sans-serif !important;
}

/* Apply cyberpunk color and fix font size */
section[data-testid="stSidebar"] * {
    font-family: 'Orbitron', sans-serif !important;
    color: #f500ff !important;
    font-size: 16px !important;
    text-shadow: 0 0 3px #f500ff;
}

/* Hyperlink styles in sidebar */
section[data-testid="stSidebar"] a {
    color: #00f7ff !important;
    text-shadow: 0 0 4px #00f7ff;
}
section[data-testid="stSidebar"] a:hover {
    color: #f500ff !important;
    text-shadow: 0 0 8px #f500ff;
}
</style>
""", unsafe_allow_html=True)


# Header Banner
st.markdown("""
<div class="highlight">
    <h1>🎙️ Voice Emotion & Gender Analyzer</h1>
    <p>Upload voice clips and unlock emotional intelligence through sound.</p>
</div>
""", unsafe_allow_html=True)


# Main Content Section
st.markdown('<div class="section">', unsafe_allow_html=True)

# Upload Prompt Placeholder
st.markdown("""
<div class="voice-box">
    <p><strong>🎵 Ready to try it out?</strong></p>
    <p>Upload your <code>.wav</code> audio file in the functional Voice Model section of this app.</p>
</div>
""", unsafe_allow_html=True)


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
