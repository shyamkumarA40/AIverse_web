import whisper

# Load model once
model = whisper.load_model("base")  # You can also try "small", "medium", or "large"

def transcribe_audio(file_path):
    try:
        result = model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        return f"Transcription error: {str(e)}"
