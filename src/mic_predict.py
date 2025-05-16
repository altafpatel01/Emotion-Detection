import whisper
import joblib
import sounddevice as sd
import wavio
import numpy as np
import os

# Load models once
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")

print("Loading Emotion Detection model...")
emotion_model = joblib.load("model/emotion_model.pkl")


duration = 5  # seconds
fs = 16000    # sampling rate

def record_audio(filename="temp_audio.wav"):
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wavio.write(filename, recording, fs, sampwidth=2)
    print(f"Audio recorded and saved to {filename}")

def transcribe_audio(filename):
    print("Transcribing audio...")
    result = whisper_model.transcribe(filename)
    text = result["text"]
    print(f"Transcription: {text}")
    return text
    
def predict_emotion(text):
    prediction = emotion_model.predict([text])[0]
    print(f"Predicted Emotion: {prediction}")

if __name__ == "__main__":
    while True:
        inp = input("\nPress Enter to record or type 'exit' to quit: ")
        if inp.lower() == "exit":
            break
        audio_file = "temp_audio.wav"
        record_audio(audio_file)
        text = transcribe_audio(audio_file)
        predict_emotion(text)
        os.remove(audio_file)
