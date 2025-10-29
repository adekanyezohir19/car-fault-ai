# AI CAR-FAULT DETECTION APP
# Developed by Adekanye Abdulzohir 🇳🇬

import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import joblib
import tempfile

# ---- Load your trained model (.pkl in same folder) ----
model = joblib.load("car_fault_model.pkl")

# ---- Fault explanation dictionary ----
fault_tips = {
    "engine": "Possible piston or valve issue. Check oil and compression.",
    "brake": "Uneven friction detected. Inspect brake pads and fluid.",
    "gearbox": "Vibration pattern indicates transmission slip. Check clutch oil.",
    "normal": "Sound normal. No immediate fault detected.",
}

# ---- Page setup ----
st.set_page_config(page_title="AI Car-Fault Detector", layout="centered")
st.title("🚗 AI Car-Fault Detection")
st.write("Upload or record your car sound — AI will detect possible faults.")
st.markdown("**Created by Adekanye Abdulzohir**")

# ---- File uploader OR microphone recording ----
uploaded_file = st.file_uploader("Upload car sound (.wav)", type=["wav"])
recorded = st.audio_input("Or record directly below")

if uploaded_file or recorded:
    with st.spinner("Analyzing sound..."):
        # choose the input source
        if uploaded_file:
            audio_bytes = uploaded_file.read()
        else:
            audio_bytes = recorded.getvalue()

        # save temporary wav
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            file_path = tmp.name

        # extract MFCC features
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        feature = np.mean(mfcc.T, axis=0).reshape(1, -1)

        # prediction
        pred = model.predict(feature)[0]

        # result display
        st.success(f"🔍 Detected fault: **{pred.upper()}**")
        st.info(f"💡 {fault_tips.get(pred, 'No tip available for this fault.')}")

        # play back the audio
        st.audio(audio_bytes, format="audio/wav")

else:
    st.warning("Please upload or record a sound to start.")
