import streamlit as st
import numpy as np
import librosa
import joblib
import soundfile as sf

st.title("🚗 AI Car-Fault Detection from Sound")
st.write("Upload a car sound clip (.wav) and let AI predict which part may be faulty.")

uploaded = st.file_uploader("Upload .wav file", type=["wav"])

# Define car parts (for model prediction)
parts = ["Engine", "Gearbox", "Brake", "Exhaust", "Fan Belt"]

if uploaded is not None:
    # Load audio file
    st.audio(uploaded, format="audio/wav")

    try:
        # Read audio
        y, sr = librosa.load(uploaded, sr=22050)

        # Extract MFCC features
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0).reshape(1, -1)

        # Try to load the model (if uploaded)
        try:
            model = joblib.load("car_fault_model.pkl")
            pred = model.predict(mfccs)[0]
            st.success(f"🔍 Detected faulty part → {parts[pred]}")
        except:
            # Dummy prediction (random for now)
            st.warning("⚠️ No trained model found — running test simulation.")
            pred = np.random.choice(parts)
            st.info(f"Predicted (simulation): {pred}")

    except Exception as e:
        st.error(f"Error processing audio file: {e}")
