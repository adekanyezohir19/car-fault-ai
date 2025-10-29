import streamlit as st
import librosa, numpy as np, joblib, soundfile as sf

st.title("🔍 AI Car-Fault Detection from Sound")
st.write("Upload a car sound clip and let AI guess which part is faulty.")

uploaded = st.file_uploader("Upload .wav file", type=["wav"])
if uploaded:
    y, sr = librosa.load(uploaded, sr=22050)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0).reshape(1,-1)

    # For now we’ll use a dummy classifier until you upload your trained model
    try:
        model = joblib.load("car_fault_model.pkl")
        parts = ["Engine","Gearbox","Brake","Exhaust","Fan Belt"]
        pred = model.predict(mfcc)[0]
        st.success(f"Detected faulty part → {parts[pred]}")
    except:
        st.warning("⚠️ Upload your trained model file (car_fault_model.pkl) to the repo.")

    st.audio(uploaded, format="audio/wav")
