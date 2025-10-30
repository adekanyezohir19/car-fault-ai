import streamlit as st
import numpy as np
import librosa
import joblib
import soundfile as sf
from gtts import gTTS
import os
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("car_fault_model.pkl")

# Component descriptions
descriptions = {
    "Engine": """The engine is the heart of your car. It converts fuel into power that drives your vehicle forward.
Common problems include knocking sounds, oil leaks, overheating, or loss of power.
⚠️ Possible causes: low oil, worn pistons, or poor fuel quality.
🧰 Fix: Check oil level, radiator coolant, and spark plugs. If noise continues, visit a mechanic immediately.""",

    "Brake": """The braking system ensures your car can stop safely.
Squeaking or grinding noises usually mean your brake pads are worn or your rotors are damaged.
⚠️ Possible causes: thin brake pads, low brake fluid, or debris between pads.
🧰 Fix: Replace brake pads, refill brake fluid, and have the brake lines checked.""",

    "Gearbox": """The gearbox transfers power from the engine to the wheels.
If you hear grinding noises or experience hard shifting, it may indicate a gearbox issue.
⚠️ Possible causes: low transmission fluid, worn gears, or clutch misalignment.
🧰 Fix: Check and top up gearbox oil. Visit a professional if shifting remains rough.""",

    "Clutch": """The clutch connects and disconnects the engine and gearbox.
A burning smell or slipping during gear changes often signals clutch wear.
⚠️ Possible causes: worn clutch plate, weak pressure plate, or hydraulic leak.
🧰 Fix: Avoid holding the clutch pedal halfway. Replace clutch plate if slipping continues.""",

    "Exhaust": """The exhaust system removes gases from the engine and reduces noise.
Loud exhausts or smoke can mean leaks or blockages.
⚠️ Possible causes: damaged muffler, cracked exhaust pipe, or dirty catalytic converter.
🧰 Fix: Inspect for leaks under the car. Replace rusted parts to avoid engine strain.""",

    "Fan Belt": """The fan belt powers key parts like the alternator, cooling fan, and water pump.
If you hear squealing, the belt may be loose or cracked.
⚠️ Possible causes: worn belt, bad pulley, or misaligned tensioner.
🧰 Fix: Check belt tension and condition. Replace if worn or frayed.""",
}

# Streamlit UI
st.set_page_config(page_title="AI Car Fault Detector", page_icon="🚗", layout="centered")
st.title("🚗 AI Car Fault Detection System")
st.image("military_car.jpg", use_container_width=True)
st.write("Upload a car sound (.wav) to detect faults in engine, brakes, clutch, gearbox, or other parts.")

# Upload file
uploaded_file = st.file_uploader("🎵 Upload Car Sound File (.wav)", type=["wav"])

if uploaded_file is not None:
    data, sr = sf.read(uploaded_file)
    st.audio(uploaded_file, format="audio/wav")

    # Extract MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20).T, axis=0).reshape(1, -1)

    # Predict fault
    pred = model.predict(mfcc)[0]
    info = descriptions.get(pred, "No detailed information available for this component.")

    # Text feedback
    st.subheader(f"🧠 Detected Fault: {pred}")
    st.write(info)

    # Voice feedback using gTTS
    tts_text = f"The detected fault is {pred}. {info}"
    tts = gTTS(tts_text)
    tts.save("voice.mp3")
    audio_file = open("voice.mp3", "rb")
    st.audio(audio_file.read(), format="audio/mp3")

    # Visualize sound
    st.write("🎶 Sound Analysis (MFCC Feature Map)")
    plt.figure(figsize=(8, 4))
    plt.imshow(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20), cmap="plasma", aspect="auto")
    plt.title("MFCC Visualization")
    plt.xlabel("Time Frames")
    plt.ylabel("MFCC Coefficients")
    st.pyplot(plt)
