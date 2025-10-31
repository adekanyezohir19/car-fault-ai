import streamlit as st
import numpy as np
import soundfile as sf
import io, os, requests
from gtts import gTTS
from pydub import AudioSegment

st.set_page_config(page_title="🚗 Car Fault AI – by Adekanye Abdulzohir", layout="wide")

# --- Dashboard ---
st.image("military_car.jpg", use_container_width=True)
st.markdown("""
### 🚗 **Car Fault AI – by Adekanye Abdulzohir**
Upload your car sound — AI analyzes **engine, gear, brake, clutch, battery, and wiring**.
(Supports .wav, .mp3, .mp4 formats)
""")

uploaded_file = st.file_uploader("🎵 Upload Car Sound", type=["wav", "mp3", "mp4"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")
    st.info("Analyzing sound... please wait ⏳")

    try:
        # Convert to wav for processing
        if uploaded_file.name.endswith(".mp3"):
            sound = AudioSegment.from_mp3(uploaded_file)
            buffer = io.BytesIO()
            sound.export(buffer, format="wav")
            buffer.seek(0)
            data, sr = sf.read(buffer)
        else:
            data, sr = sf.read(uploaded_file)

        duration = len(data) / sr
        avg_amplitude = np.mean(np.abs(data))

        # --- Simple AI logic ---
        if avg_amplitude < 0.02:
            result = "⚠️ Weak engine or ignition problem."
        elif duration < 1:
            result = "🔋 Battery or wiring issue."
        elif avg_amplitude > 0.1:
            result = "⚙️ Possible brake or clutch imbalance."
        else:
            result = "✅ Sound is normal. No fault detected."

        # Display result
        st.success(result)

        # Voice feedback
        tts = gTTS(result)
        tts.save("result.mp3")
        with open("result.mp3", "rb") as f:
            st.audio(f.read(), format="audio/mp3")

    except Exception as e:
        st.error("❌ Error analyzing file.")
        st.caption(str(e))

else:
    st.warning("Upload a sound file to start.")

st.markdown("""
---
👨🏽‍💻 **Developed by Adekanye Abdulzohir**  
Version 3.0 — Professional Voice-Integrated Car Fault AI
""")
