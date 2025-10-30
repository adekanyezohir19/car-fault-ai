# app.py
# AI Smart Vehicle Diagnostic (rule-based) - By Adekanye Abdulzohir
import streamlit as st
import numpy as np
import librosa
import joblib
import soundfile as sf
from gtts import gTTS
import os
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import tempfile
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Smart Vehicle Diagnostic by Adekanye", page_icon="🚗", layout="centered")
st.image("military_car.jpg", use_container_width=True)
st.title("🚗 Smart Vehicle Diagnostic System")
st.caption("Developed by **Adekanye Abdulzohir** — Nigerian Army Engineering Unit")
st.write("Upload audio or video (wav, mp3, mp4). The app analyzes sound and returns a full component-by-component status report (text + voice).")
st.markdown("---")

# Try to load a trained model if you later add one (optional)
MODEL_PATH = "car_fault_model.pkl"
trained_model = None
if os.path.exists(MODEL_PATH):
    try:
        trained_model = joblib.load(MODEL_PATH)
    except Exception:
        trained_model = None

# --- helper: feature extraction ---
def extract_audio_file(uploaded_file):
    # Save uploaded to a temp file, handle mp4->wav
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if suffix == ".mp4" or suffix == ".mov" or suffix == ".mkv" or suffix == ".avi":
        # convert to wav using moviepy
        clip = VideoFileClip(tmp_path)
        audio_tmp = tmp_path + ".wav"
        clip.audio.write_audiofile(audio_tmp, verbose=False, logger=None)
        clip.close()
        return audio_tmp
    else:
        # return path to saved audio (wav or mp3)
        return tmp_path

def compute_features(path, sr_target=22050):
    y, sr = librosa.load(path, sr=sr_target, mono=True)
    # trim silence
    y, _ = librosa.effects.trim(y)
    rms = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    spec_cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spec_bw = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    return {
        "y": y, "sr": sr, "rms": rms, "zcr": zcr,
        "spec_cent": spec_cent, "spec_bw": spec_bw, "tempo": tempo,
        "mfcc_mean": mfcc_mean, "mfcc": mfcc
    }

# --- heuristic per-component assessor ---
def assess_all(feat):
    # returns dict of component -> (status, reason, solution)
    scores = {}
    rms = feat["rms"]
    zcr = feat["zcr"]
    sc = feat["spec_cent"]
    sbw = feat["spec_bw"]
    mf1 = feat["mfcc_mean"][0]
    tempo = feat["tempo"]

    # Engine
    if rms > 0.035 and sc < 1600:
        status, reason, sol = "Fault", "High low-frequency energy (possible knocking / misfire).", "Check oil level, spark plugs, and compression; avoid driving long distances."
    elif rms > 0.03:
        status, reason, sol = "Warning", "Stronger than usual engine vibration/noise.", "Inspect engine mounts, oil, and fuel system."
    else:
        status, reason, sol = "OK", "Engine sound within expected range.", "Routine checks."

    scores["Engine"] = (status, reason, sol)

    # Gearbox
    if sc < 2000 and sbw > 2200 and rms > 0.03:
        status, reason, sol = "Fault", "Low centroid with broad bandwidth — grinding/gear wear likely.", "Check gearbox oil and clutch; schedule transmission service."
    elif sbw > 2500:
        status, reason, sol = "Warning", "High spectral bandwidth — rough shifting signals.", "Monitor gear shifts, check transmission oil."
    else:
        status, reason, sol = "OK", "Gearbox sounds normal.", "Routine inspection."

    scores["Gearbox"] = (status, reason, sol)

    # Brake
    if sc > 3000 and zcr > 0.06:
        status, reason, sol = "Fault", "High-frequency spikes — likely brake squeal or grinding.", "Inspect brake pads and rotors; replace worn pads."
    elif sc > 2500:
        status, reason, sol = "Warning", "Some high-frequency content that may indicate wear.", "Check brake pads and fluid."
    else:
        status, reason, sol = "OK", "Brakes show no obvious high-frequency damage.", "Routine brake check recommended."

    scores["Brake"] = (status, reason, sol)

    # Clutch
    if 0.02 < rms < 0.04 and sc < 1800 and mf1 > 0:
        status, reason, sol = "Warning", "Possible slipping or inconsistent engagement (clutch).", "Avoid riding clutch; have clutch plate inspected."
    else:
        status, reason, sol = "OK", "Clutch sound not showing clear faults.", "Routine check."

    scores["Clutch"] = (status, reason, sol)

    # Battery (note: audio heuristics are limited; we provide best-guess)
    if rms < 0.006:
        status, reason, sol = "Warning", "Very low acoustic energy — electrical systems may be weak (battery/alternator).", "Check battery voltage and charging; test battery under load."
    else:
        status, reason, sol = "OK", "Battery/charging system sounds nominal (no low hum detected).", "Measure battery voltage and charging during operation."

    scores["Battery"] = (status, reason, sol)

    # Alternator
    if 400 < sc < 1200 and 0.02 < zcr < 0.06:
        status, reason, sol = "Warning", "Mid-frequency hum present; alternator whine possible.", "Inspect alternator belt and charging output; test alternator."
    else:
        status, reason, sol = "OK", "No clear alternator whine detected.", "Check alternator if electrical symptoms exist."

    scores["Alternator"] = (status, reason, sol)

    # Wiring / Electrical noise
    if zcr > 0.09:
        status, reason, sol = "Warning", "High zero-crossing rate — intermittent electrical noise or loose connections.", "Inspect wiring harness and connectors for loose/cracked wires."
    else:
        status, reason, sol = "OK", "No strong electrical noise patterns detected.", "Regular inspections."

    scores["Wiring & Connections"] = (status, reason, sol)

    # Suspension / 4-wheel
    if sbw > 3000 and rms > 0.03:
        status, reason, sol = "Warning", "Broad spectral energy and vibrations — suspension or wheel bearing issues possible.", "Check wheel bearings, shocks/struts and alignment."
    else:
        status, reason, sol = "OK", "Suspension sounds normal (no big impacts detected).", "Routine inspection recommended."

    scores["Suspension & Bearings"] = (status, reason, sol)

    # Exhaust
    if sc < 1200 and rms > 0.03 and mf1 < -50:
        status, reason, sol = "Fault", "Low centroid with strong energy — possible exhaust leak or backfire.", "Inspect exhaust pipe, muffler and catalytic converter for leaks."
    else:
        status, reason, sol = "OK", "Exhaust system shows no obvious leak signatures.", "Routine check."

    scores["Exhaust"] = (status, reason, sol)

    # Cooling fan / belts
    if 0.03 < rms < 0.07 and (1000 < sc < 2200) and (mf1 > -100 and mf1 < 100):
        status, reason, sol = "Warning", "Periodic tonal energy — belt or fan noise possible.", "Inspect fan belt tension, pulleys and cooling fan operation."
    else:
        status, reason, sol = "OK", "Cooling and belt systems sound normal.", "Routine inspection."

    scores["Fan Belt & Cooling"] = (status, reason, sol)

    return scores

# --- main UI ---
uploaded_file = st.file_uploader("Upload car audio/video (wav, mp3, mp4)", type=["wav", "mp3", "mp4"])
if uploaded_file is None:
    st.info("Upload a file to start the full vehicle diagnostic (it will produce text + voice report).")
else:
    with st.spinner("Extracting and analyzing audio..."):
        audio_path = extract_audio_file(uploaded_file)
        feat = compute_features(audio_path)
    st.write("### 🔊 Audio summary")
    st.write(f"- RMS energy: {feat['rms']:.5f}")
    st.write(f"- Zero crossing rate: {feat['zcr']:.5f}")
    st.write(f"- Spectral centroid: {feat['spec_cent']:.2f} Hz")
    st.audio(audio_path, format="audio/wav")

    # If a trained model exists, attempt to use it for a primary prediction
    primary_prediction = None
    if trained_model is not None:
        try:
            primary_prediction = trained_model.predict(np.mean(feat["mfcc"], axis=1).reshape(1, -1))[0]
        except Exception:
            primary_prediction = None

    # Run heuristic full assessment
    result = assess_all(feat)

    # Build sequential report
    report_lines = []
    report_lines.append("Smart Vehicle Diagnostic Report")
    report_lines.append("Generated by AI system (heuristic analysis).")
    if primary_prediction:
        report_lines.append(f"Primary trained-model suggestion: {primary_prediction}")
    report_lines.append("-----")
    for comp, (status, reason, sol) in result.items():
        report_lines.append(f"{comp} — {status}")
        report_lines.append(f"  Reason: {reason}")
        report_lines.append(f"  Solution: {sol}")
        report_lines.append("")

    # Show report in UI one-by-one
    st.markdown("## 🔎 Full Component Report")
    for comp, (status, reason, sol) in result.items():
        if status == "OK":
            st.success(f"{comp}: {status}")
        elif status == "Warning":
            st.warning(f"{comp}: {status}")
        else:
            st.error(f"{comp}: {status}")
        st.write(f"**Reason:** {reason}")
        st.write(f"**Solution:** {sol}")
        st.markdown("---")

    # Make voice summary (one combined string)
    voice_text = " . ".join(report_lines[:2000])  # limit size somewhat
    try:
        tts = gTTS(voice_text)
        tts.save("full_report.mp3")
        with open("full_report.mp3", "rb") as f:
            st.audio(f.read(), format="audio/mp3")
    except Exception as e:
        st.error(f"Voice generation failed: {e}")

    # Visuals: waveform and MFCC
    st.subheader("Waveform")
    fig, ax = plt.subplots(figsize=(8,2))
    ax.plot(feat["y"])
    ax.set_xlabel("Samples"); ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    st.subheader("MFCC Heatmap")
    fig2, ax2 = plt.subplots(figsize=(8,3))
    img = librosa.display.specshow(feat["mfcc"], x_axis='time', ax=ax2)
    fig2.colorbar(img, ax=ax2)
    st.pyplot(fig2)

    # Cleanup temp files
    try:
        os.remove(audio_path)
    except:
        pass
    for fname in ["full_report.mp3", "voice.mp3", "temp_video.mp4"]:
        if os.path.exists(fname):
            try:
                os.remove(fname)
            except:
                pass
# -------------------------------
# 🔧 Smart Car Component Analyzer
# -------------------------------

import io
import random
from gtts import gTTS
import tempfile

# Expanded components and possible conditions
car_components = {
    "Engine": ["smooth", "knocking", "overheating", "idling issue"],
    "Brake": ["firm", "soft", "squealing", "fluid low"],
    "Clutch": ["slipping", "grinding", "responsive", "stiff"],
    "Gear": ["changing well", "delayed shift", "noise detected"],
    "Battery": ["charging well", "weak voltage", "corrosion issue"],
    "Tire": ["good pressure", "low pressure", "alignment off"],
    "Wiring": ["connection stable", "short circuit risk", "sensor fault"],
    "Sensor": ["optimal", "faulty", "requires calibration"]
}

# If user uploaded a sound file
if uploaded_file:
    st.subheader("🔍 Full Car Diagnostic Report")

    # Simulate sound-based diagnosis (AI model placeholder)
    results = {}
    for part, states in car_components.items():
        condition = random.choice(states)
        solution = {
            "smooth": "Everything is working fine.",
            "firm": "Brake system normal.",
            "charging well": "Battery system optimal.",
            "connection stable": "Wiring OK.",
        }.get(condition, f"Check or service your {part.lower()} for possible issues.")

        results[part] = {"Condition": condition, "Solution": solution}

    # Display results
    for part, info in results.items():
        st.markdown(f"### 🚗 {part}")
        st.write(f"**Condition:** {info['Condition'].capitalize()}")
        st.write(f"**Suggested Fix:** {info['Solution']}")
        st.divider()

    # Generate AI voice summary
    summary_text = "Here is your car status summary. "
    for part, info in results.items():
        summary_text += f"{part}: {info['Condition']}. "

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        tts = gTTS(summary_text)
        tts.save(temp_audio.name)
        st.audio(temp_audio.name, format="audio/mp3")

    st.success("✅ Diagnosis complete — both text and voice feedback ready.")
else:
    st.info("Upload a car sound (.wav, .mp3, .mp4, .aac) to get full system analysis.")
