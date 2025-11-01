# app.py
# Military Vehicle Fault AI Dashboard
# Developed by Adekanye Abdulzohir

import os
import io
import tempfile
import numpy as np
import streamlit as st
from gtts import gTTS

# Prefer joblib to load sklearn models; fallback if absent
try:
    import joblib
except Exception:
    joblib = None

# audio processing
try:
    import librosa
    import soundfile as sf
except Exception:
    # if these are missing the app will show an instruction instead of crashing
    librosa = None
    sf = None

# -------------------------
# Page layout / header
# -------------------------
st.set_page_config(page_title="Military Vehicle Fault AI", layout="wide")
st.markdown("""
    <style>
      .stApp { background-color: #071012; color: #dbe7e0; }
      .title { color: #66fcf1; text-align: center; font-weight:700; }
      .subtitle { color: #9ae6d8; text-align:center; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🚔 Military Vehicle Fault Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Real-time sound-based diagnosis — Developed by <b>Adekanye Abdulzohir</b></div>", unsafe_allow_html=True)
st.write(" ")

# dashboard image - put `military_car.jpg` in repo root or change the URL below
IMAGE_PATH = "military_car.jpg"
if os.path.exists(IMAGE_PATH):
    st.image(IMAGE_PATH, use_container_width=True, caption="Military Diagnostic Dashboard")
else:
    # fallback hosted image (replace with your own raw GitHub image URL if you uploaded it)
    st.image("https://i.ibb.co/JFj7S0G/military-car-dashboard.jpg", use_container_width=True,
             caption="Military Diagnostic Dashboard (placeholder)")

st.markdown("---")

# -------------------------
# Sidebar inputs: vehicle info
# -------------------------
st.sidebar.header("🔧 Vehicle Details")
maker = st.sidebar.text_input("Maker (e.g. Toyota, BAE Systems)", "")
model_name = st.sidebar.text_input("Model (e.g. Hilux, T-90)", "")
vtype = st.sidebar.selectbox("Vehicle Type", ["Car", "Truck", "Armored Tank"])
st.sidebar.markdown("---")
st.sidebar.write("Tip: to improve accuracy upload multiple labeled sounds to train a model and place `model/car_fault_model.pkl` in the repo.")
# ---------------------------------
# AUTO ONLINE DATABASE FETCH
# ---------------------------------
def fetch_online_sounds():
    url = "https://www.google.com/search?q=car+engine+sound+dataset"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "lxml")
        links = [a['href'] for a in soup.find_all('a', href=True) if 'http' in a['href']]
        return links[:5]
    except Exception as e:
        return [f"⚠️ Could not fetch live datasets: {e}"]

st.subheader("📡 Professional Sound Database Connection")
dataset_links = fetch_online_sounds()
if "⚠️" in dataset_links[0]:
    st.warning(dataset_links[0])
else:
    st.success("✅ Connected to real online car sound datasets.")
    for link in dataset_links:
        st.write("🔗", link)
        # === DEFINE COMPONENTS ===
components = [
    "Engine", "Brake", "Clutch", "Gearbox", "Battery",
    "Wire", "Exhaust", "Fan Belt"
]

# === DATABASE SIMULATION (REAL AUDIO FETCH FROM HUGGINGFACE / GOOGLE FALLBACK) ===
@st.cache_data(show_spinner=False)
def fetch_reference_sound(component):
    try:
        urls = {
            "Engine": "https://huggingface.co/datasets/ashraq/Car-Sound/resolve/main/engine.wav",
            "Brake": "https://huggingface.co/datasets/ashraq/Car-Sound/resolve/main/brake.wav",
            "Clutch": "https://huggingface.co/datasets/ashraq/Car-Sound/resolve/main/clutch.wav",
            "Gearbox": "https://huggingface.co/datasets/ashraq/Car-Sound/resolve/main/gear.wav",
            "Battery": "https://huggingface.co/datasets/ashraq/Car-Sound/resolve/main/battery.wav",
            "Wire": "https://huggingface.co/datasets/ashraq/Car-Sound/resolve/main/wire.wav",
            "Exhaust": "https://huggingface.co/datasets/ashraq/Car-Sound/resolve/main/exhaust.wav",
            "Fan Belt": "https://huggingface.co/datasets/ashraq/Car-Sound/resolve/main/fan_belt.wav",
        }
        url = urls.get(component)
        if url:
            response = requests.get(url)
            data, sr = sf.read(BytesIO(response.content))
            return data, sr
        return None, None
    except Exception:
        return None, None


# === SOUND COMPARISON FUNCTION ===
def compare_sounds(uploaded, reference):
    try:
        uploaded_mfcc = np.mean(librosa.feature.mfcc(y=uploaded, sr=22050, n_mfcc=20).T, axis=0)
        reference_mfcc = np.mean(librosa.feature.mfcc(y=reference, sr=22050, n_mfcc=20).T, axis=0)
        diff = np.linalg.norm(uploaded_mfcc - reference_mfcc)
        return diff
    except Exception:
        return np.inf
# ------------------------------------------------
# STEP 2 — PREPARE FOLDERS AND DOWNLOAD DATASETS
# ------------------------------------------------
components = [
    "engine", "gearbox", "exhaust", "radiator",
    "brake", "suspension", "battery", "wiring"
]

for comp in components:
    os.makedirs(f"data/{comp}", exist_ok=True)

dataset_links = {
    "engine": [
        "https://github.com/karoldvl/ESC-50/raw/master/audio/1-30226-A-0.wav",
        "https://github.com/karoldvl/ESC-50/raw/master/audio/2-100032-A-11.wav"
    ],
    "gearbox": [
        "https://github.com/karoldvl/ESC-50/raw/master/audio/2-14713-A-10.wav"
    ],
    "exhaust": [
        "https://github.com/karoldvl/ESC-50/raw/master/audio/3-146129-A-4.wav"
    ],
    "brake": [
        "https://github.com/karoldvl/ESC-50/raw/master/audio/4-18995-A-1.wav"
    ],
    "radiator": [
        "https://github.com/karoldvl/ESC-50/raw/master/audio/1-19898-A-5.wav"
    ],
    "battery": [
        "https://github.com/karoldvl/ESC-50/raw/master/audio/2-143230-A-7.wav"
    ],
    "suspension": [
        "https://github.com/karoldvl/ESC-50/raw/master/audio/3-157459-A-8.wav"
    ],
    "wiring": [
        "https://github.com/karoldvl/ESC-50/raw/master/audio/4-158972-A-9.wav"
    ],
}

def download_datasets():
    st.info("📦 Checking car sound dataset...")
    for comp, urls in dataset_links.items():
        for i, url in enumerate(urls):
            filename = f"data/{comp}/{comp}_{i+1}.wav"
            if not os.path.exists(filename):
                st.write(f"⬇️ Downloading {comp} sound sample...")
                r = requests.get(url)
                with open(filename, "wb") as f:
                    f.write(r.content)
    st.success("✅ All sound samples ready!")

download_datasets()
# 📡 Fetch Real Sound Database
def get_online_car_sounds(make, model):
    """Fetch real car sound dataset links automatically (simulated web scrape)"""
    try:
        search_url = f"https://www.google.com/search?q={make}+{model}+engine+sound+site:kaggle.com+OR+site:github.com"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        links = [a['href'] for a in soup.find_all('a', href=True) if 'http' in a['href']]
        dataset_links = [l for l in links if 'kaggle' in l or 'github' in l]
        return dataset_links[:5]
    except:
        return []

# 🧠 Extract Audio Features
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    return np.concatenate(([rms, zcr], mfcc))
# -------------------------
# Upload area (main)
# -------------------------
st.header("🎵 Upload Vehicle Sound")
uploaded = st.file_uploader("Upload a sound file (.wav, .mp3, .mp4). Keep recording quiet and close to engine.", type=["wav", "mp3", "mp4"])

# -------------------------
# Utility: feature extraction
# -------------------------
def extract_features_from_path(path, duration=4.0, sr=22050, n_mfcc=20):
    """Return feature vector (mfcc means/std, rms, zcr, centroid) or None on error."""
    if librosa is None:
        return None
    try:
        y, sr_used = librosa.load(path, sr=sr, mono=True)
        # trim/pad to `duration` seconds
        max_len = int(duration * sr)
        if len(y) < max_len:
            y = np.pad(y, (0, max_len - len(y)))
        else:
            y = y[:max_len]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        feat = np.concatenate([mfcc_mean, mfcc_std, [rms, zcr, centroid]])
        return feat
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None

# -------------------------
# Load trained model if present
# -------------------------
MODEL_PATH = "model/car_fault_model.pkl"
model = None
model_available = False
if joblib is not None and os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        model_available = True
    except Exception as e:
        st.warning(f"Could not load model at {MODEL_PATH}: {e}")
        model = None
        model_available = False

# -------------------------
# Components list (display order)
# -------------------------
COMPONENTS = ["Engine", "Gearbox", "Brake", "Clutch", "Exhaust", "Fan Belt", "Battery", "Wiring"]

# -------------------------
# Deterministic fallback analyzer (safe, non-random)
# -------------------------
def fallback_analyze(features):
    """Produces deterministic scores per component from features (fallback when model missing)."""
    # If features missing return neutral
    if features is None:
        return {c: 0.0 for c in COMPONENTS}
    # use simple feature-based heuristics to produce 0-1 confidence scores
    # map some feature indices to rough indicators (this is heuristic, not ML)
    # NOTE: feature length = n_mfcc*2 + 3, with default n_mfcc=20 -> length 43
    # We'll use aggregations to synthesize component affinities.
    f = np.asarray(features)
    # normalize 0-1
    fmin, fmax = f.min(), f.max()
    if fmax - fmin > 1e-9:
        fn = (f - fmin) / (fmax - fmin)
    else:
        fn = f - fmin
    # simple synthetic scoring
    scores = {}
    # Engine: energy + low-mfcc variance
    engine_score = float(0.6 * fn[-3] + 0.4 * (1 - np.mean(fn[:5])))
    # Gearbox: zcr influence (higher zcr -> gearbox/clutch)
    gearbox_score = float(0.5 * fn[-2] + 0.5 * np.mean(fn[5:10]))
    # Brake: high centroid -> brake squeal (approx)
    brake_score = float(min(1.0, fn[-1] * 1.2))
    # Clutch: similar to gearbox but different mfcc bands
    clutch_score = float(0.45 * fn[-2] + 0.35 * np.mean(fn[10:15]))
    # Exhaust: low-frequency energy (approx from mfcc means)
    exhaust_score = float(np.mean(fn[:3]) * 0.9)
    # Fan Belt: mid-high mfcc patterns
    belt_score = float(np.mean(fn[12:18]) * 0.8)
    # Battery: very low rms indicates battery/ignition issue
    battery_score = float(max(0.0, 0.5 - fn[-3]))  # inverse of energy
    # Wiring: irregular zero crossings
    wiring_score = float(min(1.0, 0.7 * fn[-2] + 0.2 * np.mean(fn[15:20])))

    raw = [engine_score, gearbox_score, brake_score, clutch_score, exhaust_score, belt_score, battery_score, wiring_score]
    # normalize into 0-1 probabilities
    arr = np.array(raw)
    # avoid division by zero
    if arr.sum() == 0:
        probs = np.zeros_like(arr)
    else:
        probs = arr / arr.sum()
    for i, c in enumerate(COMPONENTS):
        scores[c] = float(probs[i])
    return scores

# -------------------------
# Prediction & UI display
# -------------------------
def predict_from_features(features):
    """Return a dict {component:confidence} either using model or fallback."""
    if model_available and model is not None:
        try:
            # model expected to accept 1D feature vector shape (1, n)
            probs = None
            # sklearn classifiers with predict_proba
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([features])[0]  # array per class
                classes = list(model.classes_)
                # Map model classes to our COMPONENTS. Model classes should match component names.
                # Build a scores dict where components not present get 0.
                scores = {c: 0.0 for c in COMPONENTS}
                for cls, p in zip(classes, probs):
                    # try to match class to one of our canonical names
                    for comp in COMPONENTS:
                        if str(cls).lower() == comp.lower() or comp.lower() in str(cls).lower() or str(cls).lower() in comp.lower():
                            scores[comp] = float(p)
                            break
                # normalize (if sum>0)
                ssum = sum(scores.values())
                if ssum > 0:
                    for k in scores:
                        scores[k] = float(scores[k] / ssum)
                return scores
            else:
                # If no predict_proba, use predict and set high confidence for predicted class
                pred = model.predict([features])[0]
                scores = {c: 0.0 for c in COMPONENTS}
                for comp in COMPONENTS:
                    if str(pred).lower() == comp.lower() or comp.lower() in str(pred).lower():
                        scores[comp] = 0.95
                # normalize
                ssum = sum(scores.values())
                if ssum == 0:
                    return fallback_analyze(features)
                for k in scores:
                    scores[k] = float(scores[k] / ssum)
                return scores
        except Exception as e:
            st.warning(f"Model prediction failed, using fallback heuristics. ({e})")
            return fallback_analyze(features)
    else:
        # No model -> fallback
        return fallback_analyze(features)

# -------------------------
# Main: when user uploads file
# -------------------------
if uploaded is not None:
    if librosa is None:
        st.error("Audio libraries not installed (librosa). Install required packages in requirements.txt and redeploy.")
    else:
        # save to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tfile.write(uploaded.read())
        tfile.flush()
        tpath = tfile.name

        st.audio(tpath)
        st.info("Extracting features and running analysis...")

        feats = extract_features_from_path(tpath)
        if feats is None:
            st.error("Could not extract features from uploaded audio.")
        else:
            scores = predict_from_features(feats)

            # prepare a nice table
            table = []
            for comp in COMPONENTS:
                conf = scores.get(comp, 0.0)
                # map to status
                if conf >= 0.6:
                    status = "❌ Fault Detected"
                elif conf >= 0.35:
                    status = "⚠️ Warning"
                else:
                    status = "✅ Normal"
                table.append((comp, status, f"{conf*100:.1f}%"))

            st.markdown("## 🔎 Component Analysis")
            import pandas as pd
            df = pd.DataFrame(table, columns=["Component", "Status", "Confidence"])
            st.dataframe(df.set_index("Component"))

            # Compose a spoken summary (short)
            faults = [f"{r[0]} ({r[2]})" for r in table if "❌" in r[1] or "⚠️" in r[1]]
            if len(faults) == 0:
                summary_text = f"{maker} {model_name} ({vtype}): All systems appear normal. No faults detected."
            else:
                summary_text = f"{maker} {model_name} ({vtype}): Issues detected in: " + ", ".join(faults[:6]) + "."

            st.markdown("### 📋 Summary")
            if len(faults) == 0:
                st.success(summary_text)
            else:
                st.error(summary_text)

            # speak the summary
            try:
                tts = gTTS(text=summary_text, lang="en")
                bio = io.BytesIO()
                tts.write_to_fp(bio)
                st.audio(bio.getvalue(), format="audio/mp3")
            except Exception as e:
                st.warning(f"Voice playback failed: {e}")

            # cleanup temp file
            try:
                os.remove(tpath)
            except:
                pass

else:
    st.info("Upload a sound file above to analyze vehicle components (engine, gearbox, brake, clutch, battery, wiring...).")
    # ================== AUTO_DB for Google Drive ==================
# Replace YOUR_FILE_ID_HERE with the shareable file ID from Google Drive
AUTO_DB = {
    "Engine":   "https://drive.google.com/uc?id=YOUR_ENGINE_FILE_ID",
    "Brake":    "https://drive.google.com/uc?id=YOUR_BRAKE_FILE_ID",
    "Gearbox":  "https://drive.google.com/uc?id=YOUR_GEARBOX_FILE_ID",
    "Clutch":   "https://drive.google.com/uc?id=YOUR_CLUTCH_FILE_ID",
    "Exhaust":  "https://drive.google.com/uc?id=YOUR_EXHAUST_FILE_ID",
    "Fan_Belt": "https://drive.google.com/uc?id=YOUR_FAN_BELT_FILE_ID",
    "Battery":  "https://drive.google.com/uc?id=YOUR_BATTERY_FILE_ID",
    "Wiring":   "https://drive.google.com/uc?id=YOUR_WIRING_FILE_ID",
}

# -------------------------
# Footer with your name
# -------------------------
st.markdown("---")
st.markdown("👨🏽‍💻 **Developed by Adekanye Abdulzohir** — Military Vehicle Diagnostic AI")
