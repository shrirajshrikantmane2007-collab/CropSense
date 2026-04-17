import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="CropSense",
    page_icon="🌾",
    layout="centered"
)

# ---------------------------------------------------
# LOAD MODEL FILES
# ---------------------------------------------------
model = joblib.load("crop_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
le_target = joblib.load("label_encoder.pkl")

# ---------------------------------------------------
# DATA
# ---------------------------------------------------
CROP_EMOJI = {
    "apple":"🍎", "banana":"🍌", "blackgram":"🌱", "chickpea":"🌱",
    "coconut":"🥥", "coffee":"☕", "cotton":"🌿", "grapes":"🍇",
    "jute":"🌾", "kidneybeans":"🌱", "lentil":"🌱", "maize":"🌽",
    "mango":"🥭", "mothbeans":"🌱", "mungbean":"🌱", "muskmelon":"🍈",
    "orange":"🍊", "papaya":"🟠", "pigeonpeas":"🌱",
    "pomegranate":"🍎", "rice":"🌾", "watermelon":"🍉"
}

CROP_DESC = {
    "apple":"Thrives in cool climates with well-drained soil.",
    "banana":"Loves tropical warmth, high humidity, rich soil.",
    "blackgram":"Hardy legume that enriches soil nitrogen.",
    "chickpea":"Ideal drought-resistant crop for dry climates.",
    "coconut":"Needs sandy soil and coastal humidity.",
    "coffee":"Best in mild temperatures with rainfall.",
    "cotton":"Warm climate crop with deep fertile soil.",
    "grapes":"Prefers dry summers and mild winters.",
    "jute":"Hot humid climate with heavy rainfall.",
    "kidneybeans":"Needs fertile and well-drained soil.",
    "lentil":"Cool season crop with drought tolerance.",
    "maize":"Versatile crop with moderate rain.",
    "mango":"Fruit tree loving heat and dry flowering season.",
    "mothbeans":"Very drought resistant legume.",
    "mungbean":"Fast-growing warm climate legume.",
    "muskmelon":"Needs sunny warm weather.",
    "orange":"Subtropical fruit crop.",
    "papaya":"Warm moist tropical fruit.",
    "pigeonpeas":"Semi-arid drought tolerant legume.",
    "pomegranate":"Heat and drought tolerant fruit tree.",
    "rice":"Needs flooded fields and humidity.",
    "watermelon":"Warm climate crop with sandy soil."
}

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f2418,#173622,#102d1b);
    color: white;
}

.main-title {
    text-align:center;
    font-size:58px;
    font-weight:800;
    color:#d7f8de;
    margin-bottom:0px;
}

.tagline {
    text-align:center;
    color:#9dd3a8;
    font-size:16px;
    margin-top:-10px;
    margin-bottom:30px;
}

.card {
    background: rgba(255,255,255,0.06);
    padding:25px;
    border-radius:18px;
    border:1px solid rgba(255,255,255,0.08);
    margin-bottom:25px;
}

.section-title {
    color:#f3d36b;
    font-size:20px;
    font-weight:700;
    margin-bottom:18px;
}

.result-box {
    background: rgba(255,255,255,0.07);
    padding:20px;
    border-radius:20px;
    text-align:center;
}

.big-emoji { font-size:70px; }

.crop-name {
    font-size:38px;
    font-weight:800;
    color:#dfffe4;
}

.desc {
    color:#9ed8aa;
    font-size:15px;
}

.bar-label {
    font-size:15px;
    margin-top:14px;
    margin-bottom:4px;
}

footer {
    text-align:center;
    color:#7fa788;
    margin-top:30px;
}

/* ---------------------------
   BLUE SLIDER HANDLE / TRACK
----------------------------*/
.stSlider [data-baseweb="slider"] div[role="slider"]{
    background-color:#3b82f6 !important;
    border-color:#3b82f6 !important;
}

.stSlider [data-baseweb="slider"] > div > div > div{
    background:#3b82f6 !important;
}

/* ---------------------------
   BLUE NUMBER VALUES ON SLIDER
----------------------------*/
.stSlider label,
.stSlider span,
.stSlider p,
.stSlider div[data-testid="stTickBarMin"],
.stSlider div[data-testid="stTickBarMax"] {
    color:#60a5fa !important;
}

/* ---------------------------
   BLUE PROGRESS BAR
----------------------------*/
.stProgress > div > div > div > div{
    background-color:#3b82f6 !important;
}

/* ---------------------------
   BLUE GLASS BUTTON
----------------------------*/
.stButton > button {
    width:100%;
    height:52px;
    border-radius:16px;
    border:1px solid rgba(255,255,255,0.18);
    background: linear-gradient(
        135deg,
        rgba(59,130,246,0.35),
        rgba(37,99,235,0.28)
    );
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    color:white;
    font-weight:700;
    font-size:16px;
    letter-spacing:0.4px;
    box-shadow:
        0 8px 24px rgba(37,99,235,0.28),
        inset 0 1px 0 rgba(255,255,255,0.22);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    border:1px solid rgba(255,255,255,0.28);
    background: linear-gradient(
        135deg,
        rgba(96,165,250,0.42),
        rgba(59,130,246,0.34)
    );
    box-shadow:
        0 12px 30px rgba(37,99,235,0.36),
        inset 0 1px 0 rgba(255,255,255,0.28);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown("<div class='main-title'>🌾 CropSense</div>", unsafe_allow_html=True)
st.markdown("<div class='tagline'>Crop Recommendation System · Powered by Random Forest</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# INPUTS
# ---------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Soil & Climate Parameters</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    N = st.slider("Nitrogen (N)", 0, 140, 70)
    P = st.slider("Phosphorus (P)", 5, 145, 53)
    K = st.slider("Potassium (K)", 5, 205, 48)
    temperature = st.slider("Temperature (°C)", 8.0, 44.0, 25.6)

with col2:
    humidity = st.slider("Humidity (%)", 14, 100, 71)
    ph = st.slider("Soil pH", 3.5, 9.9, 6.5)
    rainfall = st.slider("Rainfall (mm)", 20, 299, 103)

predict = st.button("🌱 Analyse & Recommend", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
if predict:

    input_df = pd.DataFrame(
        [[N,P,K,temperature,humidity,ph,rainfall]],
        columns=['N','P','K','temperature','humidity','ph','rainfall']
    )

    input_scaled = preprocessor.transform(input_df)
    probabilities = model.predict_proba(input_scaled)[0]
    top_indices = np.argsort(probabilities)[::-1][:3]

    results = []

    for idx in top_indices:
        crop = le_target.classes_[idx]
        confidence = round(float(probabilities[idx]) * 100, 1)

        results.append({
            "crop": crop,
            "confidence": confidence,
            "emoji": CROP_EMOJI.get(crop, "🌱"),
            "description": CROP_DESC.get(crop, "")
        })

    top = results[0]

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Recommendation</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='result-box'>
        <div class='big-emoji'>{top['emoji']}</div>
        <div class='crop-name'>{top['crop'].title()}</div>
        <div class='desc'>{top['description']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Top 3 Predictions · Confidence</div>", unsafe_allow_html=True)

    for r in results:
        st.markdown(
            f"<div class='bar-label'>{r['emoji']} {r['crop'].title()} — {r['confidence']}%</div>",
            unsafe_allow_html=True
        )
        st.progress(int(r["confidence"]))

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown(
    "<footer>CropSense · Built with Streamlit & scikit-learn · Random Forest Classifier</footer>",
    unsafe_allow_html=True
)
