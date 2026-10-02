import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(
    page_title="CardioPredict",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_model():
    df = pd.read_csv("heart_disease_uci.csv")
    df["target"] = (df["num"] > 0).astype(int)

    data = df.drop(columns=["id", "num", "dataset"]).copy()
    data["trestbps"] = data["trestbps"].replace(0, np.nan)
    data["chol"] = data["chol"].replace(0, np.nan)

    X = data.drop(columns=["target"])
    y = data["target"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numerical_features = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

    numerical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            min_samples_split=5,
            random_state=42,
        )),
    ])
    model.fit(X_train, y_train)
    return model


model = load_model()

# ---------- Theme tokens ----------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
dark = st.session_state.dark_mode

if dark:  # black + blue
    BG        = "#04070f"
    SURFACE   = "#0a1020"
    INPUT     = "#101a30"   # one fill for every field
    TEXT      = "#e8eefc"
    MUTED     = "#8592b3"
    BORDER    = "#1e2b4a"   # one border for every field and card
    HOVER     = "#2a3b66"
    PRIMARY   = "#3b82f6"
    PRIMARY_D = "#2563eb"
    ACCENT    = "#7db3ff"
    GLOW      = "rgba(59,130,246,0.18)"
    SHADOW    = "0 10px 30px -18px rgba(0,0,0,0.9)"
else:  # white + blue
    BG        = "#f3f7fe"
    SURFACE   = "#ffffff"
    INPUT     = "#ffffff"
    TEXT      = "#0b1b3a"
    MUTED     = "#5d6f94"
    BORDER    = "#d3ddf0"
    HOVER     = "#a9bce0"
    PRIMARY   = "#2563eb"
    PRIMARY_D = "#1d4ed8"
    ACCENT    = "#2563eb"
    GLOW      = "rgba(37,99,235,0.14)"
    SHADOW    = "0 10px 30px -20px rgba(37,99,235,0.35)"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
    }}
    .stApp {{
        background:
            radial-gradient(900px 480px at 8% -8%, {GLOW}, transparent 60%),
            radial-gradient(700px 420px at 100% 0%, {GLOW}, transparent 60%),
            {BG};
        color: {TEXT};
    }}
    .block-container {{ max-width: 1120px; padding-top: 2rem; padding-bottom: 4rem; }}
    header, footer, #MainMenu {{ visibility: hidden; }}

    h1, h2, h3, h4, h5, h6 {{ color: {TEXT} !important; letter-spacing: -0.02em; font-weight: 700; }}
    p, label, span, li {{ color: {TEXT}; }}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{
        color: {MUTED} !important; font-size: 0.86rem;
    }}
    [data-testid="stWidgetLabel"] p {{
        color: {TEXT} !important; font-weight: 600; font-size: 0.88rem;
    }}

    /* ---------- Cards ---------- */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 16px !important;
        box-shadow: {SHADOW};
    }}

    /* ==========================================================
       FORM CONTROLS: number inputs and dropdowns share ONE style
       ========================================================== */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] {{
        background: {INPUT} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
        min-height: 44px !important;
        box-shadow: none !important;
        transition: border-color .15s ease, box-shadow .15s ease;
    }}
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"]:hover {{
        border-color: {HOVER} !important;
    }}
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="select"] > div[aria-expanded="true"],
    div[data-baseweb="input"]:focus-within {{
        border-color: {PRIMARY} !important;
        box-shadow: 0 0 0 3px {GLOW} !important;
    }}

    /* inner wrappers/inputs: transparent so the outer box is the only surface */
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"],
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] input {{
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        outline: none !important;
    }}
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] div,
    div[data-baseweb="select"] span {{
        color: {TEXT} !important;
        -webkit-text-fill-color: {TEXT} !important;
        font-size: 0.95rem !important;
        font-weight: 500;
    }}
    div[data-baseweb="select"] svg {{ fill: {MUTED} !important; }}

    /* number-input +/- buttons match the field */
    [data-testid="stNumberInputStepUp"],
    [data-testid="stNumberInputStepDown"] {{
        background: transparent !important;
        color: {MUTED} !important;
        border: 0 !important;
        border-left: 1px solid {BORDER} !important;
        border-radius: 0 !important;
    }}
    [data-testid="stNumberInputStepUp"]:hover,
    [data-testid="stNumberInputStepDown"]:hover {{
        background: {GLOW} !important;
        color: {PRIMARY} !important;
    }}
    [data-testid="stNumberInputStepUp"] svg,
    [data-testid="stNumberInputStepDown"] svg {{ fill: currentColor !important; }}

    /* dropdown menu (rendered in a portal) */
    div[data-baseweb="popover"] > div,
    ul[role="listbox"] {{
        background: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
    }}
    ul[role="listbox"] li {{
        background: transparent !important;
        color: {TEXT} !important;
    }}
    ul[role="listbox"] li:hover,
    ul[role="listbox"] li[aria-selected="true"] {{
        background: {GLOW} !important;
        color: {ACCENT} !important;
    }}

    /* ---------- Buttons ---------- */
    .stButton > button {{
        border-radius: 12px !important;
        min-height: 3rem !important;
        font-weight: 600 !important;
        border: 1px solid {BORDER} !important;
        background: {SURFACE} !important;
        color: {TEXT} !important;
        transition: border-color .15s ease, background .15s ease;
    }}
    .stButton > button:hover {{
        border-color: {PRIMARY} !important;
        color: {PRIMARY} !important;
    }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {PRIMARY}, {PRIMARY_D}) !important;
        color: #ffffff !important;
        border: 0 !important;
        box-shadow: 0 10px 24px -12px {PRIMARY};
    }}
    .stButton > button[kind="primary"] p {{ color: #ffffff !important; }}
    .stButton > button[kind="primary"]:hover {{ filter: brightness(1.08); color: #ffffff !important; }}

    /* ---------- Metric + progress ---------- */
    [data-testid="stMetric"] {{
        background: {INPUT};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
    }}
    [data-testid="stMetricLabel"] * {{ color: {MUTED} !important; font-weight: 600 !important; }}
    [data-testid="stMetricValue"] * {{ color: {TEXT} !important; font-weight: 800 !important; }}
    [data-testid="stProgress"] > div > div {{ background: {BORDER} !important; border-radius: 999px; }}
    [data-testid="stProgress"] > div > div > div {{
        background: linear-gradient(90deg, {PRIMARY}, {ACCENT}) !important; border-radius: 999px;
    }}
    [data-testid="stAlert"] {{ border-radius: 12px !important; border: 1px solid {BORDER} !important; }}
    hr {{ border-color: {BORDER} !important; opacity: .7; }}

    /* ---------- Hero + sections ---------- */
    .hero-title {{
        font-size: 2.5rem; line-height: 1.12; font-weight: 800;
        letter-spacing: -0.03em; color: {TEXT}; margin: 0 0 .6rem 0;
    }}
    .hero-title em {{ font-style: normal; color: {PRIMARY}; }}
    .hero-sub {{ color: {MUTED} !important; font-size: 1rem; line-height: 1.65; max-width: 620px; margin: 0; }}
    .mobile-top-gap {{ display: none; }}
    .section-title {{
        display: flex; align-items: center; gap: 10px;
        font-size: 1.05rem; font-weight: 700; color: {TEXT}; margin: .2rem 0 .1rem 0;
    }}
    .section-title::before {{
        content: ""; width: 4px; height: 18px; border-radius: 2px; background: {PRIMARY};
    }}

    /* ---------- Performance cards ---------- */
    .perf-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
    .perf-card {{
        background: {SURFACE}; border: 1px solid {BORDER};
        border-radius: 14px; padding: 18px 20px;
    }}
    .perf-label {{ color: {MUTED}; font-size: .82rem; font-weight: 600; margin-bottom: 6px; }}
    .perf-value {{ color: {TEXT}; font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em; white-space: nowrap; }}

    @media (max-width: 768px) {{
        .hero-title {{ font-size: 2.05rem; }}
        .perf-grid {{ grid-template-columns: 1fr; }}
        .mobile-top-gap {{ display: block; height: 20px; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Top bar ----------
top_left, top_right = st.columns([7, 2], vertical_alignment="center")

with top_left:
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:40px;height:40px;border-radius:12px;
                background:linear-gradient(135deg,{PRIMARY},{PRIMARY_D});
                display:flex;align-items:center;justify-content:center;
                font-size:20px;box-shadow:0 8px 20px -8px {PRIMARY};">❤️</div>
            <div>
                <div style="font-weight:800;font-size:2.2rem;color:{TEXT};line-height:1.1;">CardioPredict</div>
                <div style="color:{MUTED};font-size:.8rem;">Cardiovascular risk estimation</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    st.markdown("<div class='mobile-top-gap'></div>", unsafe_allow_html=True)
    if st.button("☀️  Light mode" if dark else "🌙  Dark mode", use_container_width=True):
        st.session_state.dark_mode = not dark
        st.rerun()

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

# ---------- Hero ----------
with st.container(border=True):
    st.markdown(
        """
        <div style="padding:12px 8px 8px 8px;">
            <div class="hero-title">Heart disease prediction,<br/>powered by <em>machine learning</em>.</div>
            <p class="hero-sub">
                Enter a patient's clinical measurements and a trained Gradient Boosting
                model will estimate the probability of heart disease.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

# ---------- Form ----------
st.markdown('<div class="section-title">Patient information</div>', unsafe_allow_html=True)
st.caption("Enter the available clinical measurements below.")
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.markdown("#### Basic information")
        age = st.number_input("Age", 1, 100, 50, 1)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox(
            "Chest pain type",
            ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
        )
        trestbps = st.number_input("Resting blood pressure (mm Hg)", 50, 250, 130, 1)
        chol = st.number_input("Cholesterol (mg/dl)", 50, 700, 200, 1)
        fbs = st.selectbox(
            "Fasting blood sugar > 120 mg/dl",
            [False, True],
            format_func=lambda x: "Yes" if x else "No",
        )
        restecg = st.selectbox(
            "Resting ECG", ["normal", "lv hypertrophy", "st-t abnormality"]
        )

with right:
    with st.container(border=True):
        st.markdown("#### Cardiac measurements")
        thalch = st.number_input("Maximum heart rate achieved", 50, 250, 150, 1)
        exang = st.selectbox(
            "Exercise-induced angina",
            [False, True],
            format_func=lambda x: "Yes" if x else "No",
        )
        oldpeak = st.number_input("ST depression (Oldpeak)", -5.0, 10.0, 1.0, 0.1)
        slope = st.selectbox("ST segment slope", ["upsloping", "flat", "downsloping"])
        ca = st.number_input("Number of major vessels (0–3)", 0.0, 3.0, 0.0, 1.0)
        thal = st.selectbox(
            "Thalassemia", ["normal", "fixed defect", "reversable defect"]
        )

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ---------- Predict ----------
if st.button("🔍  Predict heart disease", type="primary", use_container_width=True):
    patient = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalch": thalch, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }])

    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0][1]

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="section-title">Prediction result</div>', unsafe_allow_html=True)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        if prediction == 1:
            st.error("⚠️  Heart disease detected by the model.")
        else:
            st.success("✅  No heart disease detected by the model.")

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        r1, r2 = st.columns([1, 2], vertical_alignment="center")
        with r1:
            st.metric("Estimated probability", f"{probability:.1%}")
        with r2:
            st.progress(float(probability), text=f"Model probability: {probability:.1%}")

    st.caption("⚠️ This is an educational machine-learning prediction, not a medical diagnosis.")

st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)

# ---------- Model performance ----------
st.markdown('<div class="section-title">Model performance</div>', unsafe_allow_html=True)
st.caption("Measured on the held-out test dataset.")
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="perf-grid">
        <div class="perf-card"><div class="perf-label">Model</div><div class="perf-value">Gradient Boosting</div></div>
        <div class="perf-card"><div class="perf-label">Test accuracy</div><div class="perf-value">82.61%</div></div>
        <div class="perf-card"><div class="perf-label">ROC-AUC</div><div class="perf-value">0.909</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown("#### About this model")
    st.write(
        "The final model uses a leakage-safe preprocessing pipeline with "
        "missing-value imputation, feature scaling, categorical encoding, "
        "and Gradient Boosting classification."
    )

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
st.divider()
st.caption("CardioPredict is an educational machine learning project, not a medical diagnostic system.")
