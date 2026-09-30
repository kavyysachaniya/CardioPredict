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

if dark:
    BG          = "#0a0e1a"
    SURFACE     = "#111827"
    SURFACE_2   = "#1a2332"
    INPUT       = "#0f1828"
    TEXT        = "#f1f5f9"
    MUTED       = "#94a3b8"
    BORDER      = "#1e293b"
    BORDER_SOFT = "#243247"
    PRIMARY     = "#3b82f6"
    PRIMARY_HOV = "#2563eb"
    ACCENT      = "#60a5fa"
    GLOW        = "rgba(59,130,246,0.15)"
else:
    BG          = "#f6f8fc"
    SURFACE     = "#ffffff"
    SURFACE_2   = "#f8fafc"
    INPUT       = "#ffffff"
    TEXT        = "#0f172a"
    MUTED       = "#64748b"
    BORDER      = "#e2e8f0"
    BORDER_SOFT = "#eef2f7"
    PRIMARY     = "#2563eb"
    PRIMARY_HOV = "#1d4ed8"
    ACCENT      = "#3b82f6"
    GLOW        = "rgba(37,99,235,0.10)"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
    }}

    .stApp {{
        background:
            radial-gradient(1200px 600px at 10% -10%, {GLOW}, transparent 60%),
            radial-gradient(900px 500px at 100% 0%, {GLOW}, transparent 55%),
            {BG};
    }}

    .block-container {{
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }}

    header, footer, #MainMenu {{ visibility: hidden; }}

    h1, h2, h3, h4, h5, h6 {{
        color: {TEXT} !important;
        letter-spacing: -0.02em;
        font-weight: 700;
    }}
    p, label, span {{ color: {TEXT}; }}

    [data-testid="stCaptionContainer"] {{
        color: {MUTED} !important;
        font-size: 0.86rem;
    }}

    /* ---------- Cards ---------- */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 18px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 8px 24px -16px rgba(0,0,0,0.15);
        transition: border-color 0.2s ease;
    }}
    [data-testid="stVerticalBlockBorderWrapper"]:hover {{
        border-color: {BORDER_SOFT} !important;
    }}

    /* ---------- Inputs ---------- */
    /* Keep selects and number inputs visually identical */
    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] > div {
        background: ${INPUT} !important;
        border: 1px solid ${BORDER} !important;
        border-radius: 10px !important;
        min-height: 40px !important;
        box-shadow: none !important;
    }

    div[data-baseweb="select"] > div {{
        background: {INPUT} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }}
    div[data-baseweb="select"] > div:hover {{
        border-color: {ACCENT} !important;
    }}
    div[data-baseweb="select"] > div:focus-within {{
        border-color: {PRIMARY} !important;
        box-shadow: 0 0 0 3px {GLOW} !important;
    }}
    div[data-baseweb="select"] span {{ color: {TEXT} !important; }}
    div[data-baseweb="select"] svg {{ fill: {MUTED} !important; }}

    div[data-testid="stNumberInput"] {{
        margin-top: 0 !important;
    }}
    div[data-testid="stNumberInput"] input {{
        background: {INPUT} !important;
        color: {TEXT} !important;
        border: 0 !important;
        border-radius: 10px 0 0 10px !important;
        height: 38px !important;
        box-shadow: none !important;
    }}
    div[data-testid="stNumberInput"] > div:focus-within {{
        border-color: {PRIMARY} !important;
        box-shadow: 0 0 0 3px {GLOW} !important;
    }}
    div[data-testid="stNumberInput"] button {{
        background: {INPUT} !important;
        color: {MUTED} !important;
        border: 0 !important;
        border-left: 1px solid {BORDER} !important;
        min-height: 38px !important;
    }}
    div[data-testid="stNumberInput"] button:hover {{
        background: {SURFACE_2} !important;
        color: {PRIMARY} !important;
    }}

    /* ---------- Buttons ---------- */
    .stButton > button {{
        border-radius: 12px !important;
        min-height: 3rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border: 1px solid {BORDER} !important;
        background: {SURFACE_2} !important;
        color: {TEXT} !important;
        transition: all 0.18s ease !important;
    }}
    .stButton > button:hover {{
        border-color: {ACCENT} !important;
        color: {PRIMARY} !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 18px -8px {GLOW};
    }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {PRIMARY}, {PRIMARY_HOV}) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 8px 24px -10px {PRIMARY};
    }}
    .stButton > button[kind="primary"]:hover {{
        background: linear-gradient(135deg, {PRIMARY_HOV}, {PRIMARY}) !important;
        box-shadow: 0 12px 28px -10px {PRIMARY};
        transform: translateY(-1px);
    }}

    /* ---------- Metrics ---------- */
    [data-testid="stMetric"] {{
        background: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 14px !important;
        padding: 1rem 1.1rem !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }}
    [data-testid="stMetricLabel"] {{
        color: {MUTED} !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    [data-testid="stMetricValue"] {{
        color: {TEXT} !important;
        font-weight: 700 !important;
    }}

    /* ---------- Custom performance cards ---------- */
    .perf-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
    }}
    .perf-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        transition: border-color 0.2s ease, transform 0.2s ease;
    }}
    .perf-card:hover {{
        border-color: {BORDER_SOFT};
        transform: translateY(-1px);
    }}
    .perf-label {{
        color: {MUTED};
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }}
    .perf-value {{
        color: {TEXT};
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.2;
        white-space: nowrap;
    }}

    /* ---------- Progress bar ---------- */
    [data-testid="stProgress"] > div > div {{
        background: linear-gradient(90deg, {PRIMARY}, {ACCENT}) !important;
        border-radius: 999px !important;
    }}
    [data-testid="stProgress"] > div {{
        background: {BORDER} !important;
        border-radius: 999px !important;
        height: 8px !important;
    }}

    /* ---------- Alerts ---------- */
    [data-testid="stAlert"] {{
        border-radius: 12px !important;
        border: 1px solid {BORDER} !important;
    }}

    hr {{ border-color: {BORDER} !important; opacity: 0.6; }}

    /* ---------- Hero ---------- */
    .hero-badge {{
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        background: {GLOW};
        color: {ACCENT};
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        border: 1px solid {BORDER};
        margin-bottom: 1rem;
    }}
    .hero-title {{
        font-size: 2.6rem !important;
        line-height: 1.1 !important;
        font-weight: 800 !important;
        margin: 0 0 0.6rem 0 !important;
        background: linear-gradient(135deg, {TEXT} 40%, {ACCENT});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    .hero-sub {{
        color: {MUTED} !important;
        font-size: 1.02rem;
        line-height: 1.6;
        max-width: 640px;
    }}

    .section-label {{
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: {MUTED};
        margin: 0.4rem 0 0.2rem 0;
    }}
    .section-label::before {{
        content: "";
        width: 22px;
        height: 2px;
        background: {PRIMARY};
        border-radius: 2px;
    }}

    .field-hint {{
        color: {MUTED};
        font-size: 0.78rem;
        margin-top: -0.5rem;
        margin-bottom: 0.6rem;
    }}

    @media (max-width: 768px) {{
        .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
        .hero-title {{ font-size: 2rem !important; }}
        .perf-grid {{ grid-template-columns: 1fr !important; }}
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
        <div style="display:flex;align-items:center;gap:12px;padding:4px 0;">
            <div style="
                width:40px;height:40px;border-radius:12px;
                background:linear-gradient(135deg,{PRIMARY},{ACCENT});
                display:flex;align-items:center;justify-content:center;
                font-size:20px;box-shadow:0 8px 20px -8px {PRIMARY};
            ">❤️</div>
            <div>
                <div style="font-weight:700;font-size:1.05rem;color:{TEXT};line-height:1.1;">CardioPredict</div>
                <div style="color:{MUTED};font-size:0.78rem;">Machine learning · Cardiovascular risk</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    if st.button(
        "☀️  Light mode" if dark else "🌙  Dark mode",
        use_container_width=True,
    ):
        st.session_state.dark_mode = not dark
        st.rerun()

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ---------- Hero ----------
with st.container(border=True):
    st.markdown(
        f"""
        <div style="padding:8px 4px 4px 4px;">
            <div class="hero-badge">Machine Learning · Cardiovascular Risk</div>
            <div class="hero-title">Heart disease prediction,<br/>powered by machine learning.</div>
            <p class="hero-sub">
                CardioPredict analyzes clinical parameters using a trained
                Gradient Boosting model to estimate the probability of heart disease.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

# ---------- Form header ----------
st.markdown('<div class="section-label">Patient information</div>', unsafe_allow_html=True)
st.caption("Enter the available clinical measurements below.")

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.markdown("#### Basic information")
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

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
            "Resting ECG",
            ["normal", "lv hypertrophy", "st-t abnormality"],
        )

with right:
    with st.container(border=True):
        st.markdown("#### Cardiac measurements")
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        thalch = st.number_input("Maximum heart rate achieved", 50, 250, 150, 1)
        exang = st.selectbox(
            "Exercise-induced angina",
            [False, True],
            format_func=lambda x: "Yes" if x else "No",
        )
        oldpeak = st.number_input(
            "ST depression (Oldpeak)", -5.0, 10.0, 1.0, 0.1
        )
        slope = st.selectbox(
            "ST segment slope",
            ["upsloping", "flat", "downsloping"],
        )
        ca = st.number_input("Number of major vessels (0–3)", 0.0, 3.0, 0.0, 1.0)
        thal = st.selectbox(
            "Thalassemia",
            ["normal", "fixed defect", "reversable defect"],
        )

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ---------- Predict ----------
if st.button("🔍  Predict heart disease", type="primary", use_container_width=True):
    patient = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }])

    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0][1]

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="section-label">Prediction result</div>', unsafe_allow_html=True)
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
            st.progress(
                float(probability),
                text=f"Model probability: {probability:.1%}",
            )

    st.caption(
        "⚠️ This is an educational machine-learning prediction, not a medical diagnosis."
    )

st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)

# ---------- Model performance ----------
st.markdown('<div class="section-label">Model performance</div>', unsafe_allow_html=True)
st.caption("Measured on the held-out test dataset.")

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="perf-grid">
        <div class="perf-card">
            <div class="perf-label">Model</div>
            <div class="perf-value">Gradient Boosting</div>
        </div>
        <div class="perf-card">
            <div class="perf-label">Test accuracy</div>
            <div class="perf-value">82.61%</div>
        </div>
        <div class="perf-card">
            <div class="perf-label">ROC-AUC</div>
            <div class="perf-value">0.909</div>
        </div>
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
st.caption(
    "CardioPredict · Educational machine learning project · "
    "Not a medical diagnostic system."
)