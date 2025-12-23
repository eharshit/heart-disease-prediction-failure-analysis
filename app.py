import streamlit as st
import pandas as pd
import joblib
import numpy as np
import base64
import matplotlib.pyplot as plt
import shap
from streamlit_shap import st_shap
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Minimal Dark Theme CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    .stApp {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
        color: #E0E0E0;
    }

    /* Force text color for readability */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stMarkdown p {
        color: #E0E0E0 !important;
    }
    
    /* Muted text for descriptions */
    .muted-text {
        color: #8B949E !important;
    }

    /* Cards/Containers */
    .minimal-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #010409;
        border-right: 1px solid #30363D;
    }
    
    /* Sidebar Font Sizes */
    [data-testid="stSidebar"] h2 {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] .muted-text {
        font-size: 1.1rem !important;
    }
    
    /* Sidebar Toggle Button (Collapse/Expand) */
    [data-testid="stSidebar"] button, [data-testid="collapsedControl"] {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] button svg, [data-testid="collapsedControl"] svg {
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }
    
    /* Widget Labels (Sidebar & Main) */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #FFFFFF !important;
        font-weight: 500;
    }

    /* Input Fields - Standardize Dark Look */
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input {
        background-color: #0D1117 !important;
        color: #E0E0E0 !important;
        border: 1px solid #30363D !important;
    }
    
    /* Selectbox is tricky, target the container */
    [data-baseweb="select"] > div {
        background-color: #0D1117 !important;
        color: #E0E0E0 !important;
        border: 1px solid #30363D !important;
    }
    
    /* Number Input Buttons (inc/dec) */
    button[kind="secondary"] {
        background-color: #161B22 !important;
        color: #E0E0E0 !important;
        border: 1px solid #30363D !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 16px;
    }
    [data-testid="stMetricLabel"] {
        color: #B0B8C1 !important;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }
    
    /* Slider Min/Max & Value adjustments */
    [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {
        color: #E0E0E0 !important;
        font-size: 0.9rem !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #238636;
        color: white !important;
        border: 1px solid rgba(240, 246, 252, 0.1);
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #2EA043;
        border-color: #8B949E;
    }
    
    /* Primary Button override */
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #1F6FEB;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background-color: #388BFD;
    }

    /* "View Dashboard" Button Specific Styling (Secondary buttons in header) */
    div[data-testid="stButton"] button[kind="secondary"] {
        font-weight: 700 !important; /* Bold */
        white-space: nowrap !important; /* Force single line */
        background-color: #21262d !important; /* Distinct dark grey */
        border: 1px solid #30363d !important;
        padding-left: 24px !important;
        padding-right: 24px !important;
    }
    div[data-testid="stButton"] button[kind="secondary"]:hover {
        background-color: #30363d !important;
        border-color: #8b949e !important;
    }
    
    /* Risk Indicators */
    .risk-high { border-left: 4px solid #F85149; }
    .risk-low { border-left: 4px solid #238636; }

    /* Hide Streamlit Header/Toolbar to fix "white bar" issue */
    header[data-testid="stHeader"] {
        background-color: #0E1117;
    }

</style>
""", unsafe_allow_html=True)

# --- Dashboard Dialog ---
@st.dialog("Power BI Dashboard", width="large")
def view_dashboard():
    # CSS to force a wider modal and minimize padding
    st.markdown("""
    <style>
        div[data-testid="stDialog"] div[role="dialog"] {
            width: 90vw !important;
            max-width: 1600px !important;
        }
        div[data-testid="stDialog"] div[role="dialog"] button[aria-label="Close"] {
            z-index: 999;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Base64 encode the image to embed it in HTML with constrained height
    with open("img/Heart Disease Analysis Dashboard.png", "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
        
    st.markdown(f"""
        <div style="display: flex; justify-content: center; align-items: center;">
            <img src="data:image/png;base64,{data}" 
                 style="max-width: 100%; max-height: 85vh; object-fit: contain; border-radius: 5px;">
        </div>
    """, unsafe_allow_html=True) 

# --- Header ---
# Adjusted column weights to ensure title stays on one line (logo + title)
col1, col2 = st.columns([1, 15])
with col1:
    st.image("img/11424074.png", width=85)
with col2:
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin-bottom: 8px;">Heart Disease Predictor</h1>
        <p class="muted-text">Advanced AI-Powered Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)

# --- Dashboard Button (Below Header) ---
# Separate row for the button to avoid crowding the title
b_col1, b_col2 = st.columns([9, 3])
with b_col2:
    if st.button("View Dashboard", use_container_width=True):
        view_dashboard()

# --- Model Loading ---
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load("models/rf_model.pkl")
        xgb_model = joblib.load("models/xgb_model.pkl")
        shap_background = joblib.load("models/shap_background.pkl")
        feature_names = joblib.load("models/feature_names.pkl")
        return rf_model, xgb_model, shap_background, feature_names, True
    except FileNotFoundError:
        return None, None, None, None, False

rf_model, xgb_model, shap_background, feature_names, artifacts_loaded = load_models()

if not artifacts_loaded:
    st.error("Model artifacts not found. Please run the training notebook.")
    st.stop()

# --- Sidebar ---
st.sidebar.header("Patient Data")
st.sidebar.markdown("<p class='muted-text' style='margin-bottom: 20px;'>Enter medical information below</p>", unsafe_allow_html=True)

# Model Selection
model_choice = st.sidebar.selectbox("Select Model", ("XGBoost", "Random Forest"), index=0)
st.sidebar.markdown("---")

def user_input_features():
    st.sidebar.subheader("Demographics")
    
    # Compact Layout: Age and Gender side-by-side
    c1, c2 = st.sidebar.columns(2)
    with c1:
        age = st.number_input('Age (years)', 20, 90, 50)
    with c2:
        sex = st.selectbox('Gender', ('Male', 'Female'))
    
    st.sidebar.subheader("Clinical Symptoms")
    
    # Simplified Chest Pain Labels
    cp_options = {
        'No Pain (Asymptomatic)': 'ASY',
        'Classic Heart Pain (Typical Angina)': 'TA',
        'Atypical Heart Pain (Atypical Angina)': 'ATA',
        'Non-Heart Pain (Non-Anginal)': 'NAP'
    }
    cp_label = st.sidebar.selectbox(
        'Chest Pain Type', 
        list(cp_options.keys())
    )
    cp_val = cp_options[cp_label]
    
    st.sidebar.subheader("Vitals")
    
    # Compact Layout: BP and Cholesterol
    c3, c4 = st.sidebar.columns(2)
    with c3:
        resting_bp = st.slider('Resting BP', 80, 200, 120, help="Resting Blood Pressure (mm Hg)")
    with c4:
        cholesterol = st.slider('Cholesterol', 0, 600, 200, help="Serum Cholesterol (mm/dl)")
    
    # Simplified Fasting Blood Sugar
    fbs_options = {
        'Normal (< 120 mg/dl)': 0,
        'High (> 120 mg/dl)': 1
    }
    fbs_label = st.sidebar.selectbox('Blood Sugar Level', list(fbs_options.keys()))
    fbs_val = fbs_options[fbs_label]
    
    st.sidebar.subheader("Cardiac Tests")
    
    # Compact Layout: ECG and Max HR
    c5, c6 = st.sidebar.columns(2)
    with c5:
        ecg_options = {
            'Normal': 'Normal',
            'Abnormal (ST)': 'ST',
            'Thickened (LVH)': 'LVH'
        }
        ecg_label = st.selectbox('Resting ECG', list(ecg_options.keys()))
        ecg_val = ecg_options[ecg_label]
    with c6:
        max_hr = st.slider('Max HR', 60, 220, 150, help="Maximum Heart Rate")
    
    # Simplified Exercise Angina
    exang_options = {
        'No': 'N',
        'Yes': 'Y'
    }
    exang_label = st.sidebar.selectbox('Chest Pain During Exercise?', list(exang_options.keys()))
    exang_val = exang_options[exang_label]
    
    st.sidebar.subheader("Advanced")
    
    # Compact Layout: Oldpeak and Slope
    c7, c8 = st.sidebar.columns(2)
    with c7:
        oldpeak = st.number_input('ST Depression', -2.0, 6.0, 0.0, step=0.1, help="ECG Feature: ST Depression")
    with c8:
        st_slope_options = {
            'Up (Normal)': 'Up',
            'Flat (Warn)': 'Flat',
            'Down (Bad)': 'Down'
        }
        st_slope_label = st.selectbox('ST Slope', list(st_slope_options.keys()))
        st_slope_val = st_slope_options[st_slope_label]

    # Mappings
    sex_val = "M" if sex == "Male" else "F"

    # Create dataframe
    data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex_val],
        "ChestPainType": [cp_val],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fbs_val],
        "RestingECG": [ecg_val],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exang_val],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope_val]
    })
    
    return data

input_df = user_input_features()

# --- Main Content ---

# Display Input Summary in a clean grid
st.markdown("### Patient Profile")
with st.container():
    st.markdown("<div class='minimal-card'>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Age", f"{input_df['Age'][0]}")
    c2.metric("BP", f"{input_df['RestingBP'][0]}")
    c3.metric("Cholesterol", f"{input_df['Cholesterol'][0]}")
    c4.metric("Max HR", f"{input_df['MaxHR'][0]}")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("### Risk Analysis")
predict_button = st.button('Analyze Heart Disease Risk', type="primary", use_container_width=True)

if predict_button:
    active_model = rf_model if "Random Forest" in model_choice else xgb_model
    
    with st.spinner('Analyzing...'):
        try:
            # Prediction
            prediction = active_model.predict(input_df)[0]
            probability = active_model.predict_proba(input_df)[0]
            
            prob_disease = probability[1] * 100
            prob_no_disease = probability[0] * 100
            
            # Result Card
            if prediction == 0:
                st.markdown(f"""
                <div class="minimal-card risk-low">
                    <h3 style="color: #3FB950; margin-bottom: 8px;">Low Risk Detected</h3>
                    <p class="muted-text">The model predicts a lower probability ({prob_no_disease:.1f}%) of heart disease.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="minimal-card risk-high">
                    <h3 style="color: #F85149; margin-bottom: 8px;">High Risk Detected</h3>
                    <p class="muted-text">The model predicts a higher probability ({prob_disease:.1f}%) of heart disease. Clinical consultation recommended.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed Metrics
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Probability of Disease", f"{prob_disease:.1f}%")
            with m2:
                st.metric("Model Confidence", f"{max(prob_disease, prob_no_disease):.1f}%")
            
            # SHAP Explanation
            st.markdown("---")
            st.markdown("### Feature Contribution (SHAP)")
            st.markdown("<p class='muted-text'>Explains which factors pushed the prediction towards High (Red) or Low (Blue) risk.</p>", unsafe_allow_html=True)
            
            try:
                # Check for pipeline steps
                if hasattr(active_model, "named_steps"):
                    preprocessor = active_model.named_steps.get("preprocessor")
                    classifier = active_model.named_steps.get("classifier")
                    
                    if preprocessor and classifier:
                        input_transformed = preprocessor.transform(input_df)
                        explainer = shap.TreeExplainer(classifier)
                        shap_values = explainer.shap_values(input_transformed)
                    else:
                        explainer = shap.TreeExplainer(active_model)
                        shap_values = explainer.shap_values(input_df)
                else:
                    explainer = shap.TreeExplainer(active_model)
                    shap_values = explainer.shap_values(input_df)

                # Handle SHAP return format
                if isinstance(shap_values, list):
                    shap_val = shap_values[1][0]
                    base_val = explainer.expected_value[1]
                else:
                    shap_val = shap_values[0]
                    base_val = explainer.expected_value

                # Render SHAP plot
                st_shap(shap.force_plot(base_val, shap_val, feature_names=feature_names, matplotlib=False), height=150)
                
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {str(e)}")

        except Exception as e:
            st.error(f"Analysis Error: {str(e)}")