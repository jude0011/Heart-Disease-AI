import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from streamlit_shap import st_shap
import matplotlib.pyplot as plt

# 1. Page Configuration & Custom CSS
st.set_page_config(page_title="Cardiac Intelligence Portal", layout="wide", page_icon="❤️")

# Professional Styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🩺 Cardiac Risk Assessment Intelligence")
st.markdown("### *Bridging Clinical Data and Predictive Analytics*")
st.write("---")

# 2. Asset Loader
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('heart_model_calibrated.pkl')
        scaler = joblib.load('scaler.pkl')
        ood_detector = joblib.load('ood_detector.pkl')
        explainer = joblib.load('shap_explainer.pkl')
        return model, scaler, ood_detector, explainer
    except Exception as e:
        st.error(f"⚠️ Critical Error: Missing Model Files ({e})")
        return None, None, None, None

model, scaler, ood_detector, explainer = load_assets()

if model is None:
    st.stop()

# 3. Sidebar: Patient Clinical Vitals
st.sidebar.header("📋 Clinical Inputs")
st.sidebar.markdown("Provide precise patient metrics for assessment.")

def get_inputs():
    age = st.sidebar.slider("Patient Age", 20, 90, 50, help="Age is a primary non-modifiable risk factor.")
    sex = st.sidebar.selectbox("Biological Sex", (0, 1), format_func=lambda x: "Male" if x==1 else "Female")
    
    cp = st.sidebar.selectbox("Chest Pain Type (CP)", (0, 1, 2, 3), 
                             help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic")
    
    trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120, 
                                      help="Measured in the systemic arterial system.")
    
    chol = st.sidebar.number_input("Serum Cholestoral (mg/dl)", 100, 600, 200, 
                                   help="Total serum cholesterol levels.")
    
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", (0, 1), format_func=lambda x: "Yes" if x==1 else "No")
    
    restecg = st.sidebar.selectbox("Resting ECG Results", (0, 1, 2), 
                                  help="0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy")
    
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
    
    exang = st.sidebar.selectbox("Exercise Induced Angina", (0, 1), format_func=lambda x: "Yes" if x==1 else "No")
    
    oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, 
                               help="ST depression induced by exercise relative to rest.")
    
    slope = st.sidebar.selectbox("Peak Exercise ST Slope", (0, 1, 2), 
                                help="0: Upsloping, 1: Flat, 2: Downsloping")
    
    ca = st.sidebar.selectbox("Major Vessels Colored (0-3)", (0, 1, 2, 3))
    
    thal = st.sidebar.selectbox("Thalassemia Status", (1, 2, 3), 
                               format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x-1])

    data = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal}
    return pd.DataFrame(data, index=[0])

input_df = get_inputs()

# 4. Processing & Prediction
input_scaled = scaler.transform(input_df)
prob = model.predict_proba(input_scaled)[0][1]
is_ood = ood_detector.predict(input_scaled) == -1

# 5. Dashboard Layout
col_main, col_shap = st.columns([1, 1.2])

with col_main:
    st.subheader("Diagnostic Assessment")
    
    # Confidence Score & Risk Meter
    risk_level = "HIGH" if prob > 0.5 else "LOW"
    color = "inverse" if risk_level == "HIGH" else "normal"
    
    st.metric(label="Calculated Risk Probability", value=f"{prob*100:.1f}%", delta=risk_level, delta_color=color)
    
    if is_ood:
        st.warning("⚠️ **Data Divergence Detected:** This patient's vitals are statistically unusual. Interpret results with caution.")

    # Clinical Next Steps
    st.markdown("---")
    st.markdown("#### 🩺 Clinical Guidance")
    if prob > 0.7:
        st.error("**Urgent Evaluation Recommended:** High probability of significant coronary artery disease.")
    elif prob > 0.3:
        st.warning("**Intermediate Risk:** Consider further diagnostic testing (e.g., Stress Echo or CT Calcium Scoring).")
    else:
        st.success("**Low Risk:** Continue routine preventative screening and lifestyle management.")

with col_shap:
    st.subheader("Explainable AI (XAI)")
    st.write("This visualization explains the **why** behind the prediction by showing the contribution of each clinical feature.")
    
    # SHAP Visualization
    shap_values = explainer.shap_values(input_df)
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], input_df), height=250)
    
    st.write("🔴 Features in red **increase** risk. 🔵 Features in blue **decrease** risk.")

# 6. Actionable Output
st.write("---")
report_data = f"""
CARDIAC ASSESSMENT REPORT
-------------------------
Result: {risk_level} RISK
Probability: {prob*100:.2f}%
Patient Data: {input_df.to_dict(orient='records')[0]}

Note: This report is generated by an AI assistant and must be verified by a board-certified cardiologist.
"""
st.download_button("📥 Download Clinical Summary (PDF/TXT)", data=report_data, file_name="patient_cardiac_report.txt")

# 7. Ethics & Disclaimers
with st.expander("Model Card & Transparency"):
    st.write("""
    - **Dataset:** UCI Cleveland Heart Disease (303 instances).
    - **Model:** Random Forest with Sigmoid Calibration.
    - **Security:** Data is processed locally in memory and never stored on servers.
    - **XAI:** SHAP (SHapley Additive exPlanations) values represent feature attribution.
    """)
