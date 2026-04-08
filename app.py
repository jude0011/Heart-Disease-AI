import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Setup the Page
st.set_page_config(page_title="Cardiac Risk Assistant", layout="wide")
st.title("🩺 Heart Disease Prediction Assistant")
st.write("Enter patient clinical data to assess cardiovascular risk.")

# 2. Load the Data & Train Model (Simplified for the prototype)
@st.cache_data
def load_and_train():
    df = pd.read_csv('heart.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns

model, feature_names = load_and_train()

# 3. Create Sidebar Inputs
st.sidebar.header("Patient Vitals")
def user_input_features():
    age = st.sidebar.slider("Age", 1, 100, 50)
    sex = st.sidebar.selectbox("Sex", (0, 1), format_func=lambda x: "Male" if x==1 else "Female")
    cp = st.sidebar.slider("Chest Pain Type (0-3)", 0, 3, 1)
    trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.sidebar.slider("Cholesterol", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", (0, 1))
    restecg = st.sidebar.slider("Resting ECG Results", 0, 2, 0)
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", (0, 1))
    oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
    slope = st.sidebar.slider("Slope of peak exercise ST segment", 0, 2, 1)
    ca = st.sidebar.slider("Number of major vessels (0-3)", 0, 3, 0)
    thal = st.sidebar.slider("Thal (1 = normal; 2 = fixed defect; 3 = reversable defect)", 1, 3, 2)
    
    data = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal}
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 4. Display Prediction
st.subheader("Analysis Result")
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

col1, col2 = st.columns(2)

with col1:
    if prediction[0] == 1:
        st.error("⚠️ Result: HIGH RISK")
    else:
        st.success("✅ Result: LOW RISK")

with col2:
    st.metric("Confidence Level", f"{prediction_proba[0][prediction[0]]*100:.2f}%")

st.write("---")
st.write("**Note:** This is a prototype for educational purposes and should not be used for medical diagnosis.")