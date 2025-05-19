import streamlit as st
import numpy as np
import joblib  # Replaced pickle with joblib
from sklearn.preprocessing import StandardScaler

# Load the saved model and scaler using joblib
@st.cache_resource
def load_model():
    model = joblib.load("ckd_model.pkl")  # Assumes you saved with joblib.dump()
    return model

@st.cache_resource
def load_scaler():
    # Ideally, load the actual trained scaler from a file
    # For demo purposes, we'll simulate a scaler here
    scaler = StandardScaler()
    sample_data = np.array([
        [1.0, 0, 1.020, 14.0, 44, 5.0, 80, 120, 15, 1],
        [2.0, 1, 1.010, 12.5, 40, 4.8, 85, 140, 30, 0]
    ])
    scaler.fit(sample_data)
    return scaler

model = load_model()
scaler = load_scaler()

st.title("CKD Prediction Web App")

# Input widgets for the 10 attributes
sc = st.number_input("Serum Creatinine (sc)", min_value=0.1, max_value=20.0, value=1.0)
al = st.number_input("Albumin (al)", min_value=0, max_value=5, value=0)
sg = st.number_input("Specific Gravity (sg)", min_value=1.005, max_value=1.030, step=0.001, value=1.020)
hemo = st.number_input("Hemoglobin (hemo)", min_value=5.0, max_value=20.0, value=14.0)
pcv = st.number_input("Packed Cell Volume (pcv)", min_value=10, max_value=60, value=44)
rc = st.number_input("Red Blood Cell Count (rc)", min_value=2.0, max_value=7.0, value=5.0)
bp = st.number_input("Blood Pressure (bp)", min_value=40, max_value=200, value=80)
bgr = st.number_input("Blood Glucose Random (bgr)", min_value=50, max_value=400, value=120)
bu = st.number_input("Blood Urea (bu)", min_value=5, max_value=150, value=15)
rbc = st.selectbox("Red Blood Cells (urine) (rbc)", options=[0, 1], format_func=lambda x: "Abnormal" if x==0 else "Normal")

if st.button("Predict CKD"):
    input_data = np.array([[sc, al, sg, hemo, pcv, rc, bp, bgr, bu, rbc]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("CKD Detected")
    else:
        st.success("No CKD Detected")
