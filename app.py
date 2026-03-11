import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Custom CSS for modern UI
st.markdown("""
<style>
/* App background */
.stApp {
    background-color: #cce7ff;  /* soft blue */
}

/* Input fields and number_input boxes */
input, .stNumberInput>div>div>input {
    background-color: #ffffff !important;   /* white fields */
    color: #000000 !important;              /* black text */
    border-radius: 10px !important;         /* rounded corners */
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1) !important; /* subtle shadow */
    padding: 8px !important;
    height: 40px !important;
}

/* Selectbox / dropdown */
.stSelectbox>div>div>div>div {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 10px !important;
    padding-left: 8px !important;
    font-size: 16px !important;
}

/* Buttons with modern style */
.stButton>button {
    background-color: #007acc;
    color: white;
    border-radius: 12px;
    padding: 0.6em 1.5em;
    font-weight: bold;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
    transition: background-color 0.3s ease;
}
.stButton>button:hover {
    background-color: #005f99;
}

/* Progress bar color */
.css-1q1n0ol.edgvbvh3 {
    background-color: #007acc !important;
}

/* Headings color */
h1, h2, h3 {
    color: #003366;
}

/* Add space between input fields */
.stNumberInput, .stSelectbox {
    margin-bottom: 15px !important;
}
</style>
""", unsafe_allow_html=True)

# App title
st.title("Heart Disease Prediction System")
st.write("Enter patient medical details to predict heart disease risk.")
st.subheader("Patient Information")

# Two columns for inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0)
    sex = st.selectbox("Sex", ["Select", "Male", "Female"], index=0)
    cp = st.number_input("Chest Pain Type", min_value=0)
    trestbps = st.number_input("Resting Blood Pressure", min_value=0)
    chol = st.number_input("Cholesterol", min_value=0)
    fbs = st.selectbox("Fasting Blood Sugar > 120 ?", ["Select", "Yes", "No"], index=0)
    restecg = st.number_input("Rest ECG", min_value=0)

with col2:
    thalach = st.number_input("Maximum Heart Rate", min_value=0)
    exang = st.selectbox("Exercise Induced Angina", ["Select", "Yes", "No"], index=0)
    oldpeak = st.number_input("Oldpeak", min_value=0.0, format="%.2f")
    slope = st.number_input("Slope", min_value=0)
    ca = st.number_input("Major Vessels", min_value=0)
    thal = st.number_input("Thal", min_value=0)

# Prediction logic
if st.button("Predict Heart Disease"):
    if age == 0 or sex == "Select" or cp == 0 or trestbps == 0 or chol == 0 or thalach == 0:
        st.warning("Please fill all required fields")
    else:
        sex_val = 1 if sex == "Male" else 0
        fbs_val = 1 if fbs == "Yes" else 0
        exang_val = 1 if exang == "Yes" else 0

        input_data = np.array([[age, sex_val, cp, trestbps, chol, fbs_val,
                                restecg, thalach, exang_val, oldpeak,
                                slope, ca, thal]])

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        risk = probability[0][1] * 100

        st.subheader("Prediction Result")
        st.write("Heart Disease Risk:", round(risk, 2), "%")
        st.progress(int(risk))

        if prediction[0] == 1:
            st.error("High Risk of Heart Disease")
            st.subheader("Health Tips")
            st.write("Exercise daily")
            st.write("Reduce oily food")
            st.write("Avoid smoking")
            st.write("Maintain healthy weight")
            st.write("Check blood pressure regularly")
        else:
            st.success("Low Risk of Heart Disease")
            st.subheader("Healthy Tips")
            st.write("Maintain balanced diet")
            st.write("Exercise regularly")
            st.write("Drink enough water")
            st.write("Regular health checkups")

st.write("---")
st.write("Machine Learning Mini Project using Python and Streamlit")