import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

# Function to load model safely
def load_model():
    model_path = "trained_rf_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as file:
            return pickle.load(file)
    else:
        return None

# Load the trained model
model_rf = load_model()

# Check if model loaded successfully
if model_rf is None:
    st.error("🚨 Model file `trained_rf_model.pkl` not found! Please upload it to the repository and redeploy.")
    st.stop()


# Streamlit App Title
st.title("🔧 AI-Powered Predictive Maintenance Dashboard")
st.markdown("Monitor machine health & predict failures in real-time!")

# Sidebar Inputs for Sensor Readings
st.sidebar.header("📊 Input Sensor Data")

def user_input_features():
    air_temp = st.sidebar.slider("Air Temperature (K)", 295.0, 320.0, 300.0)
    process_temp = st.sidebar.slider("Process Temperature (K)", 305.0, 340.0, 310.0)
    rotational_speed = st.sidebar.slider("Rotational Speed (rpm)", 1200, 3000, 1500)
    torque = st.sidebar.slider("Torque (Nm)", 3.0, 80.0, 40.0)
    tool_wear = st.sidebar.slider("Tool Wear (min)", 0, 250, 100)
    
    # Ensure input matches the trained model's feature names
    data = {
        "Air temperature [K]": [air_temp],
        "Process temperature [K]": [process_temp],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear],
        "Type_L": [0],  # Assume machine type is 'M' unless specified
        "Type_M": [0]
    }
    return pd.DataFrame(data)

# Get user input
df_input = user_input_features()

# Display input data
st.subheader("📌 Input Machine Data")
st.write(df_input)

# Predict Failure Probability
if st.button("🔍 Predict Machine Failure"):
    prediction = model_rf.predict(df_input)[0]
    failure_prob = model_rf.predict_proba(df_input)[0][1] * 100

    if prediction == 1:
        st.error(f"⚠️ High Risk: Machine Failure Likely! (Failure Probability: {failure_prob:.2f}%)")
    else:
        st.success(f"✅ Low Risk: Machine Operating Normally. (Failure Probability: {failure_prob:.2f}%)")

    # Show failure probability gauge
    st.subheader("📊 Failure Probability")
    st.progress(failure_prob / 100)
