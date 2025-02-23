import streamlit as st
import pandas as pd
import pickle
import os
import sklearn

# Function to safely load the model
def load_model():
    model_path = "trained_rf_model.pkl"  # Ensure the filename matches your saved model
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as file:
                return pickle.load(file)
        except Exception as e:
            st.error(f"🚨 Model loading error: {e}")
            return None
    else:
        st.error("🚨 Model file not found! Upload `trained_rf_model.pkl` to the repository.")
        return None

# Load the trained model
model_rf = load_model()

# Stop execution if model fails to load
if model_rf is None:
    st.stop()

# Get the trained model's feature names
input_features = model_rf.feature_names_in_

# Streamlit App Title
st.title("🔧 AI-Powered Predictive Maintenance Dashboard")
st.markdown("Monitor machine health & predict failures in real-time!")

# Sidebar Inputs for Sensor Readings
st.sidebar.header("📊 Input Sensor Data")

def user_input_features():
    # Machine Type Selection (Encoding for Model)
    machine_type = st.sidebar.selectbox("Machine Type", ["L", "M", "H"])  # Assuming 'H' is another type
    type_encoded = {"Type_L": 0, "Type_M": 0, "Type_H": 0}
    type_encoded[f"Type_{machine_type}"] = 1  # Set the correct type to 1

    air_temp = st.sidebar.slider("Air Temperature (K)", 295.0, 320.0, 300.0)
    process_temp = st.sidebar.slider("Process Temperature (K)", 305.0, 340.0, 310.0)
    rotational_speed = st.sidebar.slider("Rotational Speed (rpm)", 1200, 3000, 1500)
    torque = st.sidebar.slider("Torque (Nm)", 3.0, 80.0, 40.0)
    tool_wear = st.sidebar.slider("Tool Wear (min)", 0, 250, 100)

    # Create DataFrame with correct feature order
    data = {
        "Type_L": [type_encoded["Type_L"]],
        "Type_M": [type_encoded["Type_M"]],
        "Type_H": [type_encoded["Type_H"]],
        "Air temperature [K]": [air_temp],
        "Process temperature [K]": [process_temp],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear],
        "TWF": [0],  # Defaulting to 0, change if needed
        "HDF": [0],
        "PWF": [0],
        "OSF": [0],
        "RNF": [0]
    }
    
    df = pd.DataFrame(data)
    
    # Ensure the columns match the trained model's feature set
    df = df[input_features]
    
    return df

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

