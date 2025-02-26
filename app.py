import streamlit as st
import pandas as pd
import pickle

# Load the trained model pipeline
model_path = "mimo_rf_model.pkl"
with open(model_path, "rb") as file:
    rf_pipeline = pickle.load(file)

# Streamlit UI
st.set_page_config(page_title="AI-Powered Predictive Maintenance", layout="wide")

st.markdown("""
# 🔧 AI-Powered Predictive Maintenance Dashboard
Monitor machine health & predict failures in real-time!
""")

# Sidebar Inputs
st.sidebar.header("📊 Input Sensor Data")

air_temp = st.sidebar.slider("Air Temperature (K)", 295.0, 320.0, 307.85, step=0.1)
process_temp = st.sidebar.slider("Process Temperature (K)", 305.0, 340.0, 310.0, step=0.1)
rotational_speed = st.sidebar.slider("Rotational Speed (rpm)", 1200, 3000, 2381, step=1)
torque = st.sidebar.slider("Torque (Nm)", 3.0, 80.0, 40.0, step=0.1)
tool_wear = st.sidebar.slider("Tool Wear (min)", 0, 250, 100, step=1)

# Machine Type (Categorical)
machine_type = st.sidebar.selectbox("Machine Type", ["L", "M", "H"])

# Create DataFrame with proper columns
input_data = pd.DataFrame([[air_temp, process_temp, rotational_speed, torque, tool_wear, machine_type]],
                          columns=["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
                                   "Torque [Nm]", "Tool wear [min]", "Type"])

# Display Input Data
st.subheader("📌 Input Machine Data")
st.dataframe(input_data)

# --- Prediction Button ---
if st.button("🔍 Predict Machine Failure"):
    # Ensure input is processed through the full pipeline
    processed_input = rf_pipeline.named_steps["preprocessor"].transform(input_data)
    predictions = rf_pipeline.named_steps["classifier"].predict(processed_input)  
    failure_probs = rf_pipeline.named_steps["classifier"].predict_proba(processed_input)

    # Extract failure probability for "Machine failure"
    failure_prob = failure_probs[0][:, 1][0] * 100  

    # Determine Risk Level
    if failure_prob > 50:
        st.error(f"⚠️ **High Risk: Machine Failure Likely!** (Failure Probability: {failure_prob:.2f}%)")
    elif failure_prob > 20:
        st.warning(f"⚠️ **Moderate Risk: Maintenance Required Soon!** (Failure Probability: {failure_prob:.2f}%)")
    else:
        st.success(f"✅ **Low Risk: Machine Operating Normally.** (Failure Probability: {failure_prob:.2f}%)")
    
    # Display Failure Mode Predictions
    failure_labels = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    st.subheader("📊 Failure Mode Prediction")
    for i, mode in enumerate(failure_labels):
        if predictions[0][i] == 1:
            st.error(f"❌ **{mode} Failure Detected!**")
        else:
            st.success(f"✅ **No {mode} Failure.**")
