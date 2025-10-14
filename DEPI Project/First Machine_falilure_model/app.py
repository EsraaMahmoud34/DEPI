import streamlit as st
import pandas as pd
import joblib

# --- Load Trained Model ---
model = joblib.load("machine_failure_model.pkl")

# --- Page Config ---
st.set_page_config(page_title="AI-Powered Machine Failure Predictor", page_icon="⚙️", layout="wide")

# --- App Title ---
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>⚙️ AI-Powered Machine Failure Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict potential equipment failure using AI & IoT sensor data.</p>", unsafe_allow_html=True)
st.write("---")

# --- Sidebar for Info ---
with st.sidebar:
    st.header("ℹ️ About the App")
    st.write("""
    This app predicts **machine failure** based on sensor readings such as:
    - Air & process temperatures  
    - Rotational speed  
    - Torque  
    - Tool wear  
    - Failure indicators (TWF, HDF, etc.)
    
    Developed using **XGBoost** and **Streamlit**.
    """)
    st.markdown("---")
    st.caption("Developed by Esraa Mahmoud & Team 🧠")

# --- User Inputs ---
st.subheader("🧩 Machine & Process Details")

col1, col2 = st.columns(2)
with col1:
    product_id = st.text_input("🆔 Product ID", "")
    machine_id = st.text_input("🏭 Machine ID", "")
with col2:
    type_option = st.selectbox("⚙️ Type", ["L", "M", "H"])

st.write("---")
st.subheader("🌡️ Sensor Measurements")

col1, col2, col3 = st.columns(3)
with col1:
    air_temp = st.text_input("Air temperature [K]", "300")
with col2:
    process_temp = st.text_input("Process temperature [K]", "310")
with col3:
    rot_speed = st.text_input("Rotational speed [rpm]", "1500")

col4, col5 = st.columns(2)
with col4:
    torque = st.text_input("Torque [Nm]", "40")
with col5:
    tool_wear = st.text_input("Tool wear [min]", "120")

# --- Failure Types ---
st.write("---")
st.subheader("⚠️ Failure Type Indicators")
col1, col2, col3, col4, col5 = st.columns(5)
twf = col1.radio("TWF", ["No", "Yes"])
hdf = col2.radio("HDF", ["No", "Yes"])
pwf = col3.radio("PWF", ["No", "Yes"])
osf = col4.radio("OSF", ["No", "Yes"])
rnf = col5.radio("RNF", ["No", "Yes"])

# --- Encoding ---
encode = lambda x: 1 if x == "Yes" else 0
twf, hdf, pwf, osf, rnf = map(encode, [twf, hdf, pwf, osf, rnf])

type_map = {'L': 0, 'M': 1, 'H': 2}
type_encoded = type_map.get(type_option, 0)

# --- DataFrame for Model (now includes Type as a feature) ---
input_data = pd.DataFrame([[
    type_encoded,
    float(air_temp),
    float(process_temp),
    float(rot_speed),
    float(torque),
    float(tool_wear),
    twf,
    hdf,
    pwf,
    osf,
    rnf
]], columns=[
    'Type',
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
    'TWF',
    'HDF',
    'PWF',
    'OSF',
    'RNF'
])

# --- Predict Button ---
st.write("---")
if st.button("🚀 Predict Machine Failure", use_container_width=True):
    try:
        prediction = model.predict(input_data)[0]
        probability = float(model.predict_proba(input_data)[0][1] * 100)  # convert to Python float

        st.subheader("📊 Prediction Result")
        progress_value = float(min(probability / 100, 1.0))  # ensure it's native float
        
        if prediction == 1:
            st.error(f"⚠️ **Machine Failure Predicted!** \n\nFailure Probability: **{probability:.2f}%**")
            st.progress(progress_value)
        else:
            st.success(f"✅ **No Failure Expected.** \n\nFailure Probability: **{probability:.2f}%**")
            st.progress(progress_value)

    except Exception as e:
        st.error(f"❌ Error: {e}")

