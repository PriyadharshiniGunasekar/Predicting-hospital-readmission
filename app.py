import streamlit as st
import pandas as pd
import joblib

# ---------------------
# Streamlit Config
# ---------------------
st.set_page_config(page_title="Hospital Readmission Predictor", layout="centered")

st.title("🏥 Hospital Readmission Prediction (within 30 days)")
st.write("Fill in patient details to predict the risk of readmission within 30 days.")

# ---------------------
# Load Model + Scaler + Feature Names
# ---------------------
try:
    model = joblib.load("models/xgb_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    st.success("✅ Model, Scaler & Feature Names loaded successfully!")
except FileNotFoundError:
    st.error("❌ Missing model files. Ensure 'xgb_model.pkl', 'scaler.pkl', and 'feature_names.pkl' are in 'models/' folder.")
    st.stop()

# ---------------------
# User Input Form
# ---------------------
with st.form("patient_form"):
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    blood_pressure = st.text_input("Blood Pressure (format: systolic/diastolic)", "120/80")
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=50, max_value=400, value=180)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, format="%.1f")
    diabetes = st.selectbox("Diabetes", ["Yes", "No"])
    hypertension = st.selectbox("Hypertension", ["Yes", "No"])
    medication_count = st.number_input("Medication Count", min_value=0, max_value=50, value=2)
    length_of_stay = st.number_input("Length of Stay (days)", min_value=1, max_value=365, value=5)
    discharge_destination = st.selectbox(
        "Discharge Destination",
        ["Home", "Nursing_Facility", "Rehab", "Other"]
    )
    
    submit = st.form_submit_button("Predict Readmission")

# ---------------------
# Preprocessing
# ---------------------
if submit:
    # Parse BP
    try:
        systolic_bp, diastolic_bp = map(int, blood_pressure.split("/"))
    except:
        st.error("⚠️ Enter BP in format systolic/diastolic (e.g., 120/80).")
        st.stop()

    # Mappings (match train.py)
    gender_map = {"Male": 1, "Female": 0, "Other": 2}
    yes_no_map = {"Yes": 1, "No": 0}
    discharge_map = {"Home": 0, "Nursing_Facility": 1, "Rehab": 2, "Other": 3}

    data = {
        "age": age,
        "gender": gender_map[gender],
        "cholesterol": cholesterol,
        "bmi": bmi,
        "diabetes": yes_no_map[diabetes],
        "hypertension": yes_no_map[hypertension],
        "medication_count": medication_count,
        "length_of_stay": length_of_stay,
        "discharge_destination": discharge_map[discharge_destination],
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
    }

    # ---- Feature Engineering (same as train.py) ----
    data["pulse_pressure"] = data["systolic_bp"] - data["diastolic_bp"]
    data["high_bp_flag"] = 1 if data["systolic_bp"] > 140 or data["diastolic_bp"] > 90 else 0
    data["comorbidity_index"] = data["diabetes"] + data["hypertension"]

    if data["bmi"] < 18.5:
        data["bmi_category"] = 0
    elif data["bmi"] < 25:
        data["bmi_category"] = 1
    elif data["bmi"] < 30:
        data["bmi_category"] = 2
    else:
        data["bmi_category"] = 3

    data["discharge_risk"] = 1 if data["discharge_destination"] in [1, 2] else 0
    data["meds_per_day"] = data["medication_count"] / (data["length_of_stay"] + 1)
    data["meds_per_comorbidity"] = data["medication_count"] / (data["comorbidity_index"] + 1)

    # ---- Clinical-inspired Features ----
    data["age_risk_flag"] = 1 if data["age"] >= 60 else 0
    data["metabolic_syndrome"] = 1 if (
        data["diabetes"] == 1 and data["hypertension"] == 1 and data["bmi"] >= 30
    ) else 0

    # ---- Convert to DataFrame ----
    input_data = pd.DataFrame([data])

    # ---- Align with training ----
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

    # ---- Scale input ----
    input_scaled = scaler.transform(input_data)

    # ---- Prediction ----
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][1]

    # ---------------------
    # Clinical Override Rules
    # ---------------------
    override_flag = False
    override_reason = []

    if data["age"] >= 60 and data["comorbidity_index"] >= 2:
        override_flag = True
        override_reason.append("Age ≥ 60 with multiple comorbidities")

    if data["metabolic_syndrome"] == 1:
        override_flag = True
        override_reason.append("Metabolic syndrome present (diabetes + hypertension + BMI ≥ 30)")

    if data["length_of_stay"] > 7 and data["diabetes"] == 1:
        override_flag = True
        override_reason.append("Prolonged stay with diabetes")

    # ---------------------
    # Display Results
    # ---------------------
    if prediction == 1 or override_flag:
        st.error(f"🔴 High Risk: Patient likely to be readmitted "
                 f"(model probability {prediction_proba:.2f})")
        if override_flag:
            st.warning(f"⚠️ Clinical override applied due to: {', '.join(override_reason)}")
    else:
        st.success(f"🟢 Low Risk: Patient unlikely to be readmitted "
                   f"(probability {prediction_proba:.2f})")
