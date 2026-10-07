import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="SugarSense — Diabetes Risk Screening",
    page_icon="🩺",
    layout="centered"
)

MODEL_PATH = (
    Path(__file__).parent
    / "Bonus Challenges"
    / "Bonus 3"
    / "sugarsense_model.joblib"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🩺 SugarSense")
st.subheader("Early Diabetes Risk Screening")

st.write(
    "Enter the patient's clinical measurements below. "
    "The trained SugarSense model will estimate diabetes risk."
)

st.info(
    "Educational screening tool only — this result is not a medical diagnosis. "
    "Please consult a qualified healthcare professional for clinical evaluation."
)

with st.form("patient_form"):
    st.markdown("### Patient Information")

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    glucose = st.number_input(
        "Glucose (mg/dL)",
        min_value=0.0,
        max_value=300.0,
        value=120.0,
        step=1.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure (mm Hg)",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )

    skin_thickness = st.number_input(
        "Skin Thickness (mm)",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0
    )

    insulin = st.number_input(
        "Insulin (µU/mL)",
        min_value=0.0,
        max_value=900.0,
        value=80.0,
        step=1.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=80.0,
        value=25.0,
        step=0.1
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.47,
        step=0.01
    )

    age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=120,
        value=33,
        step=1
    )

    submitted = st.form_submit_button(
        "🔍 Screen Patient",
        use_container_width=True
    )

if submitted:
    patient = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }])

    try:
        probability = float(model.predict_proba(patient)[0][1])

        if probability < 0.30:
            risk = "Low"
        elif probability < 0.60:
            risk = "Medium"
        else:
            risk = "High"

        st.markdown("### Screening Result")

        if risk == "Low":
            st.success(f"Risk Level: {risk}")
        elif risk == "Medium":
            st.warning(f"Risk Level: {risk}")
        else:
            st.error(f"Risk Level: {risk}")

        st.metric("Predicted Diabetes Probability", f"{probability:.1%}")

        st.caption(
            "Risk thresholds used by the SugarSense Mini App: "
            "<30% = Low, 30%–<60% = Medium, ≥60% = High."
        )

    except Exception as e:
        st.error("The model could not process the entered values.")
        st.exception(e)

st.markdown("---")
st.caption("SugarSense | Supervised Machine Learning Diabetes Risk Screening")
