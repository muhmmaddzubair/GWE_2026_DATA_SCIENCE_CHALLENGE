

# =====================================================
# IMPORT LIBRARIES
# =====================================================

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "hypertension_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "hypertension_model.pkl"

print(DATA_PATH)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Hypertension Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv(DATA_PATH)

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load(MODEL_PATH)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🩺 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Halaman Utama",
        "EDA Dashboard",
        "Prediction",
        "About"
    ]
)

# =====================================================
# HOME
# =====================================================

if page == "Halaman Utama":

    st.title("🩺 Hypertension Risk Prediction")

    st.markdown("""
Welcome to our **End-to-End Machine Learning Project**
developed for **GWE 2026**.

This application predicts whether a patient has a risk
of hypertension using the **Random Forest Classification**
algorithm.
""")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Records",
            len(df)
        )

    with col2:
        st.metric(
            "Total Features",
            len(df.columns)-1
        )

    with col3:
        st.metric(
            "Target",
            "Has_Hypertension"
        )

    st.divider()

    st.subheader("Dataset Features")

    st.markdown("""
- Age
- Salt Intake
- Stress Score
- Blood Pressure History
- Sleep Duration
- BMI
- Medication
- Family History
- Exercise Level
- Smoking Status
""")

    st.divider()

    st.subheader("Project Workflow")

    st.markdown("""
1. Data Understanding

2. Data Cleaning

3. Exploratory Data Analysis (EDA)

4. Feature Encoding

5. Correlation Analysis

6. Random Forest Classification

7. Model Deployment using Streamlit
""")

# =====================================================
# EDA DASHBOARD
# =====================================================

elif page == "EDA Dashboard":

    st.title("📊 Exploratory Data Analysis")

    st.write(
        "This page displays several visualizations from the hypertension dataset."
    )

    st.divider()

    st.subheader("Target Distribution")

    fig, ax = plt.subplots(figsize=(6,4))

    sns.countplot(
        x="Has_Hypertension",
        data=df,
        ax=ax
    )

    st.pyplot(fig)

    st.divider()

    st.subheader("BMI Distribution")

    fig, ax = plt.subplots(figsize=(6,4))

    sns.boxplot(
        y=df["BMI"],
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Salt Intake Distribution")

    fig, ax = plt.subplots(figsize=(6,4))

    sns.boxplot(
        y=df["Salt_Intake"],
        ax=ax
    )

    st.pyplot(fig)

    st.divider()
    st.subheader("Correlation Heatmap")

    # Encode categorical columns for correlation
    df_encoded = df.copy()

    df_encoded["Has_Hypertension"] = df_encoded["Has_Hypertension"].map({"Yes":1,"No":0})
    df_encoded["Family_History"] = df_encoded["Family_History"].map({"Yes":1,"No":0})
    df_encoded["Exercise_Level"] = df_encoded["Exercise_Level"].map({"Low":0,"Moderate":1,"High":2})
    df_encoded["Smoking_Status"] = df_encoded["Smoking_Status"].map({"Non-Smoker":0,"Smoker":1})
    df_encoded["BP_History"] = df_encoded["BP_History"].map({"Normal":0,"Prehypertension":1,"Hypertension":2})

    numerical_cols = [
        "Age",
        "Salt_Intake",
        "Stress_Score",
        "BP_History",
        "Sleep_Duration",
        "BMI",
        "Family_History",
        "Exercise_Level",
        "Smoking_Status",
        "Has_Hypertension"
    ]

    corr = df_encoded[numerical_cols].corr()

    fig, ax = plt.subplots(figsize=(10,8))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=ax
    )

    st.pyplot(fig)

# =====================================================
# PREDICTION
# =====================================================

elif page == "Prediction":

    st.title("🤖 Hypertension Risk Prediction")

    st.write(
        "Enter the patient's information below."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=25
        )

        salt = st.number_input(
            "Salt Intake",
            min_value=0.0,
            max_value=20.0,
            value=5.0
        )

        stress = st.number_input(
            "Stress Score",
            min_value=0,
            max_value=100,
            value=50
        )

        bp = st.selectbox(
            "Blood Pressure History",
            [
                "Normal",
                "Prehypertension",
                "Hypertension"
            ]
        )

        sleep = st.number_input(
            "Sleep Duration",
            min_value=0.0,
            max_value=24.0,
            value=7.0
        )

    with col2:

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=22.0
        )

        family = st.selectbox(
            "Family History",
            [
                "No",
                "Yes"
            ]
        )

        exercise = st.selectbox(
            "Exercise Level",
            [
                "Low",
                "Moderate",
                "High"
            ]
        )

        smoking = st.selectbox(
            "Smoking Status",
            [
                "Non-Smoker",
                "Smoker"
            ]
        )

    st.divider()

    if st.button("Predict"):

# ==========================================
# Encoding (Sama seperti notebook)
# ==========================================

        bp_value = {
            "Normal": 0,
            "Prehypertension": 1,
            "Hypertension": 2
        }[bp]

        family_value = {
            "No": 0,
            "Yes": 1
        }[family]

        exercise_value = {
            "Low": 0,
            "Moderate": 1,
            "High": 2
        }[exercise]

        smoking_value = {
            "Non-Smoker": 0,
            "Smoker": 1
        }[smoking]

        # ==========================================
        # Create DataFrame
        # ==========================================

        input_data = pd.DataFrame({

            "Age":[age],
            "Salt_Intake":[salt],
            "Stress_Score":[stress],
            "BP_History":[bp_value],
            "Sleep_Duration":[sleep],
            "BMI":[bmi],
            "Family_History":[family_value],
            "Exercise_Level":[exercise_value],
            "Smoking_Status":[smoking_value]

        })

        # ==========================================
        # Prediction
        # ==========================================

        prediction = model.predict(input_data)

        probability = model.predict_proba(input_data)

        confidence = probability[0].max() * 100

        st.divider()

        st.subheader("Prediction Result")

        if prediction[0] == 1:

            st.error("⚠️ High Risk of Hypertension")

        else:

            st.success("✅ Low Risk of Hypertension")

        st.metric(
            label="Model Confidence",
            value=f"{confidence:.2f}%"
        )

        st.divider()

        st.subheader("Input Summary")

        summary = pd.DataFrame({

            "Feature":[
                "Age",
                "Salt Intake",
                "Stress Score",
                "Blood Pressure History",
                "Sleep Duration",
                "BMI",
                "Family History",
                "Exercise Level",
                "Smoking Status"
            ],

            "Value":[
                age,
                salt,
                stress,
                bp,
                sleep,
                bmi,
                family,
                exercise,
                smoking
            ]

        })

        st.dataframe(summary, use_container_width=True)

# =====================================================
# ABOUT
# =====================================================

elif page == "About":

    st.title("ℹ️ About This Project")

    st.markdown("""
### Hypertension Risk Prediction

This application was developed as an End-to-End Machine Learning project for **GWE 2026**.

The objective is to predict whether a patient has hypertension based on lifestyle and medical history.
""")

    st.divider()

    st.subheader("Machine Learning Algorithm")

    st.write("""
Random Forest Classifier
""")

    st.subheader("Dataset Features")

    st.write("""
- Age
- Salt Intake
- Stress Score
- Blood Pressure History
- Sleep Duration
- BMI
- Family History
- Exercise Level
- Smoking Status
""")

    st.divider()

    st.subheader("Project Workflow")

    st.write("""
✔ Data Understanding

✔ Data Cleaning

✔ Exploratory Data Analysis

✔ Feature Encoding

✔ Correlation Analysis

✔ Random Forest Classification

✔ Deployment using Streamlit
""")

    st.divider()

    st.subheader("Team")

    st.write("""
- Naufal Asa Malika (103012500151)

- Muhammad Zubair (102022500261)
""")

    st.success("Thank you for visiting our application!")
