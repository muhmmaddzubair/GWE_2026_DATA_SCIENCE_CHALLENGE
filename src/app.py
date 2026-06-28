

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
## Deskripsi Proyek

Aplikasi ini merupakan implementasi **End-to-End Machine Learning**
yang dikembangkan untuk **GWE 2026 Data Science Challenge**.

Model Machine Learning digunakan untuk memprediksi apakah seseorang
memiliki risiko hipertensi berdasarkan kondisi kesehatan,
riwayat medis, dan gaya hidup.
""")

    st.divider()

    st.subheader("📌 Latar Belakang")

    st.write("""
Hipertensi merupakan salah satu penyakit tidak menular yang menjadi penyebab utama penyakit jantung, stroke, dan gagal ginjal.
Deteksi dini sangat penting agar pasien dapat melakukan tindakan pencegahan lebih awal.

Melalui pemanfaatan Machine Learning, proses prediksi risiko hipertensi dapat dilakukan dengan lebih cepat dan membantu proses pengambilan keputusan.
""")

    st.divider()

    st.subheader("🎯 Tujuan Proyek")

    st.write("""
- Menganalisis faktor-faktor yang memengaruhi hipertensi.
- Membangun model Machine Learning untuk memprediksi hipertensi.
- Menyediakan aplikasi berbasis Streamlit yang mudah digunakan.
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

    st.subheader("📋 Dataset Features")

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

    st.subheader("🧭 Navigasi Aplikasi")

    st.markdown("""
🏠 **Home**

Menampilkan informasi umum mengenai proyek.

📊 **EDA Dashboard**

Menampilkan visualisasi hasil Exploratory Data Analysis.

🔍 **Prediction**

Melakukan prediksi risiko hipertensi berdasarkan data yang dimasukkan pengguna.

ℹ️ **About**

Menampilkan informasi model, evaluasi, cara penggunaan, dan tim pengembang.
""")

    st.divider()

    st.subheader("⚙️ Project Workflow")

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

    st.write("""
    Halaman ini menampilkan hasil Exploratory Data Analysis (EDA) untuk
    memahami karakteristik dataset serta hubungan antar variabel yang
    digunakan dalam prediksi hipertensi.
    """)

    st.divider()

    # ======================================================
    # Target Distribution
    # ======================================================

    st.subheader("Target Distribution")

    fig, ax = plt.subplots(figsize=(6,4))

    sns.countplot(
        x="Has_Hypertension",
        data=df,
        ax=ax
    )

    ax.set_title("Target Distribution")

    st.pyplot(fig)

    st.info("""
📌 **Insight:**
Grafik menunjukkan distribusi jumlah pasien yang mengalami hipertensi dan tidak mengalami hipertensi.
Visualisasi ini membantu mengetahui keseimbangan data target sebelum proses pemodelan.
""")

    st.divider()

    # ======================================================
    # BMI Distribution
    # ======================================================

    st.subheader("BMI Distribution")

    fig, ax = plt.subplots(figsize=(6,4))

    sns.boxplot(
        y=df["BMI"],
        ax=ax
    )

    ax.set_title("BMI Distribution")

    st.pyplot(fig)

    st.info("""
📌 **Insight:**
Distribusi BMI digunakan untuk melihat penyebaran nilai BMI pada seluruh data
serta mendeteksi kemungkinan adanya nilai ekstrem (outlier).
""")

    st.divider()

    # ======================================================
    # Salt Intake Distribution
    # ======================================================

    st.subheader("Salt Intake Distribution")

    fig, ax = plt.subplots(figsize=(6,4))

    sns.boxplot(
        y=df["Salt_Intake"],
        ax=ax
    )

    ax.set_title("Salt Intake Distribution")

    st.pyplot(fig)

    st.info("""
📌 **Insight:**
Grafik menunjukkan distribusi konsumsi garam pada dataset.
Perbedaan konsumsi garam dapat menjadi salah satu faktor yang berkaitan dengan risiko hipertensi.
""")

    st.divider()

    # ======================================================
    # Age vs Hypertension
    # ======================================================

    st.subheader("Age vs Hypertension")

    fig, ax = plt.subplots(figsize=(6,4))

    sns.boxplot(
        x="Has_Hypertension",
        y="Age",
        data=df,
        ax=ax
    )

    ax.set_title("Age vs Hypertension")

    st.pyplot(fig)

    st.info("""
📌 **Insight:**
Kelompok pasien dengan hipertensi cenderung memiliki usia yang lebih tinggi dibandingkan kelompok yang tidak mengalami hipertensi.
""")

    st.divider()

    # ======================================================
    # BMI vs Hypertension
    # ======================================================

    st.subheader("BMI vs Hypertension")

    fig, ax = plt.subplots(figsize=(6,4))

    sns.boxplot(
        x="Has_Hypertension",
        y="BMI",
        data=df,
        ax=ax
    )

    ax.set_title("BMI vs Hypertension")

    st.pyplot(fig)

    st.info("""
📌 **Insight:**
Pasien dengan hipertensi memiliki kecenderungan nilai BMI yang lebih tinggi,
menunjukkan adanya hubungan antara obesitas dan hipertensi.
""")

    st.divider()

    # ======================================================
    # Family History vs Hypertension
    # ======================================================

    st.subheader("Family History vs Hypertension")

    fig, ax = plt.subplots(figsize=(6,4))

    sns.countplot(
        x="Family_History",
        hue="Has_Hypertension",
        data=df,
        ax=ax
    )

    ax.set_title("Family History vs Hypertension")

    st.pyplot(fig)

    st.info("""
📌 **Insight:**
Riwayat hipertensi dalam keluarga menunjukkan hubungan dengan risiko hipertensi,
di mana individu yang memiliki riwayat keluarga cenderung lebih berisiko.
""")

    st.divider()

    # ======================================================
    # Smoking Status vs Hypertension
    # ======================================================

    st.subheader("Smoking Status vs Hypertension")

    fig, ax = plt.subplots(figsize=(6,4))

    sns.countplot(
        x="Smoking_Status",
        hue="Has_Hypertension",
        data=df,
        ax=ax
    )

    ax.set_title("Smoking Status vs Hypertension")

    st.pyplot(fig)

    st.info("""
📌 **Insight:**
Grafik membandingkan status merokok dengan kondisi hipertensi.
Kebiasaan merokok dapat menjadi salah satu faktor yang memengaruhi risiko hipertensi.
""")

    st.divider()

    # ======================================================
    # Correlation Heatmap
    # ======================================================

    st.subheader("Correlation Heatmap")

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

    ax.set_title("Correlation Heatmap")

    st.pyplot(fig)

    st.info("""
📌 **Insight:**
Heatmap digunakan untuk melihat hubungan antar variabel.
Semakin mendekati nilai 1 atau -1 menunjukkan hubungan yang semakin kuat,
sedangkan nilai yang mendekati 0 menunjukkan hubungan yang lemah.
""")

    st.divider()

    st.subheader("📌 Kesimpulan EDA")

    st.success("""
Berdasarkan hasil Exploratory Data Analysis (EDA), variabel seperti usia, BMI,
riwayat keluarga, dan status merokok menunjukkan hubungan terhadap risiko hipertensi.
Hasil analisis ini menjadi dasar dalam pembangunan model Machine Learning untuk memprediksi hipertensi.
""")

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

Aplikasi ini dikembangkan sebagai proyek **End-to-End Machine Learning**
untuk **GWE 2026 Data Science Challenge**.

Tujuan aplikasi adalah memprediksi risiko hipertensi berdasarkan
karakteristik kesehatan dan gaya hidup pasien.
""")

    st.divider()

    st.subheader("🤖 Machine Learning Model")

    st.write("""
Model yang digunakan adalah **Random Forest Classifier**.

Random Forest dipilih karena memiliki performa yang baik,
mampu menangani data numerik maupun kategorikal,
serta cukup stabil dalam proses klasifikasi.
""")

    st.divider()

    st.subheader("📈 Evaluation Metrics")

    st.write("""
Accuracy : **96.73%**

Precision : **0.97**

Recall : **0.97**

F1-Score : **0.97**
""")

    st.divider()

    st.subheader("📂 Dataset Features")

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

    st.subheader("⚙️ Project Workflow")

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

    st.subheader("📝 Cara Menggunakan Aplikasi")

    st.write("""
1. Pilih menu **Prediction**.
2. Masukkan seluruh data pasien.
3. Klik tombol **Predict**.
4. Sistem akan menampilkan hasil prediksi hipertensi beserta probabilitasnya.
""")

    st.divider()

    st.subheader("👥 Team")

    st.write("""
**Naufal Asa Malika**  (103012500151)

**Muhammad Zubair**  (102022500261)
""")

    st.success("Terima kasih telah menggunakan aplikasi kami!")
