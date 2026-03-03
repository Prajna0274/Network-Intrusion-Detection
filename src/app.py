import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Cyber Security Dashboard",
    layout="wide"
)

# =========================
# CUSTOM CYBER CSS
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #0B1F2A;
    color: #E0F7FA;
}

h1, h2, h3 {
    color: #00F5D4;
}

.metric-card {
    background: linear-gradient(145deg, #132F3F, #0B1F2A);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 0 15px #00F5D4;
    text-align: center;
}

.stButton>button {
    background-color: #00F5D4;
    color: black;
    border-radius: 8px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #00c9a7;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ AI-Powered Network Intrusion Detection System")

# =========================
# FIXED MODEL PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "best_model.pkl")
selector_path = os.path.join(BASE_DIR, "selector.pkl")
features_path = os.path.join(BASE_DIR, "feature_columns.pkl")

if not all(os.path.exists(p) for p in [model_path, selector_path, features_path]):
    st.error("❌ Model files not found inside src folder.")
    st.stop()

model = joblib.load(model_path)
selector = joblib.load(selector_path)
feature_columns = joblib.load(features_path)

# =========================
# SESSION TRACKING
# =========================
if "attack_count" not in st.session_state:
    st.session_state.attack_count = 0

if "normal_count" not in st.session_state:
    st.session_state.normal_count = 0

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🔐 Navigation")
page = st.sidebar.radio(
    "Go To:",
    ["Dashboard", "Prediction", "Live Simulation"]
)

# =========================================================
# DASHBOARD PAGE
# =========================================================
if page == "Dashboard":

    st.subheader("🌐 Security Overview")

    col1, col2, col3 = st.columns(3)

    col1.markdown("""
    <div class="metric-card">
        <h2>Model Accuracy</h2>
        <h1>99.78%</h1>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown("""
    <div class="metric-card">
        <h2>SMOTE Applied</h2>
        <h1>✔ Balanced</h1>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown("""
    <div class="metric-card">
        <h2>Algorithm</h2>
        <h1>XGBoost</h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if os.path.exists(os.path.join(BASE_DIR, "roc_curve.png")):
        st.subheader("📊 ROC Curve")
        st.image(os.path.join(BASE_DIR, "roc_curve.png"))

    if os.path.exists(os.path.join(BASE_DIR, "feature_importance.png")):
        st.subheader("🔥 Feature Importance")
        st.image(os.path.join(BASE_DIR, "feature_importance.png"))

    # Attack Distribution Pie Chart
    st.markdown("---")
    st.subheader("📊 Real-Time Attack Distribution")

    total = st.session_state.attack_count + st.session_state.normal_count

    if total > 0:
        fig2, ax2 = plt.subplots()
        ax2.pie(
            [st.session_state.normal_count, st.session_state.attack_count],
            labels=["Normal", "Attack"],
            autopct='%1.1f%%'
        )
        ax2.set_title("Traffic Distribution")
        st.pyplot(fig2)
    else:
        st.info("No predictions made yet.")

# =========================================================
# PREDICTION PAGE
# =========================================================
elif page == "Prediction":

    st.subheader("🧠 Predict Network Traffic")

    mode = st.radio(
        "Select Prediction Mode",
        ["Manual Input", "Upload CSV File"]
    )

    # ================= MANUAL =================
    if mode == "Manual Input":

        user_input = {}
        for col in feature_columns[:20]:
            user_input[col] = st.number_input(col, value=0.0)

        input_df = pd.DataFrame([user_input])

        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[feature_columns]
        input_selected = selector.transform(input_df)

        if st.button("🚀 Analyze Traffic"):

            prediction = model.predict(input_selected)[0]
            prob = model.predict_proba(input_selected)[0][1]

            # Track counts
            if prediction == 1:
                st.session_state.attack_count += 1
            else:
                st.session_state.normal_count += 1

            if prediction == 1:
                st.error(f"🚨 ATTACK DETECTED (Confidence: {prob:.4f})")
            else:
                st.success(f"✅ Normal Traffic (Confidence: {1-prob:.4f})")

            # Risk Level
            st.subheader("⚠ Threat Severity Level")
            if prob > 0.75:
                st.error("🔴 HIGH RISK")
            elif prob > 0.40:
                st.warning("🟡 MEDIUM RISK")
            else:
                st.success("🟢 LOW RISK")

            # SHAP
            st.subheader("🔍 AI Explanation")

            selected_feature_names = np.array(feature_columns)[selector.get_support()]
            input_selected_df = pd.DataFrame(
                input_selected,
                columns=selected_feature_names
            )

            explainer = shap.Explainer(model)
            shap_values = explainer(input_selected_df)

            fig, ax = plt.subplots()
            shap.plots.bar(shap_values, show=False)
            st.pyplot(fig)

            # Top 5 Features
            st.subheader("📌 Top 5 Contributing Features")

            shap_values_array = shap_values.values[0]

            feature_importance_df = pd.DataFrame({
                "Feature": selected_feature_names,
                "Impact": shap_values_array
            })

            feature_importance_df["Abs_Impact"] = feature_importance_df["Impact"].abs()

            top5 = feature_importance_df.sort_values(
                by="Abs_Impact",
                ascending=False
            ).head(5)

            st.table(top5[["Feature", "Impact"]])

    # ================= CSV =================
    else:

        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

        if uploaded_file is not None:

            df = pd.read_csv(uploaded_file)
            st.write("Preview of Uploaded Data", df.head())

            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0

            df = df[feature_columns]
            df_selected = selector.transform(df)

            predictions = model.predict(df_selected)
            probabilities = model.predict_proba(df_selected)[:, 1]

            df["Prediction"] = np.where(predictions == 1, "Attack", "Normal")
            df["Attack_Probability"] = probabilities

            # Update session
            attack_total = sum(predictions)
            normal_total = len(predictions) - attack_total

            st.session_state.attack_count += attack_total
            st.session_state.normal_count += normal_total

            # Batch Summary
            st.subheader("📊 Batch Threat Summary")

            summary_df = pd.DataFrame({
                "Metric": [
                    "Total Records",
                    "Normal Traffic",
                    "Attack Traffic",
                    "Attack Percentage"
                ],
                "Value": [
                    len(predictions),
                    normal_total,
                    attack_total,
                    f"{(attack_total/len(predictions))*100:.2f}%"
                ]
            })

            st.table(summary_df)

            st.success("✅ Batch Prediction Completed")
            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results",
                data=csv,
                file_name="prediction_results.csv",
                mime="text/csv"
            )

# =========================================================
# LIVE SIMULATION
# =========================================================
elif page == "Live Simulation":

    st.subheader("📡 Real-Time Intrusion Monitoring")

    if st.button("Start Monitoring"):

        placeholder = st.empty()

        for i in range(20):

            random_input = pd.DataFrame(
                np.random.rand(1, len(feature_columns)),
                columns=feature_columns
            )

            random_selected = selector.transform(random_input)

            pred = model.predict(random_selected)[0]
            prob = model.predict_proba(random_selected)[0][1]

            if pred == 1:
                st.session_state.attack_count += 1
            else:
                st.session_state.normal_count += 1

            result = "🚨 ATTACK" if pred == 1 else "✅ NORMAL"

            placeholder.markdown(f"""
            <div class="metric-card">
                <h2>Network Status</h2>
                <h1>{result}</h1>
                <p>Attack Probability: {prob:.4f}</p>
            </div>
            """, unsafe_allow_html=True)

            time.sleep(1)

    st.info("Simulation generates synthetic traffic for demonstration.")
