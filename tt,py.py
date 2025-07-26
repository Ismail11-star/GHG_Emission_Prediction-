import streamlit as st
import joblib
import numpy as np
import pandas as pd
from utils.preprocessor import preprocess_input

# Page config
st.set_page_config(page_title="GHG Emission Predictor", page_icon="🌍", layout="wide")

# ---------- Custom CSS Styling ----------
st.markdown("""
    <style>
        body {
            background-color: #f5fff5;
        }

        .stApp {
            font-family: 'Segoe UI', sans-serif;
        }

        .main-title {
            text-align: center;
            font-size: 2.8rem;
            font-weight: bold;
            color: #145214;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            text-align: center;
            font-size: 1.1rem;
            color: #1b3d1f;
            margin-bottom: 2rem;
        }

        .section {
            background-color: #ffffff;
            padding: 2.5rem 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
            max-width: 1000px;
            margin: auto;
        }

        label, .stSelectbox label, .stNumberInput label, .stSlider label {
            font-weight: bold !important;
            font-size: 1.05rem !important;
            color: #144d14 !important;
        }

        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > div {
            border: 2px solid #4CAF50 !important;
            border-radius: 8px !important;
            font-size: 1.05rem !important;
        }

        .stSlider > div > div {
            padding-top: 1rem;
        }

        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            font-size: 1rem;
            margin-top: 1rem;
        }

        .stButton > button:hover {
            background-color: #388e3c;
        }

        @media (max-width: 768px) {
            .section {
                padding: 1.5rem 1rem;
            }

            .main-title {
                font-size: 2rem;
            }

            label {
                font-size: 1rem !important;
            }
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Load Model and Scaler ----------
model = joblib.load('models/LR_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# ---------- Header ----------
st.markdown("<div class='main-title'>🌱 GHG Emission Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Estimate Supply Chain Emission Factors with DQ Metrics</div>", unsafe_allow_html=True)

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["📥 Predict Emissions", "📘 About"])

# ---------- Tab 1: Prediction Form ----------
with tab1:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("📝 Input Parameters")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            substance = st.selectbox("🌫️ Substance", ['carbon dioxide', 'methane', 'nitrous oxide', 'other GHGs'])
            unit = st.selectbox("📐 Unit", ['kg/2018 USD, purchaser price', 'kg CO2e/2018 USD, purchaser price'])
            source = st.selectbox("🏭 Source", ['Commodity', 'Industry'])

        with col2:
            supply_wo_margin = st.number_input("🚛 Supply Chain Emission Factors *without Margins*", min_value=0.0, format="%.4f")
            margin = st.number_input("💰 Margins of Supply Chain Emission Factors", min_value=0.0, format="%.4f")

        st.markdown("### 📊 Data Quality (DQ) Metrics")

        dq_reliability = st.slider("📉 DQ Reliability", 0.0, 1.0, 0.5)
        dq_temporal = st.slider("⏳ DQ Temporal Correlation", 0.0, 1.0, 0.5)
        dq_geo = st.slider("🗺️ DQ Geographical Correlation", 0.0, 1.0, 0.5)
        dq_tech = st.slider("⚙️ DQ Technological Correlation", 0.0, 1.0, 0.5)
        dq_data = st.slider("📚 DQ Data Collection", 0.0, 1.0, 0.5)

        submit = st.form_submit_button("🔍 Predict Emissions")

    if submit:
        with st.spinner("🔄 Processing your input..."):
            input_data = {
                'Substance': substance,
                'Unit': unit,
                'Supply Chain Emission Factors without Margins': supply_wo_margin,
                'Margins of Supply Chain Emission Factors': margin,
                'DQ ReliabilityScore of Factors without Margins': dq_reliability,
                'DQ TemporalCorrelation of Factors without Margins': dq_temporal,
                'DQ GeographicalCorrelation of Factors without Margins': dq_geo,
                'DQ TechnologicalCorrelation of Factors without Margins': dq_tech,
                'DQ DataCollection of Factors without Margins': dq_data,
                'Source': source,
            }

            input_df = preprocess_input(pd.DataFrame([input_data]))
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)

            st.success("✅ Prediction Complete!")
            st.markdown(f"""
                ### 🌍 Predicted Emission Factor:
                **🔢 {prediction[0]:.4f}**
            """)
            st.balloons()

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Tab 2: About ----------
with tab2:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("📘 About This App")

    st.markdown("""
    This application predicts **Supply Chain Emission Factors with Margins** using a trained ML model.

    ### 📋 Features:
    - Predicts emission factors based on:
      - GHG substance
      - Margin
      - DQ metrics
    - Linear Regression model for prediction
    - Scaled, preprocessed inputs for accuracy

    ### 👨‍💻 Developer:
    - **Ismail Hossen**
    - [GitHub Profile](https://github.com/ismail11-star)

    ### ✅ Model Status:
    - ML Model & Scaler loaded successfully.
    """)

    st.markdown("</div>", unsafe_allow_html=True)
