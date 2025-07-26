import streamlit as st
import joblib
import numpy as np
import pandas as pd
from utils.preprocessor import preprocess_input

# Page config
st.set_page_config(page_title="GHG Emission Predictor", page_icon="🌍", layout="wide")

# Custom CSS styling for green theme and top-bar UI
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f9f0;
            font-family: 'Segoe UI', sans-serif;
        }
        .main-title {
            text-align: center;
            font-size: 2.5rem;
            color: #145214;
        }
        .subtext {
            text-align: center;
            font-size: 1.1rem;
            color: #1b3d1f;
            margin-bottom: 2rem;
        }
        .block {
            background-color: #ffffffcc;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.5rem 1rem;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        h2, h3 {
            color: #145214;
        }
    </style>
""", unsafe_allow_html=True)

# Load ML model and scaler
model = joblib.load('models/LR_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Main title
st.markdown("<div class='main-title'>🌱 Supply Chain Emissions Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Predict supply chain emission factors with margins using data quality metrics.</div>", unsafe_allow_html=True)

# Top tabs
tab1, tab2 = st.tabs(["📥 Predict Emissions", "📘 About"])

with tab1:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.header("📝 Enter Input Parameters")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            substance = st.selectbox("🌫️ Substance", ['carbon dioxide', 'methane', 'nitrous oxide', 'other GHGs'])
            unit = st.selectbox("📐 Unit", ['kg/2018 USD, purchaser price', 'kg CO2e/2018 USD, purchaser price'])
            source = st.selectbox("🏭 Source", ['Commodity', 'Industry'])

        with col2:
            supply_wo_margin = st.number_input("🚛 Supply Chain Emission Factors *without Margins*", min_value=0.0, format="%.4f")
            margin = st.number_input("💰 Margins of Supply Chain Emission Factors", min_value=0.0, format="%.4f")

        st.subheader("📊 Data Quality (DQ) Metrics")

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

with tab2:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.header("📘 About This App")
    st.markdown("""
    Welcome to the **GHG Emission Predictor App** developed by **Ismail Hossen**.

    This application predicts **Supply Chain Emission Factors with Margins** using a trained machine learning model.

    ### 🧾 Features:
    - 📌 Input types of GHGs (CO₂, CH₄, etc.)
    - 📊 Enter DQ (Data Quality) metrics
    - 🧠 Powered by a linear regression model
    - 🔬 Uses preprocessing and scaling for accurate results

    ### 👨‍💻 Developer:
    - **Ismail Hossen**
    - [GitHub Profile](https://github.com/ismail11-star)

    ### ✅ Models Status:
    - Model & Scaler loaded successfully!
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
