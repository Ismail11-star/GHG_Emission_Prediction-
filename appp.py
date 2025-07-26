import streamlit as st
import joblib
import numpy as np
import pandas as pd
from utils.preprocessor import preprocess_input

# Set page config
st.set_page_config(page_title="GHG Emission Predictor", page_icon="🌍", layout="centered")

# Custom CSS styling for green background and UI enhancements
st.markdown("""
    <style>
        body {
            background-color: #e6f2e6;
        }
        .stApp {
            background-color: #e6f2e6;
            color: #1b3d1f;
            font-family: 'Segoe UI', sans-serif;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            border-radius: 10px;
            background-color: #ffffffcc;
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
        .stSidebar {
            background-color: #d9f2d9;
        }
        h1, h2, h3 {
            color: #145214;
        }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
model = joblib.load('models/LR_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Sidebar
with st.sidebar:
    st.title("📊 GHG Emission Predictor")
    st.markdown("Use this app to predict **Supply Chain Emission Factors with Margins** based on input metrics.")
    st.markdown("Made ❤️ by Ismail Hossen")
    st.markdown("---")
    st.info("📁 Models loaded successfully!")

# Main Title
st.title("🌱 Supply Chain Emissions Prediction")

st.markdown("""
Welcome to the **GHG Emission Predictor App**.  
This tool helps forecast **supply chain emission factors** with margin using DQ metrics and source information.
""")

# Input form inside a tab
tab1, tab2 = st.tabs(["📥 Predict Emissions", "ℹ️ About App"])

with tab1:
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

with tab2:
    st.header("📖 About This App")
    st.markdown("""
    This application predicts **Supply Chain Emission Factors with Margins** using a trained machine learning model.
    
    **Inputs include:**
    - GHG substance type (e.g., CO₂, CH₄)
    - Emission units
    - DQ (Data Quality) metrics

    **Prediction is based on:**
    - Linear regression model
    - Preprocessed and scaled input values

    **Developer:** [Ismail Hossen](https://github.com/ismail11-star)
    """)
