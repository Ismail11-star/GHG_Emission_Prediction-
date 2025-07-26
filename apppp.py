import streamlit as st
import joblib
import numpy as np
import pandas as pd
from utils.preprocessor import preprocess_input

# Page configuration
st.set_page_config(
    page_title="GHG Emission Predictor",
    page_icon="🌍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Modern responsive styling with mobile support
st.markdown("""
    <style>
        html, body, .stApp {
            background: linear-gradient(to right, #eafaf1, #d3f9d8);
            font-family: 'Segoe UI', sans-serif;
            color: #1b3d1f;
            padding: 0;
            margin: 0;
        }
        .block-container {
            padding: 2rem 1rem;
            max-width: 700px;
            margin: auto;
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        .stSidebar {
            background-color: #d0f0c0;
            padding: 2rem 1rem;
        }
        .stSidebar .css-1d391kg {
            background-color: #ffffffcc;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 2px 4px 10px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            background-color: #28a745;
            color: white;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
            border-radius: 10px;
            border: none;
            font-size: 1rem;
        }
        .stButton>button:hover {
            background-color: #218838;
        }
        h1, h2, h3 {
            color: #145214;
        }
        @media screen and (max-width: 768px) {
            .block-container {
                padding: 1.5rem 0.5rem;
                width: 100%;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
model = joblib.load('models/LR_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Sidebar content
with st.sidebar:
    st.title("📊 GHG Emission Predictor")
    st.markdown("Predict **Supply Chain Emission Factors with Margins** based on emission metrics.")
    st.markdown("Made by ❤️ **Ismail Hossen**")
    st.markdown("---")
    st.success("📁 Models loaded successfully!")

# Main title
st.title("🌱 Supply Chain Emissions Prediction")

st.markdown("""
Welcome to the **GHG Emission Predictor**.  
Forecast **supply chain emission factors** with margins using GHG data and DQ metrics.
""")

# Tabs
tab1, tab2 = st.tabs(["📥 Predict Emissions", "ℹ️ About"])

# Prediction tab
with tab1:
    st.header("📝 Input Parameters")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            substance = st.selectbox("🌫️ Substance", ['carbon dioxide', 'methane', 'nitrous oxide', 'other GHGs'])
            unit = st.selectbox("📐 Unit", ['kg/2018 USD, purchaser price', 'kg CO2e/2018 USD, purchaser price'])
            source = st.selectbox("🏭 Source", ['Commodity', 'Industry'])

        with col2:
            supply_wo_margin = st.number_input("🚛 Supply Chain Emission Factor (No Margin)", min_value=0.0, format="%.4f")
            margin = st.number_input("💰 Margin of Emission Factor", min_value=0.0, format="%.4f")

        st.subheader("📊 Data Quality Metrics")

        dq_reliability = st.slider("📉 Reliability", 0.0, 1.0, 0.5)
        dq_temporal = st.slider("⏳ Temporal Correlation", 0.0, 1.0, 0.5)
        dq_geo = st.slider("🗺️ Geographical Correlation", 0.0, 1.0, 0.5)
        dq_tech = st.slider("⚙️ Technological Correlation", 0.0, 1.0, 0.5)
        dq_data = st.slider("📚 Data Collection", 0.0, 1.0, 0.5)

        submit = st.form_submit_button("🔍 Predict")

    if submit:
        with st.spinner("🔄 Analyzing..."):
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

            st.success("✅ Prediction Complete")
            st.markdown(f"""
            ### 🌍 Predicted Emission Factor:
            **🔢 {prediction[0]:.4f}**
            """)
            st.balloons()

# About tab
with tab2:
    st.header("📖 About the App")
    st.markdown("""
This tool uses **machine learning** to estimate Supply Chain Emission Factors **with Margins**, leveraging:
- Linear Regression model
- Data Quality (DQ) metrics
- GHG substances (e.g., CO₂, CH₄, N₂O)

**Developer:** [Ismail Hossen](https://github.com/ismail11-star)
    """)

