import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import json

# Configuration
API_BASE_URL = "http://localhost:8000"

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.status_code == 200
    except:
        return False

def make_prediction(input_data):
    """Make a prediction via the API"""
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=input_data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def get_model_info():
    """Get model information from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/model-info")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

# Streamlit App
st.set_page_config(
    page_title="AI Smart Home Energy Forecast",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 AI Smart Home Energy Forecast")
st.markdown("Predict daily energy consumption using NASA climate data and machine learning")

# Check API status
if not check_api_health():
    st.error("⚠️ API is not running! Please start the API first with: `python api/main.py`")
    st.stop()
else:
    st.success("✅ API is running")

# Sidebar for additional info
with st.sidebar:
    st.header("ℹ️ Information")
    model_info = get_model_info()
    if model_info:
        st.subheader("Model Info")
        st.write(f"**Model:** {model_info.get('model_name', 'Unknown')}")
        st.write(f"**Version:** {model_info.get('version', 'Unknown')}")
        if 'training_metrics' in model_info:
            metrics = model_info['training_metrics']
            st.write("**Training Metrics:**")
            for key, value in metrics.items():
                if isinstance(value, float):
                    st.write(f"- {key}: {value:.4f}")
                else:
                    st.write(f"- {key}: {value}")

    st.markdown("---")
    st.markdown("### 📝 How to Use")
    st.markdown("""
    1. Enter location coordinates (latitude/longitude)
    2. Input current weather conditions
    3. Provide historical solar radiation data
    4. Click 'Predict' to get energy forecast
    """)

# Main form
st.header("📊 Enter Prediction Parameters")

# Location inputs
col1, col2 = st.columns(2)
with col1:
    latitude = st.number_input(
        "Latitude",
        value=40.7128,
        format="%.4f",
        help="Latitude of the location (e.g., 40.7128 for New York City)"
    )
with col2:
    longitude = st.number_input(
        "Longitude",
        value=-74.0060,
        format="%.4f",
        help="Longitude of the location (e.g., -74.0060 for New York City)"
    )

# Weather inputs
st.subheader("🌤️ Current Weather Conditions")
col1, col2, col3 = st.columns(3)

with col1:
    temperature = st.number_input(
        "Temperature (°C)",
        value=25.0,
        help="Temperature at 2 meters (T2M)"
    )
    humidity = st.number_input(
        "Humidity (%)",
        value=60.0,
        help="Relative Humidity at 2 meters (RH2M)"
    )

with col2:
    wind_speed = st.number_input(
        "Wind Speed (m/s)",
        value=4.1,
        help="Wind Speed at 10 meters (WS10M)"
    )
    pressure = st.number_input(
        "Pressure (hPa)",
        value=1012.5,
        help="Surface Pressure (PS)"
    )

with col3:
    cloud_amount = st.number_input(
        "Cloud Amount (%)",
        value=15.0,
        help="Cloud Amount (CLOUD_AMT)"
    )
    precipitation = st.number_input(
        "Precipitation (mm/day)",
        value=0.0,
        help="Precipitation (PRECTOTCORR)"
    )

# Historical solar radiation data
st.subheader("☀️ Historical Solar Radiation Data")
col1, col2, col3 = st.columns(3)

with col1:
    lag_1 = st.number_input(
        "Yesterday's Solar Radiation (kWh/m²/day)",
        value=14.2,
        help="Solar radiation from 1 day ago (ALLSKY_SFC_SW_DWN)"
    )

with col2:
    lag_7 = st.number_input(
        "7 Days Ago Solar Radiation (kWh/m²/day)",
        value=12.5,
        help="Solar radiation from 7 days ago"
    )

with col3:
    rolling_mean_7 = st.number_input(
        "7-Day Avg Solar Radiation (kWh/m²/day)",
        value=13.1,
        help="7-day rolling average of solar radiation"
    )

# Time features
st.subheader("📅 Time Information")
col1, col2, col3 = st.columns(3)

with col1:
    day_of_week = st.selectbox(
        "Day of Week",
        options=list(range(7)),
        format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
        index=3,  # Thursday
        help="Day of week (0=Monday, 6=Sunday)"
    )

with col2:
    month = st.selectbox(
        "Month",
        options=list(range(1, 13)),
        index=5,  # June
        help="Month of year (1-12)"
    )

with col3:
    season = st.selectbox(
        "Season",
        options=[0, 1, 2, 3],
        format_func=lambda x: ["Winter", "Spring", "Summer", "Fall"][x],
        index=2,  # Summer
        help="Season index (0=Winter, 1=Spring, 2=Summer, 3=Fall)"
    )

# Predict button
if st.button("🔮 Predict Energy Consumption", type="primary", use_container_width=True):
    # Prepare input data for API
    input_data = {
        "latitude": latitude,
        "longitude": longitude,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "pressure": pressure,
        "cloud_amount": cloud_amount,
        "precipitation": precipitation,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "rolling_mean_7": rolling_mean_7,
        "day_of_week": day_of_week,
        "month": month,
        "season": season
    }

    # Show loading spinner
    with st.spinner("Making prediction..."):
        result = make_prediction(input_data)

    if result:
        st.success("✅ Prediction Successful!")

        # Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Predicted Energy Consumption",
                value=f"{result['predicted_energy_consumption']:.2f} {result['unit']}"
            )

        with col2:
            st.metric(
                label="Confidence",
                value=result['confidence'].title()
            )

        with col3:
            st.metric(
                label="Model Used",
                value=result['model_used']
            )

        # Show additional info
        with st.expander("📋 Prediction Details"):
            st.json(result)

        # Show input summary
        with st.expander("📥 Input Summary"):
            st.json(input_data)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit and FastAPI | AI for Smart Home Energy Forecasting @Dayanand</p>
    </div>
    """,
    unsafe_allow_html=True
)