import streamlit as st
import logging
from logging import getLogger
import uuid
import requests
import json
import random
import yaml
import os
import time

logger = getLogger()
if logger.handlers:  # logger is already setup, don't setup again
    pass
else:
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

def init_random_inputs():
    st.session_state.setdefault("cloud_cover", round(random.uniform(0.0, 9.0), 1))
    st.session_state.setdefault("sunshine", round(random.uniform(0.0, 16.0), 1))
    st.session_state.setdefault("global_radiation", round(random.uniform(8.0, 402.0), 1))
    st.session_state.setdefault("max_temp", round(random.uniform(-6.2, 37.9), 1))
    st.session_state.setdefault("min_temp", round(random.uniform(-11.8, 22.3), 1))
    st.session_state.setdefault("precipitation", round(random.uniform(0.0, 61.8), 1))
    st.session_state.setdefault("pressure", round(random.uniform(95960.0, 104820.0), 1))
    st.session_state.setdefault("snow_depth", round(random.uniform(0.0, 22.0), 1))

def main_page():
    st.set_page_config(page_title="Weather Prediction", layout="centered")
    st.title('🌤️ DVC + API + Docker Application')
    
    with st.status("🕒 API is warming up (free tier may take a while)...", expanded=True) as status:
        try:
            response = requests.get("https://prj-london-weather.onrender.com/", timeout=60)
            if response.status_code == 200:
                status.update(label="✅ API is ready!", state="complete")
            else:
                status.update(label=f"⚠️ API responded with status code {response.status_code}", state="error")
        except requests.RequestException:
            status.update(label="❌ Unable to reach API. It may still be waking up.", state="error")
        
    st.subheader("Enter Weather Parameters")
        
    with st.form("weather_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            cloud_cover = st.number_input("Cloud Cover (oktas)", min_value=0.0, format="%.1f", key="cloud_cover")
            sunshine = st.number_input("Sunshine (hrs)", min_value=0.0, format="%.1f", key="sunshine")
            global_radiation = st.number_input("Global Radiation (W/m²)", min_value=0.0, format="%.1f", key="global_radiation")
            max_temp = st.number_input("Max Temperature (°C)", format="%.1f", key="max_temp")

        with col2:
            min_temp = st.number_input("Min Temperature (°C)", format="%.1f", key="min_temp")
            precipitation = st.number_input("Precipitation (mm)", min_value=0.0, format="%.1f", key="precipitation")
            pressure = st.number_input("Pressure (Pa)", min_value=0.0, format="%.1f", key="pressure")
            snow_depth = st.number_input("Snow Depth (cm)", min_value=0.0, format="%.1f", key="snow_depth")

        # Every form must have a submit button.
        button_col, message_col = st.columns([1, 3])
        with button_col:
            submitted = st.form_submit_button("🔍 Submit Prediction")
        message_placeholder = message_col.empty()
        
        if submitted:
            payload_json = {
                "cloud_cover": float(cloud_cover),
                "sunshine": float(sunshine),
                "global_radiation": float(global_radiation),
                "max_temp": float(max_temp),
                "min_temp": float(min_temp),
                "precipitation": float(precipitation),
                "pressure": float(pressure),
                "snow_depth": float(snow_depth),
            }

            response = requests.post(
                "https://prj-london-weather.onrender.com/prediction/",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload_json),
                timeout=60
            )
                
            try:
                if response.status_code == 200:
                    message_placeholder.success("✅ Prediction Received!")
                    st.markdown(f"🌡️ Mean Temperature: {round(response.json().get("response"), 1)} °C")
                else:
                    message_placeholder.error(f"❌ API returned status {response.status_code}")
            except requests.RequestException as e:
                message_placeholder.error(f"Failed to reach API: {e}")
        
    docs_button = st.button("📄 Go to documentations page")
    if docs_button:
        st.session_state.page = 'docs'
        st.rerun()   
                
def docs_page():
    st.set_page_config(page_title="📄 Documentation Page", layout="centered")
    st.title('📄 Project Documentation')
    
    st.markdown("### 🧠 GitHub Repository")
    st.markdown("[🔗 View on GitHub](https://github.com/VD-git/prj-london-weather)")

    st.markdown("## 🔍 Model Metrics")
    try:
        with open(os.path.join("output", "model_metrics.json"), "r") as f:
            metrics = json.load(f)
        with st.expander("Show model metrics"):
            st.json(metrics)
    except Exception as e:
        st.error(f"Error loading model metrics: {e}")

    st.markdown("## 🐳 Docker Image")
    st.markdown("[View on Docker Hub](https://hub.docker.com/repository/docker/victordias94/london_api/tags)")
    
    st.markdown("### 🌐 API Documentation")
    st.info("🚨 Remember: this is a free API hosted on Render and is publicly accessible. "
            "Please do not overload it with too many requests, it may crash, use wisely.")

    with st.expander("🔎 GET Method"):
        st.markdown("**Endpoint:** [https://prj-london-weather.onrender.com/](https://prj-london-weather.onrender.com/)")
        st.code("curl https://prj-london-weather.onrender.com/", language="bash")
        st.markdown("Returns a simple welcome or health check message.")

    with st.expander("📬 POST Method"):
        st.markdown("**Endpoint:** [https://prj-london-weather.onrender.com/prediction/](https://prj-london-weather.onrender.com/prediction/)")
        st.markdown("Send weather-related features to get a prediction from the model.")

        st.markdown("**Example `curl` request:**")
        st.code("""
            curl -X POST https://prj-london-weather.onrender.com/prediction/ \\
                -H "Content-Type: application/json" \\
                -d '{
                    "cloud_cover": 75.0,
                    "sunshine": 3.5,
                    "global_radiation": 120.8,
                    "max_temp": 18.2,
                    "min_temp": 10.4,
                    "precipitation": 2.3,
                    "pressure": 101200.6,
                    "snow_depth": 0.0
                    }'
            """, language="bash")

        st.markdown("**JSON payload format:**")
        st.code("""
            {
            "cloud_cover": 75.0,
            "sunshine": 3.5,
            "global_radiation": 120.8,
            "max_temp": 18.2,
            "min_temp": 10.4,
            "precipitation": 2.3,
            "pressure": 101200.6,
            "snow_depth": 0.0
            }
            """, language="json")

    st.markdown("## 📦 DVC Configuration")
    try:
        with open("dvc.yaml", "r") as f:
            dvc_config = yaml.safe_load(f)
        with st.expander("Show DVC configuration"):
            st.code(yaml.dump(dvc_config), language="yaml")
    except Exception as e:
        st.error(f"Error loading DVC config: {e}")

    # Navigation button
    if st.button("🔙 Back to Main Page"):
        st.session_state.page = 'main'
        st.rerun()
    
if __name__ == "__main__":

    if "page" not in st.session_state:
        init_random_inputs()
        st.session_state.page = "main"

    if st.session_state.page == "main":
        main_page()
    elif st.session_state.page == "docs":
        docs_page()

        
    
    
                
