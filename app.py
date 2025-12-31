import streamlit as st
import pickle
import pandas as pd
import numpy as np
import base64

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="Wildfire Risk Predictor",
    page_icon="🔥",
    layout="centered"
)

# -----------------------
# BACKGROUND IMAGE
# -----------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1502252430442-aac78f397426?q=80&w=1770&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# CSS STYLING
# -----------------------
st.markdown("""
<style>
/* Headings and description */
.text-visible { color: white !important; text-shadow: 2px 2px 4px black; font-size: 22px !important; }
.sub-title { color: white !important; text-shadow: 1px 1px 3px black; font-size: 20px !important; }

/* Feature labels including Time of Day */
label { color: white !important; font-size: 18px !important; font-weight: bold; }

/* FIX: Make selectbox label white, keep dropdown text black */
.stSelectbox label p {
    color: white !important;
}

/* Inputs values readable */
.stNumberInput input, .stSelectbox div, .stTextInput input { color: black !important; font-size: 18px !important; }

/* Tooltip / help text and icon */
.stTooltipContent { color: black !important; background-color: rgba(255,255,255,0.95) !important; font-size:16px !important; }
span[data-testid="stTooltip"] svg { fill: white !important; }

/* Predict button dark grey */
div.stButton>button {
    background-color: #333333 !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 12px 20px !important;
    font-size: 20px !important;
    border: 1px solid white !important;
    box-shadow: 0px 0px 6px black !important;
}
div.stButton>button:hover { background-color: #555555 !important; }
</style>
""", unsafe_allow_html=True)



# -----------------------
# LOAD MODEL FILES
# -----------------------
with open('model.pkl', 'rb') as f: model = pickle.load(f)
with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
with open('pca.pkl', 'rb') as f: pca = pickle.load(f)


# -----------------------
# HEADINGS AND DESCRIPTION
# -----------------------
st.markdown('<div class="text-visible"><h1>Wildfire Risk Predictor</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="text-visible">Welcome to the <b>Wildfire Risk Predictor</b>!<br>'
            'Enter environmental and weather conditions to estimate the <i>probability</i> of a wildfire starting.</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title"><h3>Enter Features Below!</h3></div>', unsafe_allow_html=True)
st.markdown('<div class="text-visible"><i>All variables represent measurements over a one-hour period.</i></div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)  # spacing before columns


# -----------------------
# INPUT COLUMNS
# -----------------------
col1, col2 = st.columns(2)

with col1:
    daynight_N = st.selectbox("Time of Day:", (0, 1),
                              help="0 = Night, 1 = Day")
    lat = st.number_input("Latitude:", value=0.0, min_value=-90.0, max_value=90.0,
                          help="Geographic latitude of the location. [-90, 90]")
    lon = st.number_input("Longitude:", value=0.0, min_value=-180.0, max_value=180.0,
                          help="Geographic longitude of the location. [-180, 180]")
    fire_weather_index = st.number_input("Fire Weather Index:", min_value=0.0, max_value=250.0, value=0.0,
                                        help="Combined measure of weather conditions influencing fire hazard. [0, 250]")
    pressure_mean = st.number_input("Mean Atmospheric Pressure (hPa):", min_value=500.0, max_value=1500.0, value=1013.0,
                                    help="Average air pressure over the one hour period. [500, 1500]")
    wind_direction_mean = st.number_input("Wind Direction Mean (°):", min_value=0.0, max_value=359.0, value=0.0,
                                          help="Average wind direction in degrees over the one hour period. [0, 359]")
    wind_direction_std = st.number_input("Wind Direction Standard Deviation (°):", min_value=0.0, max_value=359.0, value=0.0,
                                         help="Variability in wind direction. [0, 359]")
    solar_radiation_mean = st.number_input("Mean Solar Radiation (W/m²):", min_value=0.0, max_value=500.0, value=0.0,
                                           help="Average sunlight intensity received at ground level. [0, 500]")

with col2:
    dewpoint_mean = st.number_input("Mean Dewpoint Temperature (°C):", min_value=-60.0, max_value=35.0, value=0.0,
                                    help="Temperature at which moisture in the air begins to condense. [-60, 35]")
    cloud_cover_mean = st.number_input("Mean Cloud Cover (%):", min_value=0.0, max_value=100.0, value=0.0,
                                       help="Percentage of sky covered by clouds. [0, 100]")
    evapotranspiration_total = st.number_input("Total Evapotranspiration (mm):", min_value=0.0, max_value=40.0, value=0.0,
                                               help="Total water evaporated from soil and transpired by plants over the one hour period [0, 40].")
    humidity_min = st.number_input("Minimum Humidity (%):", min_value=0.0, max_value=100.0, value=0.0,
                                   help="Lowest humidity recorded in the one hour period. [0, 100]")
    temp_mean = st.number_input("Mean Temperature (°C):", min_value=-50.0, max_value=50.0, value=0.0,
                                help="Average temperature over the one hour period. [-50, 50]")
    temp_range = st.number_input("Temperature Range (°C):", min_value=0.0, max_value=100.0, value=0.0,
                                 help="Difference between the highest and lowest temperature in the hour. [0, 100]")
    wind_speed_max = st.number_input("Maximum Wind Speed (km/h):", min_value=0.0, max_value=200.0, value=0.0,
                                     help="Strongest wind gust recorded in the one hour period. [0, 200]")

# -----------------------
# PREDICTION
# -----------------------
if st.button("Predict"):
    data = pd.DataFrame([[daynight_N, lat, lon, fire_weather_index, pressure_mean,
                          wind_direction_mean, wind_direction_std, solar_radiation_mean,
                          dewpoint_mean, cloud_cover_mean, evapotranspiration_total,
                          humidity_min, temp_mean, temp_range, wind_speed_max]],
                        columns=["daynight_N","lat","lon","fire_weather_index","pressure_mean",
                                 "wind_direction_mean","wind_direction_std","solar_radiation_mean",
                                 "dewpoint_mean","cloud_cover_mean","evapotranspiration_total",
                                 "humidity_min","temp_mean","temp_range","wind_speed_max"])

    log_cols = ["fire_weather_index","wind_direction_std","solar_radiation_mean",
                "evapotranspiration_total","humidity_min","temp_range","wind_speed_max"]

    data[log_cols] = np.log1p(data[log_cols])
    data = pd.DataFrame(scaler.transform(data))
    data = pd.DataFrame(pca.transform(data))

    prob = model.predict_proba(data)[0, 1]
    pred = int(prob > 0.4)

# --- Wrapped in readable panel ----
    #st.markdown('<div class="result-block">', unsafe_allow_html=True)
    
    
    st.markdown(
    "<h2 style='color:white;'>Prediction Results</h2>",
    unsafe_allow_html=True
)
    
    st.markdown(
    f"""
    <div style="
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background-color: #262730;  /* dark background similar to metric card */
        padding: 10px 20px;
        border-radius: 10px;
        width: 300px;
    ">
        <div style="font-size: 18px; color: #aaaaaa; margin-bottom: 5px;">
            Probability of Wildfire Occurrence
        </div>
        <div style="font-size: 36px; font-weight: bold; color: white;">
            {prob*100:.2f}%
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

    if pred == 0:
        st.markdown(
            """
            <div style="
                background-color: #28a745;  /* solid green */
                color: white;
                padding: 15px;
                border-radius: 8px;
                font-size: 20px;
            ">
                🌤️ <strong>It is unlikely that a wildfire will occur.</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="
                background-color: #dc3545;  /* solid red */
                color: white;
                padding: 15px;
                border-radius: 8px;
                font-size: 20px;
            ">
                🔥 <strong>A wildfire is likely to occur.</strong><br>
                <strong>Guidance:</strong> Move to safety immediately, call emergency services, and do <strong>not</strong> attempt to fight the fire.
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown('</div>', unsafe_allow_html=True)