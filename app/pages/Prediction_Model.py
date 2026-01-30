import streamlit as st
import pickle
import pandas as pd
import numpy as np

# page configuration
st.set_page_config(
    page_title="Wildfire Risk Predictor",
    page_icon="🔥",
    layout="centered"
)

# page title
st.markdown("# Wildfire Risk Predictor")

# page summary
st.markdown(
    "<p style='color:#6B7280; max-width: 800px;'>"
    "This tool estimates wildfire ignition risk using historical "
    "weather, atmospheric, and geographic data."
    "</p>",
    unsafe_allow_html=True
)

st.markdown("### Welcome to the **Wildfire Risk Predictor**!")
st.markdown("Enter environmental values in the sidebar on the left, then press **Predict** to estimate wildfire risk for the specified location and time.")

# load the model, scaler and pca files

with open('app/model.pkl', 'rb') as f: model = pickle.load(f)
with open('app/scaler.pkl', 'rb') as f: scaler = pickle.load(f)
with open('app/pca.pkl', 'rb') as f: pca = pickle.load(f)


## INPUTS

# all inputs will be on the sidebar
st.sidebar.header("Input Parameters")

# split the sidebar into sections using markdown
st.sidebar.markdown("---")
st.sidebar.markdown("## Context & Location")

# selection box
daynight_N = st.sidebar.selectbox(
    "Time of Day",
    options=[0, 1],
    format_func=lambda x: "Night" if x == 0 else "Day",
    help="Indicates whether the observation occurred during nighttime (0) or daytime (1)."
)

# number inputs
lat = st.sidebar.number_input(
    "Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=0.0,
    help="Geographic latitude of the location being evaluated."
)
lon = st.sidebar.number_input(
    "Longitude",
    min_value=-180.0,
    max_value=180.0,
    value=0.0,
    help="Geographic longitude of the location being evaluated."
)

st.sidebar.markdown("---")
st.sidebar.markdown("## Atmospheric & Moisture Conditions")

pressure_mean = st.sidebar.number_input(
    "Mean Atmospheric Pressure (hPa)",
    min_value=500.0,
    max_value=1500.0,
    value=1013.0,
    help="Average atmospheric pressure over the past hour. Lower pressure often corresponds to unstable weather."
)

dewpoint_mean = st.sidebar.slider(
    "Mean Dew Point Temperature (°C)",
    min_value=-40,
    max_value=30,
    value=5,
    help="Temperature at which air becomes saturated with moisture. Lower values indicate drier air and higher fire risk."
)

humidity_min = st.sidebar.slider(
    "Minimum Relative Humidity (%)",
    min_value=0,
    max_value=100,
    value=30,
    help="Lowest humidity observed in the past hour. Low humidity increases fuel dryness."
)

cloud_cover_mean = st.sidebar.slider(
    "Mean Cloud Cover (%)",
    min_value=0,
    max_value=100,
    value=20,
    help="Average percentage of sky covered by clouds."
)

evapotranspiration_total = st.sidebar.slider(
    "Total Evapotranspiration (mm)",
    min_value=0.0,
    max_value=40.0,
    value=2.0,
    help="Total amount of water lost from soil and vegetation. Higher values indicate drier fuels."
)

st.sidebar.markdown("---")
st.sidebar.markdown("## Radiation & Energy")

solar_radiation_mean = st.sidebar.slider(
    "Mean Solar Radiation (W/m²)",
    min_value=0,
    max_value=500,
    value=200,
    help="Average solar energy reaching the surface. Higher values increase fuel heating and drying."
)

st.sidebar.markdown("---")
st.sidebar.markdown("## Temperature Conditions")

temp_mean = st.sidebar.slider(
    "Mean Temperature (°C)",
    min_value=-20,
    max_value=50,
    value=25,
    help="Average air temperature over the past hour."
)

temp_range = st.sidebar.slider(
    "Temperature Range (°C)",
    min_value=0,
    max_value=40,
    value=10,
    help="Difference between the maximum and minimum temperature during the hour."
)

st.sidebar.markdown("---")
st.sidebar.markdown("## Wind Conditions")

wind_speed_max = st.sidebar.slider(
    "Maximum Wind Speed (km/h)",
    min_value=0,
    max_value=150,
    value=20,
    help="Strongest wind gust recorded. Higher wind speeds increase fire spread potential."
)

wind_direction_mean = st.sidebar.number_input(
    "Mean Wind Direction (°)",
    min_value=0.0,
    max_value=359.0,
    value=180.0,
    help="Average wind direction during the hour, measured in degrees."
)

wind_direction_std = st.sidebar.slider(
    "Wind Direction Variability (°)",
    min_value=0,
    max_value=180,
    value=20,
    help="Variability in wind direction. Higher values indicate more erratic winds."
)

st.sidebar.markdown("---")
st.sidebar.markdown("## Fire Indices")

fire_weather_index = st.sidebar.slider(
    "Fire Weather Index",
    min_value=0,
    max_value=200,
    value=30,
    help="Composite index summarizing weather conditions relevant to wildfire ignition and spread."
)


## PREDICTION

# create a 'Predict' button
# if pressed, then:
if st.button("Predict"):
    
    # create a dataframe with the inputted values
    data = pd.DataFrame([[daynight_N, lat, lon, fire_weather_index, pressure_mean,
                          wind_direction_mean, wind_direction_std, solar_radiation_mean,
                          dewpoint_mean, cloud_cover_mean, evapotranspiration_total,
                          humidity_min, temp_mean, temp_range, wind_speed_max]],
                        columns=["daynight_N","lat","lon","fire_weather_index","pressure_mean",
                                 "wind_direction_mean","wind_direction_std","solar_radiation_mean",
                                 "dewpoint_mean","cloud_cover_mean","evapotranspiration_total",
                                 "humidity_min","temp_mean","temp_range","wind_speed_max"])

    # perform logarithmic transformation
    log_cols = ["fire_weather_index","wind_direction_std","solar_radiation_mean",
                "evapotranspiration_total","humidity_min","temp_range","wind_speed_max"]
    data[log_cols] = np.log1p(data[log_cols])
    
    # apply the scaler to the dataframe
    data = pd.DataFrame(scaler.transform(data))
    
    # apply the pca to the dataframe
    data = pd.DataFrame(pca.transform(data))

    # calculate the probability
    prob = model.predict_proba(data)[0, 1]
    # make a prediction based on the probability
    pred = int(prob > 0.4)

    # create a subheader
    st.subheader("Prediction Results")
    # print the probability as a metric widget
    st.metric("Probability of Wildfire Occurrence", f"{prob*100:.2f}%")  # multiply by 100 and round to 2dp
    
    # if prediction is 0, print a 'success' box (green box with green text)
    # print the specified text
    if pred == 0:
        st.success("**It is unlikely that a wildfire will occur.**")
   
    # if prediction is 1, print an 'error' box (red box with red text)
    # print the specified text
    else:
        st.error("**A wildfire is likely to occur.** \n\n _**Guidance:** Move to safety immediately, call emergency services, and do **not** attempt to fight the fire._")

    # display the risk level (low/moderate/high) based on the calculated probability
    if prob < 0.4:
        risk = "Low"
        colour = "green"
    elif prob < 0.6:
        risk = "Moderate"
        colour = "orange"
    else:
        risk = "High"
        colour = "red"

    st.markdown(f"#### Risk Level: :{colour}[{risk}]")


## OVERVIEW
st.markdown("---")
st.markdown("## Model Overview")

st.markdown("""
The model chosen was an XGBoost classifier trained on PCA-transformed and standardized features. 
The pipeline uses logarithmic transformations for skewed variables, standard scaling, then PCA before training. This approach helps 
stabilize training and reduce collinearity, but interpretability is at the component level.
""")

st.markdown("### Performance")

st.markdown("The chosen metric for model evaluation was recall to prioritise the early detection of wildfires.")

# create 3 columns for side-by-side display
col1, col2, col3 = st.columns(3)

# assign metrics to each column
col1.metric("Recall", "96%", "High sensitivity")
col2.metric("Precision", "56%", "Some false alarms")
col3.metric("Overall Accuracy", "60%", "Overall correctness")

st.markdown("""
> High recall achieved meaning that most wildfire events are detected.

> Moderate precision means that some false alarms are to be expected.

> The accuracy shows the overall correctness of the model but was not the primary metric for this task.
""")

st.markdown("### Confusion Matrix")

# create two columns for confusion matrix and interpretation side-by-side
col1, col2 = st.columns([2, 1])

with col1:
    # insert image of confusion matrix
    st.image(
    "app/assets/confusion_matrix.png",
    caption="Confusion matrix (rows = true class, cols = predicted class)",
    width = 450
    )

with col2:
    # interpretation of matrix
    st.markdown("\n")
    st.markdown("\n")
    st.markdown(
        "<p style='color:#374151;'>"
        "<strong>Interpretation:</strong> The model achieves very high "
        "recall for wildfires (TP / (TP+FN) ≈ 96%), meaning most actual "
        "fires are detected. <br><br>"
        "However, there are many false positives (FP), which is an "
        "intentional trade-off for early warning."
        "</p>",
        unsafe_allow_html=True
    )

st.markdown("### Decision Threshold")

st.markdown("""
Early detection of wildfires was prioritised in this model so the objective was to obtain 
a high recall score.

Therefore, a probability threshold of **0.40** is used to classify wildfire risk.
Lowering the threshold increases sensitivity to wildfire events at the cost
of additional false positives.

This threshold was selected to balance early detection with operational
usability.
""")

# add disclaimer
st.warning("""
This tool is for **educational and analytical purposes only**.
Predictions should not replace official wildfire warnings, emergency alerts,
or professional judgment. Environmental conditions can change rapidly.
""")

# page caption
st.markdown("---")
st.caption("Prediction — Wildfire Risk Prediction Project")