import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# set theme colours
PRIMARY = "#2E3B4E"
SECONDARY = "#4B627D"
ACCENT = "#bf603d"
BG_COLOR = "#F5F7FA"
GRID_COLOR = "#E2E8F0"

# set global colours for matplotlib graphs
plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": PRIMARY,
    "axes.labelcolor": PRIMARY,
    "xtick.color": PRIMARY,
    "ytick.color": PRIMARY,
    "text.color": PRIMARY,
    "font.size": 10
})

# page configuration
st.set_page_config(
    page_title="Data Overview",
    page_icon="📊",
    layout="centered"
)

# page title
st.title("Data Overview")

# page summary
st.markdown(
    "<p style='color:#6B7280; max-width: 800px;'>"
    "This page provides an overview of the dataset used to train the wildfire "
    "risk prediction model. The goal is to establish transparency, credibility, "
    "and context before exploring predictions."
    "</p>",
    unsafe_allow_html=True
)

# load the dataset
# save cache to prevent reloading data every time page is refreshed
@st.cache_data
def load_data():
    return pd.read_csv("data/final_dataset.csv")
df = load_data()
# fix spelling error
df.rename(columns = {'occured' : 'occurred'}, inplace = True)

## DATASET SUMMARY SECTION

st.markdown("## Dataset Summary")

# add a drop-down for the data source
with st.expander("Data Source"):

    # link to dataset and brief description
    st.markdown("""
    [Kaggle: Global Wildfire Dataset](https://www.kaggle.com/datasets/vijayaragulvr/wildfire-prediction)
    
    The data contains historical wildfire occurrence data combined with
    meteorological and atmospheric observations.
    """)

    # convert data to csv in memory
    csv_data = df.to_csv(index=False).encode("utf-8")
    # create download button
    st.download_button(
    label="Download full dataset (CSV)",
    data=csv_data,
    file_name="wildfire_dataset.csv",
    mime="text/csv"
)


## METRICS

# define function that formats a label and value in boxes
def metric_card(label, value):
    st.markdown(
        f"""
        <div style="
            padding: 12px 16px;
            border-radius: 8px;
            background-color: #FFFFFF;
            border: 1px solid #E5E7EB;
        ">
            <div style="font-size: 12px; color: #6B7280;">
                {label}
            </div>
            <div style="font-size: 20px; font-weight: 600; color: {PRIMARY};">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# create 3 columns
col1, col2, col3 = st.columns(3)

# apply metric_card function

# place observations and target type in column 1
with col1:
    metric_card("Observations", f"{len(df):,}")  # calculate length of dataset (number of observations)
    st.markdown("\n")
    metric_card("Target Type", "Binary")

# place features and time span in column 2
with col2:
    metric_card("Features", df.shape[1] - 1)  # number of columns - 1 (target column)
    st.markdown("\n")
    metric_card("Time Span", "2022–2023")
    
# place target and geographic scope in column 3
with col3:
    metric_card("Target", "Wildfire Occurrence")
    st.markdown("\n")
    metric_card("Geographic Scope", "Global")

st.markdown("\n")
st.markdown("**Target Definition**")
st.markdown(
    "`occurred = 1` → wildfire detected  \n"
    "`occurred = 0` → no wildfire detected"
)


## CLASS BALANCE

st.markdown("---")
st.markdown("## Class Balance")

target_col = "occurred"
class_counts = df[target_col].value_counts().sort_index()  # value counts for classes in target column (fire/no-fire)


fig, ax = plt.subplots() # create figure
ax.bar(["No Fire", "Fire"], class_counts.values, color=[SECONDARY, ACCENT])  # bar chart of class count values, x labels, chosen colours
ax.set_ylabel("Number of Observations", color=PRIMARY)  # label y axis
ax.set_title("Class Distribution", color=PRIMARY)  # label graph
ax.grid(True, linestyle="--", alpha=0.2)  # add grid lines

st.pyplot(fig)  # plot the figure

# calculate number of wildfire occurrences as a percentage of total
fire_pct = class_counts[1] / class_counts.sum() * 100

st.markdown(f"""
> Fire and non-fire observations are nearly evenly represented in this
dataset.

- **Wildfire occurrences:** {fire_pct:.2f}%
- **Non-fire observations:** {100 - fire_pct:.2f}%

This balance reduces bias toward the majority class and allows standard
probability thresholds to be used without additional re-weighting.
""")

## FEATURE GROUPS

st.markdown("---")
st.markdown("## Feature Groups")

st.markdown("""The features used in this dataset can be grouped by their physical meaning:

- **Geographic:** Latitude, longitude  
- **Atmospheric & moisture:** Pressure, humidity, dew point, cloud cover  
- **Radiation & energy:** Solar radiation  
- **Temperature:** Mean temperature, temperature range  
- **Wind:** Speed, direction, variability  
- **Fire indices:** Fire Weather Index (FWI)

This grouping informed both feature engineering and user interface design.
""")

## FEATURE DISTRIBUTIONS

st.markdown("## Feature Distributions")

st.markdown("""
Below are selected feature distributions illustrating the range,
skewness, and variability present in the data.  

""")

st.caption("Use the tabs below to navigate between feature distributions.")

# define tab labels, plot titles, and dataframe columns
tab_labels = {
    "Temperature": {
        "title": "Mean Temperature (°C)",
        "column": "temp_mean"
    },
    "Humidity": {
        "title": "Minimum Humidity (%)",
        "column": "humidity_min"
    },
    "Wind Speed": {
        "title": "Maximum Wind Speed (km/h)",
        "column": "wind_speed_max"
    },
    "FWI": {
        "title": "Fire Weather Index",
        "column": "fire_weather_index"
    }
}

# create tabs to switch between plots
# use labels from tab_labels
tabs = st.tabs(list(tab_labels.keys()))

# for each tab and df column, create a subplot
for i, (tab_label, cfg) in enumerate(tab_labels.items()):
    with tabs[i]:
        fig, ax = plt.subplots()

        # histogram for the column
        ax.hist(
            df[cfg["column"]],
            bins=50,
            color=SECONDARY
        )

        ax.set_title(cfg["title"], color=PRIMARY) # plot title
        ax.set_ylabel("Frequency", color=PRIMARY) # y axis label
        ax.grid(True, linestyle="--", alpha=0.2)  # grid lines
        st.pyplot(fig)

st.markdown("""
Several features exhibit skewed distributions, motivating the use of
logarithmic transformations prior to scaling and dimensionality reduction.
""")


## MISSING DATA

st.markdown("---")
st.markdown("## Missing Data & Preprocessing")

# count nulls
missing_pct = df.isna().mean() * 100
missing_pct = missing_pct[missing_pct > 0]

# display whether there are nulls or not
if missing_pct.empty:
    st.success("No missing values detected in the dataset.") # green box
else:
    st.warning("Some features contain missing values:") # red box
    st.dataframe(missing_pct.round(2))

# drop-down box with feature engineering steps
with st.expander("Preprocessing steps applied"):
    st.markdown("""
    - Logarithmic transformation of skewed variables  
    - Standardization using a fitted standard scaler  
    - Dimensionality reduction using Principal Component Analysis (PCA)  
    """)

## LIMITATIONS

st.markdown("---")
st.markdown("## Dataset Limitations & Assumptions")

st.markdown("""
While this dataset provides valuable insight into wildfire risk, several
limitations should be noted:

""")

# write limitations in padded box
st.markdown(
    """
    <div style="
        background-color: #FFFFFF;
        border-left: 4px solid;
        padding: 16px 20px;
        border-radius: 6px;
        color: #374151;
    ">
    <ul style="margin: 0; padding-left: 18px;">
        <li>Human ignition sources are not explicitly modeled</li>
        <li>Fuel moisture and vegetation type are not directly observed</li>
        <li>Spatial resolution may not capture microclimate effects</li>
        <li>Historical reporting bias may affect wildfire occurrence records</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("\n")
st.markdown("These limitations should be considered when interpreting predictions.")


# page caption
st.markdown("---")
st.caption("Data Overview — Wildfire Risk Prediction Project")