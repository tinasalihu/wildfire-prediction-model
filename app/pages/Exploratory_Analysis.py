import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# set theme colours
PRIMARY = "#2E3B4E"   # dark slate for main plot elements
SECONDARY = "#4B627D" # muted blue
ACCENT = "#bf603d"    # accent for highlights
BG_COLOR = "#F5F7FA"  # background
GRID_COLOR = "#E2E8F0" # subtle light gray grid

# set global colours for matplotlib graphs
plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": PRIMARY,
    "axes.labelcolor": PRIMARY,
    "xtick.color": PRIMARY,
    "ytick.color": PRIMARY,
    "text.color": PRIMARY,
    "grid.color": GRID_COLOR,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,  # subtle grid
    "font.size": 10
})

# page configuration
st.set_page_config(
    page_title="Data Exploration",
    page_icon="🗺️",
    layout="centered"
)

# page title
st.title("Exploratory Data Analysis")

# page summary
st.markdown(
    "<p style='color:#6B7280; max-width: 800px;'>"
    "This page explores spatial, temporal, and environmental patterns in the "
    "wildfire dataset. The objective is to validate domain-consistent trends "
    "and understand why multivariate modeling is required."
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

## SPATIAL DISTRIBUTION

st.markdown("## Spatial Distribution of Observations")
st.markdown("The map below shows the geographic distribution of fire occurrences in the dataset.")

# there are many observations to plot so take a small, random sample
sample_df = df.sample(n=min(5000, len(df)), random_state=42)
# select rows where a fire has occurred
fire_rows = sample_df[sample_df['occurred'] == 1]
# plot the wildfire occurrences on a map
st.map(fire_rows[["lat", "lon"]])

st.caption("Note: Plot shows a random sample of 5000 points in the dataset.")

st.markdown("""
Observation points cover a broad geographic region, allowing the model to
learn spatially varying wildfire risk patterns.
""")

## DAY VS NIGHT

st.markdown("## Day vs Night Wildfire Occurrence")

# calculate wildfire occurrence rate by time of day
daynight_rates = (
    df.groupby("daynight_N")["occurred"]
    .mean()
    .reset_index()
)

# map binary values to labels
daynight_rates["Time of Day"] = daynight_rates["daynight_N"].map(
    {0: "Night", 1: "Day"}
)

# create two columns to put plot and observation side-by-side
col1, col2 = st.columns([2, 1])

# bar chart in column 1
with col1:
    fig, ax = plt.subplots()
    ax.bar(
        daynight_rates["Time of Day"],
        daynight_rates["occurred"],
        color=[SECONDARY, ACCENT]
    )
    ax.set_ylabel("Wildfire Occurrence Rate", color=PRIMARY)
    ax.set_title("Wildfire Occurrence Rate by Time of Day", color=PRIMARY)
    ax.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig)

# observations in column 2
with col2:
    st.markdown("\n")
    st.markdown("\n")
    st.markdown(
        "<p style='color:#374151;'>"
        "<strong>Key observation:</strong> Nighttime conditions exhibit "
        "substantially higher wildfire occurrence rates.<br><br>"
        "This may reflect delayed detection, reporting artifacts, or "
        "temporal aggregation effects rather than true ignition timing."
        "</p>",
        unsafe_allow_html=True
    )


## ENVIRONMENTAL CONDITIONS
st.markdown("## Environmental Conditions During Fire Events")

st.markdown("""
The following plots compare key environmental variables between fire and
no-fire observations.
""")

st.caption("Compare distributions using the tabs to assess overlap between fire and no-fire conditions.")

env_features = {
    "Mean Temperature (°C)": "temp_mean",
    "Minimum Humidity (%)": "humidity_min",
    "Maximum Wind Speed (km/h)": "wind_speed_max",
}

# create tabs for each environmental feature
tab_labels = ["Temperature", "Humidity", "Wind Speed"]
tabs = st.tabs(tab_labels)

# generate the graphs for each tab and column
for i, (title, col) in enumerate(env_features.items()):
    with tabs[i]:
        fig, ax = plt.subplots()
        # histogram for no-fire observations
        ax.hist(
            df[df["occurred"] == 0][col],
            bins=40,
            alpha=0.8,
            label="No Fire",
            color=SECONDARY
        )
        # histogram for fire observations
        ax.hist(
            df[df["occurred"] == 1][col],
            bins=40,
            alpha=0.8,
            label="Fire",
            color=ACCENT
        )
        # formatting
        ax.set_title(title, color=PRIMARY)
        ax.set_ylabel("Frequency", color=PRIMARY)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig)



st.markdown("""
Distributions of individual environmental variables show substantial overlap
between fire and no-fire observations. This indicates that wildfire occurrence
cannot be reliably explained by single variables in isolation.

Instead, wildfire risk appears to emerge from combinations of atmospheric,
meteorological, and geographic conditions, motivating the use of multivariate
models.
""")

## WIND VARIABILITY
st.markdown("## Wind Variability Analysis")

# create side-by-side box plots for wind direction for fire and no-fire observations
fig, ax = plt.subplots()
ax.boxplot(
    [
        df[df["occurred"] == 0]["wind_direction_std"],
        df[df["occurred"] == 1]["wind_direction_std"]
    ],
    # formatting
    labels=["No Fire", "Fire"],
    patch_artist=True,
    boxprops=dict(facecolor=SECONDARY, color=PRIMARY),
    medianprops=dict(color=ACCENT)
)
ax.set_ylabel("Wind Direction Variability (°)", color=PRIMARY)
ax.set_title("Wind Direction Variability by Fire Occurrence", color=PRIMARY)
ax.grid(True, linestyle="--", alpha=0.3)

st.pyplot(fig)

st.markdown(
    "<p style='color:#374151;'>"
    "Wind direction variability shows only marginal differences between fire "
    "and no-fire observations, reinforcing the limited explanatory power of "
    "single variables in isolation."
    "</p>",
    unsafe_allow_html=True
)


## SUMMARY
st.markdown("## Summary Insights")

# write insights in padded box
st.markdown(
    """
    <div style="
        background-color: #FFFFFF;
        border-left: 4px solid #bf603d;
        padding: 16px 20px;
        border-radius: 6px;
        color: #374151;
    ">
    <ul style="margin: 0; padding-left: 18px;">
        <li>Nighttime observations show substantially higher wildfire occurrence rates.</li>
        <li>Environmental variables exhibit strong overlap between fire and no-fire cases.</li>
        <li>Wind direction variability shows minimal separation between outcomes.</li>
        <li><strong>Wildfire risk emerges from combined conditions rather than single drivers.</strong></li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# page caption
st.markdown("---")
st.caption("Exploratory Data Analysis — Wildfire Risk Prediction Project")
