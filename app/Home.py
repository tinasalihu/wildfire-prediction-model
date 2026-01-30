import streamlit as st

# page configuration
st.set_page_config(
    page_title="Wildfire Risk Predictor",
    page_icon="🔥",
    layout="centered"
)

# set theme colours
PRIMARY = "#2E3B4E"
ACCENT = "#bf603d"
SECONDARY = "#4B627D"

# page title
st.markdown(
    f"<h1 style='color:{PRIMARY};'>Wildfire Risk Prediction System</h1>",
    unsafe_allow_html=True
)

# image/banner at the top
st.image("app/assets/wildfire_banner.jpg", use_container_width=True)

# introduction
st.markdown(f"""

Welcome to the **Wildfire Risk Prediction System**, a tool designed to
estimate the probability of wildfire ignition using historical **meteorological, atmospheric, and geographic data**.

This application is intended for **educational and analytical purposes**.

""")

# navigation
st.markdown(f"<h3 style='color:{ACCENT}; border-bottom:2px solid {ACCENT};'>Getting Started</h3>", unsafe_allow_html=True)

st.markdown("\n\n")
st.markdown("""
Use the navigation menu on the left to explore the following sections:

- **Data Overview:** Learn about the dataset, its features, and key statistics  
- **Exploration & Spatial Patterns:** Visualize wildfire patterns across space and time  
- **Prediction Model:** Enter environmental conditions and estimate wildfire risk  

""")

# tips/guide for users
st.markdown(f"<h3 style='color:{ACCENT}; border-bottom:2px solid {ACCENT};'>Tips for Users</h3>", unsafe_allow_html=True)

st.markdown("\n\n")
st.markdown("""
- All predictions are based solely on historical data and environmental features  
- Nighttime observations are particularly important in this dataset  
- Combine multiple features to understand risk; no single variable predicts fire occurrence perfectly  
- Hover over charts for detailed values when available  
""")

# page caption
st.markdown("---")
st.caption("Home — Wildfire Risk Prediction Project")
