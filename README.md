# Wildfire Prediction Using Machine Learning

This project develops an end-to-end machine learning pipeline to predict wildfire
occurrence using environmental and meteorological data. It combines exploratory
data analysis, feature reduction, model training, hyperparameter optimisation,
and deployment through an interactive Streamlit web application.

---

## Project Overview

Wildfires pose a significant threat to ecosystems, human life, and property.
This project leverages historical wildfire and environmental data to predict
the likelihood of wildfire ignition for a given location and time. The workflow
covers the full machine learning lifecycle:

1. **Exploratory Data Analysis (EDA)** – Visualising spatial, temporal, and
   environmental patterns to understand the data.
2. **Feature Reduction** – Reducing dimensionality using either Variance Inflation
   Factor (VIF) analysis or Principal Component Analysis (PCA).
3. **Model Training** – Using an XGBoost classifier with optimized hyperparameters
   through Optuna.
4. **Model Evaluation** – Performance assessed via precision, recall, F1-score,
   accuracy, and confusion matrix analysis.
5. **Deployment** – Interactive predictions using a Streamlit web application
   where users can input conditions to estimate wildfire risk.

---

## Dataset

The dataset was sourced from Kaggle and contains environmental, meteorological,
and spatial features related to wildfire occurrence. Features include:

- **Geographic:** Latitude, longitude  
- **Atmospheric & Moisture:** Pressure, humidity, dew point, cloud cover  
- **Radiation & Energy:** Solar radiation  
- **Temperature:** Mean temperature, temperature range  
- **Wind:** Speed, direction, variability  
- **Fire Indices:** Fire Weather Index (FWI)  

**Target Variable:**  
`occurred = 1` indicates a wildfire occurred; `occurred = 0` indicates no wildfire.

**Data Source:** [Kaggle: Global Wildfire Dataset](https://www.kaggle.com/datasets/vijayaragulvr/wildfire-prediction)

---

## Methodology

1. **Data Cleaning & Preprocessing**  
   - Checked for missing values and handled appropriately  
   - Logarithmic transformation of skewed variables  
   - Standardization using a fitted `StandardScaler`  
   - Dimensionality reduction using PCA for certain pipelines  

2. **Exploratory Data Analysis (EDA)**  
   - Visualized distributions, class balance, and spatial patterns  
   - Analysed day vs night wildfire occurrence and environmental drivers  

3. **Feature Reduction Pipelines**  
   - **Pipeline A:** Correlation analysis and VIF to remove multicollinearity  
   - **Pipeline B:** PCA to reduce dimensions while preserving variance  

4. **Model Training & Optimization**  
   - XGBoost classifier trained on reduced feature set  
   - Hyperparameters optimized using Optuna for maximum recall  
   - Ensured reproducibility using fixed random seeds and samplers  

5. **Evaluation Metrics**  
   - Precision, recall, F1-score, and overall accuracy  
   - Confusion matrix to assess trade-offs between false positives and negatives  

6. **Deployment**  
   - Streamlit app allows interactive predictions with sidebar input controls  
   - Visualizations include feature distributions, spatial maps, and model insights  
   - Users can download the full dataset for offline analysis  

---

## Results

- **Model Performance:** High recall (sensitivity) prioritised to detect wildfires
  early, with some trade-off in precision. Overall accuracy balanced across classes.
- **Key Insights:**  
  - Nighttime observations show higher wildfire occurrence rates  
  - Environmental variables alone are insufficient to predict fire; risk emerges
    from combinations of factors  
  - Wind variability has limited predictive power on its own  
- **Interactive Application:** Users can explore data patterns and predict wildfire
  risk for custom conditions.

---

## How to Run

1. Create and activate a conda environment:

```bash
conda create -n wildfire python=3.10
conda activate wildfire

2. Install dependencies:

pip install -r requirements.txt

3. Run the Streamlit app:
cd app
streamlit run Home.py

4. Navigate through the app using the sidebar.

## Notes

This project is intended for educational and analytical purposes only.

Wildfire predictions should not replace official alerts or professional judgment.

The app and models are reproducible; changing the random seed or dataset may
slightly alter results.
