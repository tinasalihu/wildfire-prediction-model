# Wildfire Prediction Using Machine Learning

This project develops an end-to-end machine learning pipeline to predict wildfire
occurrence using environmental data from NASA. The workflow includes exploratory
data analysis, two pipilines for feature reduction (VIF scores vs PCA), model training with XGBoost optimised
using Optuna, and deployment via a Streamlit web application for interactive
predictions.


## Dataset
The dataset was sourced from Kaggle and contains environmental and meteorological
features related to wildfire occurrence. Due to Kaggle’s licensing and size
constraints, the raw data is not included in this repository.

To reproduce the project:
- Download the dataset from Kaggle
- Place the files in the `data/` directory

Dataset source: https://www.kaggle.com/datasets/vijayaragulvr/wildfire-prediction


## Methodology
- Performed exploratory data analysis (EDA) to understand distributions,
  relationships, and data quality
- Explored two pipelines to reduce dimensionality:
    - Pipeline A: Correlation Analysis
    - Pipeline B: Principal Component Analysis (PCA)
- Trained an XGBoost classification model
- Optimised hyperparameters using Optuna
- Evaluated model performance using appropriate classification metrics


## Streamlit Application
A locally hosted Streamlit web application was built to allow users to input
environmental variables and receive wildfire occurrence predictions from the
trained model.


## Results
The final model demonstrated strong predictive performance, showing the
effectiveness of feature reduction and hyperparameter optimisation. The project
highlights the practical application of machine learning for environmental risk
prediction.
