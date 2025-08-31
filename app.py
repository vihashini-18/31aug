# streamlit_predict.py

import streamlit as st
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import StandardScaler

# ------------------- Load Models -------------------
with open("models.pkl", "rb") as f:
    models = pkl.load(f)

# ------------------- Load original dataset to get unique values -------------------
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")  # Use your training dataset here

# ------------------- Define original columns -------------------
numerical_cols = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
    'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion'
]

categorical_cols = [
    'Gender', 'OverTime', 'BusinessTravel', 'Department',
    'EducationField', 'JobRole', 'MaritalStatus'
]

# ------------------- User Input -------------------
st.title("Employee Attrition Prediction")

st.subheader("Enter Employee Data:")

input_data = {}

# Categorical inputs
st.markdown("### Categorical Inputs")
for col in categorical_cols:
    unique_values = df[col].dropna().unique().tolist()
    input_data[col] = st.selectbox(col, unique_values)

# Numerical inputs
st.markdown("### Numerical Inputs")
for col in numerical_cols:
    input_data[col] = st.number_input(col, value=0)

X_manual = pd.DataFrame([input_data])

# ------------------- Preprocess -------------------
# One-hot encode categorical features
X_manual = pd.get_dummies(X_manual)

# Align with model features (fill missing columns with 0)
for model in models.values():
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in X_manual.columns:
            X_manual[col] = 0
    X_manual = X_manual[model_features]

# Scale numeric features
scaler = StandardScaler()
X_manual[numerical_cols] = scaler.fit_transform(X_manual[numerical_cols])

# ------------------- Predict -------------------
st.subheader("Predictions:")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Logistic Regression"):
        pred = models["LogisticRegression"].predict(X_manual)[0]
        st.success(f"Prediction: {pred}")

with col2:
    if st.button("Random Forest"):
        pred = models["RandomForest"].predict(X_manual)[0]
        st.success(f"Prediction: {pred}")

with col3:
    if st.button("Gradient Boosting"):
        pred = models["GradientBoosting"].predict(X_manual)[0]
        st.success(f"Prediction: {pred}")

with col4:
    if st.button("XGBoost"):
        pred = models["XGBoost"].predict(X_manual)[0]
        st.success(f"Prediction: {pred}")
