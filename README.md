# 🏨 Hotel Booking Analysis Project

## 📌 Project Overview
This project focuses on analyzing a hotel booking dataset containing around **8000 records**.  
For training and testing purposes, a random sample of **2000 rows** is taken.  
The goal is to explore the data, visualize trends, and build machine learning models to understand booking patterns.

---

## 🔧 Workflow
1. **Data Preparation**
   - Load dataset  
   - Take random 2000-row sample for training  
   - Clean missing values  
   - Drop unnecessary columns (`Booking_ID`, `reservation_date`)  

2. **Exploratory Data Analysis (EDA)**
   - **Histograms** → for numerical features (adults, children, etc.)  
   - **Countplots** → for categorical features (room type, meal, etc.)  
   - **Pie charts** → to show proportions (like booking status)  
   - **Boxplots** → to detect outliers  
   - **Heatmap** → to check correlations  

3. **Model Building**
   - Train baseline models:  
     - Logistic Regression  
     - Random Forest  
     - XGBoost  
   - Evaluate with Accuracy, Precision, Recall, F1-score, ROC-AUC  

4. **Deployment**
   - Build an interactive dashboard with **Streamlit**  
   - Allow filtering, visualization, and predictions  

---

## 📊 Tools & Libraries
- **Python 3.x**  
- **NumPy, Pandas** → data handling  
- **Matplotlib, Seaborn** → visualization  
- **Scikit-learn** → ML models & evaluation  
- **XGBoost** → advanced boosting model  
- **Streamlit** → deployment & dashboard  

---


   git clone <your-repo-url>
   cd hotel-booking
