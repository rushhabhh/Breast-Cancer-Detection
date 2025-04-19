# Breast Cancer Diagnosis Prediction

This repository contains a complete machine learning pipeline for classifying breast tumors as malignant or benign using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. The project covers data preprocessing, exploratory data analysis (EDA), model training, evaluation, and visualization, and aims to support early diagnosis through predictive modeling.

## Overview

The goal of this project is to compare multiple machine learning models for breast cancer diagnosis and evaluate them using precision-focused metrics. Given the critical nature of early cancer detection, the project emphasizes interpretability, accuracy, and model fairness.

## Project Links

- **Medium Article**: [Diagnosing Breast Cancer with Machine Learning: A Model Comparison](https://medium.com/@rushhabhh/can-ai-detect-cancer-exploring-ml-models-on-breast-tumor-data-936f63edbe55)
- **Jupyter Notebook**: Included in this repository under `breast_cancer.ipynb`

## Project Features

### 1. Data Exploration and Preprocessing
- **Dataset Overview**: WDBC dataset with 569 records and 30 numerical features.
- **Target Variable**: `Diagnosis` — encoded as 0 (Benign) and 1 (Malignant).
- **Preprocessing**: Dropped ID column, scaled features where necessary, handled slight class imbalance.

### 2. Exploratory Data Analysis (EDA)
- **Visualizations**: Count plots, correlation heatmaps, boxplots, KDE plots, PCA scatter, and pairplots.
- **Insights**: Malignant tumors show higher values for features like `radius_mean` and `concave_points_mean`.

### 3. Machine Learning Models
- **Models Used**:
  - Logistic Regression
  - Gaussian Naive Bayes (with and without scaling)
  - Support Vector Machine (SVM)
  - Decision Tree
  - Random Forest
  - XGBoost
- **Evaluation Metrics**:
  - Accuracy
  - Precision, Recall, F1-score
  - ROC-AUC and Precision-Recall curves

### 4. Model Evaluation
- All top models achieved >95% accuracy
- Random Forest and XGBoost delivered the best real-world performance
- Logistic Regression proved to be a strong, interpretable baseline

### 5. Feature Importance
- Extracted from Random Forest and XGBoost
- Top predictors: `radius_mean`, `perimeter_worst`, `concave_points_mean`

## Visualizations

- Class Distribution Plot  
- Correlation Heatmap  
- Pairplot of Informative Features  
- PCA Scatter Plot  
- ROC-AUC Curve Comparison  
- Precision-Recall Curve Comparison  
- Feature Importance Bar Charts  

## Dataset

- **Source**: [UCI Machine Learning Repository - WDBC](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Features**: 30 numerical features describing characteristics of the cell nuclei
- **Target**: Malignant (M) or Benign (B), encoded as 1 and 0

## Technologies and Tools Used

- **Libraries**: 
  - `pandas`, `numpy`, `scikit-learn`
  - `matplotlib`, `seaborn`, `plotly`
  - `xgboost`
- **Tools**:
  - Jupyter Notebook
  - Git & GitHub for version control
  - Medium for article publication

## How to Use

1. Clone this repository  
   ```bash
   git clone https://github.com/your-username/breast-cancer-diagnosis.git
