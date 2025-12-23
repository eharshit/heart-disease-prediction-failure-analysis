# Heart Disease Prediction & Failure Analysis

<div align="left">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EB5B3C?style=for-the-badge&logo=xgboost&logoColor=white)
![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)
![SHAP](https://img.shields.io/badge/Explainable_AI-SHAP-success?style=for-the-badge)

</div>

## Project Overview
Heart failure is a high-risk condition where delayed detection can lead to severe outcomes. This project implements an end-to-end machine learning system for early risk prediction using routinely available clinical data.

The system generates patient-level risk predictions and pairs them with SHAP-based explanations to ensure transparency. In parallel, a Power BI dashboard analyzes survival trends across demographic and clinical variables, enabling population-level insights alongside individual predictions. The focus is on interpretability, failure awareness, and real-world usability rather than accuracy in isolation.

## Problem Statement
- **Late Risk Detection**: Heart failure often progresses without obvious symptoms. This project focuses on early risk identification using routinely collected clinical features.
- **Lack of Model Trust**: Many clinical ML systems act as black boxes. This system integrates SHAP-based explanations to show feature-level contributions behind each prediction.
- **Complex Feature Interactions**: Traditional clinical risk scores rely on linear assumptions and fixed thresholds. This project leverages tree-based ensemble models to capture non-linear interactions among clinical features, improving risk estimation beyond rule-based methods.
- **Isolated Decision-Making**: Individual predictions lack population context. The Power BI dashboard enables analysis of survival trends and risk factors across demographic and clinical groups.

## Datasets

### Prediction Dataset
*   **Source**: Integrated Heart Disease Dataset.
*   **Records**: **918** patient records.
*   **Purpose**: Used to train the heart disease prediction model.
*   **Target**: Presence of heart disease (0/1).

### Failure Analysis Dataset
*   **Source**: Heart Failure Clinical Records Dataset.
*   **Records**: **299** patient records.
*   **Purpose**: Used for mortality and survival analysis (Power BI).
*   **Target**: `DEATH_EVENT`.


## Machine Learning Pipeline
The project follows a structured ML workflow implemented in the training notebook:

1. **Data Preparation**: Dataset validation, missing-value checks, and basic outlier handling.

2. **Preprocessing**: Categorical encoding and feature scaling using sklearn pipelines.
3. **Train–Test Split**: Data split for model evaluation.
4. **Model Training**: Ensemble models (Random Forest, XGBoost) trained for risk prediction.
5. **Evaluation**: Performance measured using classification metrics and confusion matrix.
6. **Explainability**: SHAP used for global and individual prediction explanations.
7. **Deployment**: Trained models and preprocessors saved for Streamlit inference.
