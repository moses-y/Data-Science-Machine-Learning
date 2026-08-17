# Employee Attrition Prediction - ML Pipeline Report V2 (Part A)

## 1. Introduction

This report details the revised machine learning pipeline developed to predict employee attrition using the `Part_A_Dataset.xlsx` dataset. The goal was to build and compare classification models (Logistic Regression, Random Forest, XGBoost) to determine whether an employee is likely to leave ('Attrition' = Yes) or stay ('Attrition' = No), selecting the best model based on its ability to identify employees likely to leave. This version incorporates insights from the data definitions provided within the dataset file.

## 2. Data Loading & Initial Exploration

*   **Dataset:** `Part_A_Dataset.xlsx`
*   **Data Sheet:** The primary data was loaded from the sheet named `WA_Fn-UseC_-HR-Employee-Attriti`.
*   **Data Definitions:** A second sheet named `Data Definitions` was identified, providing context for ordinal variables like `Education`, `EnvironmentSatisfaction`, `JobInvolvement`, `JobSatisfaction`, `PerformanceRating`, `RelationshipSatisfaction`, and `WorkLifeBalance`.
*   **Initial Dimensions:** The dataset contained 1470 records and 35 features.
*   **Target Variable:** `Attrition`.

## 3. Data Preparation

*   **Target Variable Conversion:** `Attrition` ('Yes'/'No') converted to numerical (1/0).
*   **Feature Dropping:**
    *   `EmployeeNumber` (Identifier).
    *   `EmployeeCount`, `StandardHours`, `Over18` (Constant value columns).
    *   **Assumption:** These columns do not contribute to predicting attrition.
*   **Resulting Data:** 1470 records, 31 columns (30 features + 1 target).
*   **Feature Identification & Preprocessing Strategy:**
    *   Based on the data definitions and column data types, features were classified:
        *   **Numerical Features (14):** Features like `Age`, `DailyRate`, `MonthlyIncome`, etc. Scaled using `StandardScaler`.
        *   **Categorical Features (16):** Includes object-type columns (`BusinessTravel`, `Department`, etc.) AND ordinal columns identified from data definitions (`Education`, `EnvironmentSatisfaction`, etc.) plus `StockOptionLevel` and `JobLevel`. All were encoded using `OneHotEncoder`.
    *   **Rationale:** Treating defined ordinal columns as categorical for One-Hot Encoding avoids imposing a potentially incorrect numerical distance between categories and is a robust approach.
*   **Train/Test Split:** 80% train (1176 records), 20% test (294 records), stratified by `Attrition`.
*   **Preprocessing Implementation:** A `ColumnTransformer` applied `StandardScaler` to numerical features and `OneHotEncoder` (with `handle_unknown='ignore'`) to all identified categorical features.
*   **Final Features:** After preprocessing, the number of features input to the models was 78.
*   **Saved Components:** The fitted `ColumnTransformer` was saved as `preprocessor.joblib`, and the original feature list as `model_columns.joblib`.

## 4. Model Training & Comparison

Three classification models were trained and evaluated on the preprocessed data. Techniques to handle the class imbalance observed in the `Attrition` variable were employed:
*   `class_weight='balanced'` for Logistic Regression and Random Forest.
*   `scale_pos_weight` calculated based on training data class distribution for XGBoost.

*   **Models Trained:**
    1.  Logistic Regression (`LogisticRegression`)
    2.  Random Forest (`RandomForestClassifier`)
    3.  XGBoost (`XGBClassifier`) - *Requires XGBoost library to be installed.*

*   **Evaluation Metrics:** Accuracy and Classification Report (including Precision, Recall, F1-score for both classes). The primary metric for comparison was the **F1-score for the positive class (Attrition=1)**, as correctly identifying employees likely to leave is often a key business objective.

*   **Results Summary:**

    | Model                | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-score (Class 1) |
    | :------------------- | :------- | :------------------ | :--------------- | :----------------- |
    | Logistic Regression  | 0.8027   | 0.43                | 0.72             | **0.5397**         |
    | Random Forest        | 0.8503   | 0.80                | 0.09             | 0.1538             |
    | XGBoost              | 0.8673   | 0.70                | 0.30             | 0.4179             |

*   **Detailed Reports:**

    *   **Logistic Regression:**
        ```
                      precision    recall  f1-score   support
           0 (No)        0.94      0.82      0.87       247
           1 (Yes)       0.43      0.72      0.54        47
        ```
    *   **Random Forest:**
        ```
                      precision    recall  f1-score   support
           0 (No)        0.85      1.00      0.92       247
           1 (Yes)       0.80      0.09      0.15        47
        ```
    *   **XGBoost:**
        ```
                      precision    recall  f1-score   support
           0 (No)        0.88      0.98      0.93       247
           1 (Yes)       0.70      0.30      0.42        47
        ```

*   **Insights:**
    *   Treating ordinal variables appropriately and addressing class imbalance significantly impacted results compared to the initial pipeline.
    *   Logistic Regression, despite lower overall accuracy, achieved the highest F1-score (and Recall) for the minority class (Attrition=Yes), suggesting it provides the best balance for identifying employees likely to leave among the models tested with these settings.
    *   Random Forest performed surprisingly poorly in recalling the minority class, even with `class_weight='balanced'`. This might warrant further hyperparameter tuning if this model were pursued.
    *   XGBoost achieved the highest accuracy but did not identify the 'Yes' cases as effectively as Logistic Regression in this configuration.

## 5. Best Model Selection & Saving

*   **Best Model:** Logistic Regression was selected as the best model based on the highest F1-score (0.5397) for the positive class (Attrition=1).
*   **Saving:** The trained Logistic Regression model object was saved as `best_attrition_model.joblib`.

## 6. Conclusion

The ML pipeline successfully incorporated data definitions, applied appropriate preprocessing, and compared three classification models (Logistic Regression, Random Forest, XGBoost) while addressing class imbalance. Logistic Regression emerged as the best-performing model for the key objective of identifying employees likely to leave (highest F1-score for Attrition=Yes). The final selected model, preprocessor, and column list have been saved for potential future use or deployment. Further improvements could involve more extensive hyperparameter tuning for all models or exploring more advanced feature engineering techniques.