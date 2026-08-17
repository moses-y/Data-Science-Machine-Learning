# Employee Attrition Prediction - ML Pipeline Report (Part A)

## 1. Introduction

This report details the machine learning pipeline developed to predict employee attrition based on a dataset provided for an organization (`Part_A_Dataset.xlsx`). The goal is to build a classification model that determines whether an employee is likely to leave the organization ('Attrition' = Yes) or stay ('Attrition' = No).

## 2. Data Loading & Initial Exploration

*   **Dataset:** `Part_A_Dataset.xlsx`
*   **Initial Dimensions:** The dataset initially contained 1470 records (employees) and 35 features (attributes).
*   **Target Variable:** The target variable for prediction is `Attrition`.

## 3. Data Preparation

This phase involved cleaning and transforming the data to make it suitable for model training.

*   **Target Variable Conversion:** The `Attrition` column, originally containing 'Yes' and 'No', was converted into a numerical format: 'Yes' mapped to 1 and 'No' mapped to 0.
*   **Feature Dropping:**
    *   `EmployeeNumber`: Dropped as it's a unique identifier for each employee and provides no predictive value.
    *   `EmployeeCount`, `StandardHours`, `Over18`: These columns were found to have only one unique value across all employees (constant columns) and were therefore dropped as they offer no discriminatory information for the model.
    *   **Assumption:** It was assumed that these identifier and constant columns do not contribute to predicting attrition.
*   **Resulting Data:** After dropping these columns, the dataset had 1470 records and 31 columns (30 features + 1 target).
*   **Feature Identification:** The remaining 30 features were categorized into:
    *   Numerical Features: 23 (e.g., `Age`, `DailyRate`, `MonthlyIncome`)
    *   Categorical Features: 7 (e.g., `BusinessTravel`, `Department`, `Gender`, `JobRole`)
*   **Train/Test Split:** The dataset was split into training (80%) and testing (20%) sets. Stratification based on the `Attrition` target variable was used to ensure that the proportion of employees leaving vs. staying was similar in both the training and testing sets. This resulted in:
    *   Training set: 1176 records
    *   Testing set: 294 records
*   **Preprocessing Steps:** A `ColumnTransformer` was used to apply different preprocessing steps to numerical and categorical features:
    *   **Numerical Features:** Scaled using `StandardScaler` (subtracting the mean and dividing by the standard deviation, fitted only on the training data).
    *   **Categorical Features:** Converted into numerical format using `OneHotEncoder`. This creates new binary (0/1) columns for each category within a feature. The `handle_unknown='ignore'` parameter was used to prevent errors if unseen categories appear in the test set.
*   **Final Features:** After preprocessing (scaling and encoding), the number of features used for model training became 51.

## 4. Model Training

*   **Model Choice:** Logistic Regression was selected as the initial baseline classification model due to its simplicity and interpretability.
*   **Training:** The Logistic Regression model was trained using the preprocessed training data (`X_train_processed`) and the corresponding target labels (`y_train`).

## 5. Model Evaluation

The trained model's performance was evaluated on the unseen preprocessed test data (`X_test_processed`).

*   **Metrics:**
    *   **Accuracy:** 86.05% (The overall percentage of correct predictions).
    *   **Confusion Matrix:**
        ```
        [[237  10]  <- Correctly predicted 'No', Incorrectly predicted 'Yes'
         [ 31  16]]  <- Incorrectly predicted 'No', Correctly predicted 'Yes'
        ```
    *   **Classification Report:**
        ```
                      precision    recall  f1-score   support
           0 (No)        0.88      0.96      0.92       247
           1 (Yes)       0.62      0.34      0.44        47

            accuracy                           0.86       294
           macro avg       0.75      0.65      0.68       294
        weighted avg       0.84      0.86      0.84       294
        ```
*   **Insights & Assumptions:**
    *   The model achieves a good overall accuracy (86.05%).
    *   **Insight:** Performance is significantly better for the majority class (Attrition=No, Class 0) than the minority class (Attrition=Yes, Class 1). The recall for Class 1 (0.34) is particularly low, meaning the model correctly identifies only 34% of the employees who actually leave.
    *   **Assumption:** Accuracy was considered a primary metric for this baseline. However, the imbalance in precision and recall between classes suggests that for practical applications (e.g., identifying employees at risk of leaving), metrics like Recall for the 'Yes' class might be more important.
    *   **Potential Improvement:** The performance difference between classes indicates potential class imbalance in the dataset. Future work could involve techniques to address this, such as:
        *   Resampling techniques (e.g., SMOTE for oversampling the minority class, or undersampling the majority class).
        *   Using models with built-in class weighting.
        *   Trying different, potentially more complex, classification algorithms (e.g., Random Forest, Gradient Boosting).
        *   Further feature engineering or selection.

## 6. Model Saving

For persistence and potential deployment, the following components were saved using `joblib`:

*   **Trained Model:** `attrition_model.joblib` (The fitted Logistic Regression model).
*   **Preprocessor:** `scaler.joblib` (The fitted `ColumnTransformer` containing the `StandardScaler` and `OneHotEncoder`). This is crucial for processing new data consistently before prediction.
*   **Model Columns:** `model_columns.joblib` (A list of the original feature column names expected by the preprocessor).

## 7. Conclusion

A baseline machine learning pipeline was successfully implemented to predict employee attrition. The Logistic Regression model achieved an accuracy of 86.05% on the test set. While the overall accuracy is reasonable, the evaluation revealed a significant performance gap between predicting employees who stay versus those who leave, likely due to class imbalance. The model and associated preprocessing steps have been saved, providing a foundation for future improvements or deployment. Further work should focus on improving the prediction of the minority class (Attrition=Yes) by addressing the class imbalance or exploring more advanced modeling techniques.