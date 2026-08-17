# Presentation of Solutions: IPA Data Science Challenge

## Introduction

My name is **Moses Yebei**, and this document presents my solutions to the **IPA Data Science Challenge**. The challenge involved analyzing data from a randomized control trial (RCT) conducted in Ghana to evaluate the impact of two financial education programs, **Aflatoun** and **Honest Money Box (HMB)**, on youth savings behavior. The dataset used for this analysis is `ghana_youth_savings.csv`.

This report outlines my approach, methodology, and findings for the five tasks provided in the challenge. The solutions are designed to be clear, actionable, and aligned with best practices in data science.

---

## Task 1: Baseline Summary Statistics

### Objective:
To create a table of baseline summary statistics for control observations, including:
- Means and standard deviations for key variables.
- Differences in means between the control group and the treatment groups (Aflatoun and HMB).

### Methodology:
1. **Data Cleaning**:
   - Filtered the dataset to include only control observations for baseline statistics.
   - Ensured all relevant columns were numeric and handled missing values appropriately.

2. **Calculations**:
   - Computed the mean and standard deviation for key variables in the control group.
   - Calculated the differences in means between the control group and the treatment groups (Aflatoun and HMB).

3. **Output**:
   - A table summarizing the baseline statistics, including:
     - Control group mean and standard deviation.
     - Differences in means for Aflatoun and HMB groups.

### Key Findings:
- The baseline characteristics were generally balanced across groups, indicating successful randomization.
- Minor differences were observed in variables such as **age** and **baseline savings amount**, but these were not substantial.

---

## Task 2: Visualization of Savings Behavior at Endline

### Objective:
To visualize the differences in savings behavior at school at endline between the control and treatment groups. Specifically:
1. The fraction of students who saved at school (`end_saveschl`).
2. The average amount saved at school (`end_saveschlamt`).

### Methodology:
1. **Data Preparation**:
   - Converted the `end_saveschl` column (Yes/No) to numeric values (1/0).
   - Cleaned the dataset by ensuring relevant columns were numeric and dropping rows with missing values.

2. **Aggregation**:
   - Grouped the data by treatment group (0 = control, 1 = treatment).
   - Calculated the mean fraction of students who saved at school and the average amount saved.

3. **Visualization**:
   - Created two bar plots:
     - **Fraction of Students Who Saved at School**.
     - **Average Amount Saved at School**.
   - Added data labels and saved the plots as high-resolution PNG files.

### Key Findings:
- The treatment group had a higher fraction of students who saved at school (9.92%) compared to the control group (2.76%).
- The treatment group also saved a higher average amount (0.6174) compared to the control group (0.1649).
- These results suggest that the intervention positively influenced savings behavior.

---

## Task 3: Regression Analysis

### Objective:
To estimate the causal impact of the Aflatoun and HMB programs on:
1. Whether the student had money saved at endline (`end_save`).
2. The amount of money saved at endline (`end_saveamt`).

### Methodology:
1. **Regression Models**:
   - Used Ordinary Least Squares (OLS) regression to estimate the impact of the programs.
   - Controlled for stratification variables (e.g., region, class size) and baseline characteristics.

2. **Handling Missing Data**:
   - Coded missing baseline values as `0` and included missingness indicators in the models.

3. **Robustness**:
   - Used robust standard errors to account for heteroskedasticity.

### Key Findings:
- **Impact on `end_save` (Binary Outcome)**:
  - Aflatoun had a small but statistically significant positive effect on the likelihood of saving at endline.
  - HMB also had a significant positive effect, larger than Aflatoun.
- **Impact on `end_saveamt` (Continuous Outcome)**:
  - The treatment groups showed higher savings amounts at endline, but the effect sizes were small.
- The R-squared values for both models were low, indicating that other unobserved factors may influence savings behavior.

---

## Task 4: Machine Learning Model for Program Participation

### Objective:
To build a machine learning model to predict program participation (take-up) using the dataset.

### Methodology:
1. **Data Preprocessing**:
   - Selected relevant features for predicting program participation.
   - Split the data into training and testing sets (80/20 split).

2. **Model Selection**:
   - Used a `RandomForestClassifier` for its robustness and ability to handle non-linear relationships.
   - Performed hyperparameter tuning using `GridSearchCV` to optimize the model.

3. **Evaluation**:
   - Evaluated the model using metrics such as precision, recall, F1-score, and AUC.

### Key Findings:
- The model achieved high accuracy (97%) but struggled with class imbalance, as the minority class (participants) was underrepresented.
- The AUC score (0.7468) indicated moderate discriminatory power.
- Recommendations:
  - Address class imbalance using techniques like SMOTE or class weighting.
  - Explore alternative models such as Gradient Boosting or Logistic Regression.

---

## Task 5: Recommendations for Monitoring System

### Objective:
To provide recommendations for setting up a monitoring system for savings clubs to track program outcomes effectively.

### Recommendations:
1. **Data Collection**:
   - Use digital tools (e.g., mobile apps) to collect real-time data on savings behavior.
   - Standardize data collection forms to ensure consistency.

2. **Data Storage**:
   - Store data securely in a cloud-based database with access controls.
   - Ensure compliance with data privacy regulations.

3. **ETL Automation**:
   - Automate data extraction, transformation, and loading (ETL) processes to reduce manual errors.

4. **Monitoring and Reporting**:
   - Develop dashboards to visualize key metrics (e.g., savings rates, amounts saved).
   - Generate automated reports for stakeholders.

5. **Scalability**:
   - Design the system to handle increasing data volumes as the program expands.
   - Use modular architecture to add new features easily.

---

## Conclusion

The IPA Data Science Challenge provided an excellent opportunity to apply data science techniques to a real-world problem. The analysis demonstrated the positive impact of the Aflatoun and HMB programs on youth savings behavior and highlighted areas for improvement in program implementation and monitoring.

Thank you for this opportunity, and I look forward to discussing these findings further.