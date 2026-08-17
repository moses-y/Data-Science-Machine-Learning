# Model Performance Report

## Model: Model with 200 estimators
- RMSE: 0.8548755139938428
- R^2: 0.9921092447761178

### Feature Importances
- Temperature: 0.001945941028637184
- Rainfall: 0.9790749560098201
- Fertilizer_Usage: 0.01663340768465428
- Pesticide_Usage: 0.0010760135074802213
- Irrigation: 0.0002363488749276526
- Region_North: 0.00011280663953094573
- Region_South: 0.00011423616038888293
- Region_West: 0.0001872589118828026
- Soil_Type_Loamy: 0.00014915233580973174
- Soil_Type_Sandy: 0.0001506277048292028
- Crop_Variety_Variety B: 0.00019041579331969166
- Crop_Variety_Variety C: 0.00012883534871930213

---
## Summary: markdown_report
# Model Performance Report

## Model: Model with 200 estimators
- RMSE: 0.8548755139938428
- R^2: 0.9921092447761178

### Feature Importances
- Temperature: 0.001945941028637184
- Rainfall: 0.9790749560098201
- Fertilizer_Usage: 0.01663340768465428
- Pesticide_Usage: 0.0010760135074802213
- Irrigation: 0.0002363488749276526
- Region_North: 0.00011280663953094573
- Region_South: 0.00011423616038888293
- Region_West: 0.0001872589118828026
- Soil_Type_Loamy: 0.00014915233580973174
- Soil_Type_Sandy: 0.0001506277048292028
- Crop_Variety_Variety B: 0.00019041579331969166
- Crop_Variety_Variety C: 0.00012883534871930213

---
## Summary: markdown_report
# Model Performance Report

## Model: Model with 200 estimators
- RMSE: 0.8548755139938428
- R^2: 0.9921092447761178

### Feature Importances
- Temperature: 0.001945941028637184
- Rainfall: 0.9790749560098201
- Fertilizer_Usage: 0.01663340768465428
- Pesticide_Usage: 0.0010760135074802213
- Irrigation: 0.0002363488749276526
- Region_North: 0.00011280663953094573
- Region_South: 0.00011423616038888293
- Region_West: 0.0001872589118828026
- Soil_Type_Loamy: 0.00014915233580973174
- Soil_Type_Sandy: 0.0001506277048292028
- Crop_Variety_Variety B: 0.00019041579331969166
- Crop_Variety_Variety C: 0.00012883534871930213

---
## Summary: markdown_report
# Model Performance Report

## Model: Model with 200 estimators
- RMSE: 0.8548755139938428
- R^2: 0.9921092447761178

### Feature Importances
- Temperature: 0.001945941028637184
- Rainfall: 0.9790749560098201
- Fertilizer_Usage: 0.01663340768465428
- Pesticide_Usage: 0.0010760135074802213
- Irrigation: 0.0002363488749276526
- Region_North: 0.00011280663953094573
- Region_South: 0.00011423616038888293
- Region_West: 0.0001872589118828026
- Soil_Type_Loamy: 0.00014915233580973174
- Soil_Type_Sandy: 0.0001506277048292028

---
## Summary: markdown_report
# Model Performance Report

# Model: Model with 200 estimators
- RMSE: 0.8548755139938428
- R^2: 0.9921092447761178

### Feature Importances

- Temperature: 0.001945941028637184
- Rainfall: 0.9790749560098201

- Fertilizer_Usage: 0.01663340768465428
- Pesticide_Usage: 0.0010760135074802213
- Irrigation: 0.0002363488749276526
- Region_North: 0.00011280663953094573
- Region_South: 0.00011423616038888293

- Region_West: 0.0001872589118828026
- Soil_Type_Loamy: 0.00014915233580973174
- Soil_Type_Sandy: 0.0001506277048292028

---
## Summary: markdown_report
# Model Performance Report

# Model: Model with 200 estimators
- RMSE: 0.854875513

## Summary: C:\Users\moses_y\OneDrive\Desktop\ML Projects\alx\Data science\Machine Learning\Random Forests\Introduction_to_random_forests_examples.ipynb
<div align="center" style=" font-size: 80%; text-align: center; margin: 0 auto">
<img src="https://raw.githubusercontent.com/Explore-AI/Pictures/master/Python-Notebook-Banners/Examples.png"  style="display: block; margin-left: auto; margin-right: auto;";/>
</div># Examples: An introduction to random forests 
© ExploreAI AcademyIn this notebook, we look at a specific ensemble learning technique, the random forest. This algorithm is the combination of multiple decision trees. We'll examine how we can implement these models in Python using the `sklearn` library and also how random forests contribute to understanding the importance of predictive variables within a dataset.
## Learning objectives

* Understand how random forests differ from decision trees.
* Understand the training process of random forests and how we can use them to make predictions.
* Know how to build a random forest regression model.
* Understand how to use random forests to assess predictive variable importance.
## 1. Introduction to random forests

### Overfitting in decision trees

Overfitting is a risk when working with decision trees – it is easy for the tree to become too complex, and thus fit details of the individual data points rather than the overall properties of the distributions they are drawn from. 

This issue can be addressed by using **random forests**.### What is ensemble learning?

Ensemble learning in machine learning is the practice of **combining multiple models** to try and achieve higher overall model performance. 

In general, ensembles consist of multiple **heterogeneous or homogeneous** models trained on the same dataset. 

Each of these models is used to make predictions using input (either exactly the same or samples out of the same data), then  **aggregated** across all models in some way (e.g. by taking the mean or having a weighted mean) to produce the final output. 

The  **`random forest`** is an example of a commonly used ensemble model.### What is a random forest?

A random forest is a powerful non-parametric algorithm and an **ensemble** method **built on decision trees**, meaning that it relies on aggregating the results of an ensemble of decision trees. 

The ensembled trees are **randomised** and the output is the **aggregated prediction** of the individual trees.

*The mean prediction is used for a regression problem while classification problems use the mode of the ensembled trees as opposed to the mean.*## 2. How do random forests work?

### Fitting the data:

Keep in mind that `N` refers to the **number of observations** (rows) in the training dataset, and `p` is the **number of predictor variables** (columns). The following is the typical algorithm for a random forest:

1. **Bootstrapping**: Drawing *with replacement* from the training dataset, randomly sampled `N` observations.
<br>

2. Use the `N` observations to **grow a random forest tree** as follows:<br>
_<br>
At each node:<br>
i. Select a random subset, `m`, of predictor variables, where $m<\sqrt{p}$.<br>
ii. Pick the subset of predictor variables that are larger than $m<\sqrt{p}$.<br>
b. Iterate through the subset of predictor variables, and select the subset that is larger than $m<\sqrt{p}$.<br>
c. Iterate through the subset of predictor variables, and select the subset that is larger than $m<\sqrt{p}$.<br>
d. Iterate through the subset of predictor variables, and select the subset that is larger than $m<\sqrt{p}$.<br>
e. Iterate through the subset of predictor variables, and select the subset that is larger than $m<\sqrt{p}$.<br>
f. Iterate through the subset of predictor variables, and select the subset that is larger than $m<\sqrt{p}$.<br>
g. Iterate through the subset of predictor variables, and select the subset that is larger than $m<\sqrt{p}$.<br>
h. Iterate through the subset of predictor variables, and select the subset that is larger than $m<\sqrt{p}$.<br>
i. Iterate through the subset of predictor variables, and select the subset that is larger than $m<\sqrt{p}$.<br>
j. Iterate through the subset of predictor variables, and select the subset that is larger

## Summary: C:\Users\moses_y\OneDrive\Desktop\ML Projects\alx\Data science\Machine Learning\Random Forests\Solving_a_regression_problem_examples.ipynb
<div align="center" style=" font-size: 80%; text-align: center; margin: 0 auto">
<img src="https://raw.githubusercontent.com/Explore-AI/Pictures/master/Python-Notebook-Banners/Examples.png"  style="display: block; margin-left: auto; margin-right: auto;";/>
</div># Examples: Solving a regression problem
 
© ExploreAI AcademyIn this train, we look at how to experiment with various modeling methods to solve a regression problem.## Learning objectives

By the end of this train, you should be able to:
* Load and prepare a dataset.
* Train and evaluate linear, non-linear and ensemble regression models.## 1. Introduction

We wish to examine and understand how socio-economic and environmental factors contribute to the rate of deforestation. We have been given a dataset that captures this information. Our goal to to model this relationship through regression analysis. 

In the sections that follow, we explore the process of training and evaluating various regression models on a similar dataset. Through the model experimentation process, we are able to compare the performance of various models, guiding the selection of the best model for predicting our target variable. ## 2. Dataset and libraries

We begin by setting up our working environment by importing the necessary `Python` libraries that we will use throughout the notebook

Then, we load and inspect the dataset to get familiar with its structure and contents.import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv('https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Data/regression_sprint/enviro_indicators.csv', index_col=0)
data.head()## 3. Data exploration

Here, we will perform a preliminary exploration of our dataset to get a better understanding of the data we are working with.

This initial exploration of our data, will help uncover underlying patterns and relationships that can inform our choice of models.# Displaying data information
data.info()`data.info()` gives us more information about our columns including their data types and the count of non-null values.

It seems we do not have any null values in our dataset. Also, all the features in the dataset are numeric.Let's now create a pairwise plot to visualise and understand the relationships among the variables. Using `sns.pairplot()`, we can generate scatter plots for each pair of variables in our dataset, providing a comprehensive overview of how each variable interacts with the others. This can help us identify patterns that may influence our modeling decisions.sns.pairplot(data)
plt.show()Only `forest coverage` and `biodiversity_index` show a linear-like relationship with the target variable `deforestation_rate`. This observation suggests the model is not suitable for the target variable, as it is not linear.

## Summary: C:\Users\moses_y\OneDrive\Desktop\ML Projects\alx\Data science\Machine Learning\Random Forests\The_random_forest_exercise.ipynb
<div align="center" style=" font-size: 80%; text-align: center; margin: 0 auto">
<img src="https://raw.githubusercontent.com/Explore-AI/Pictures/master/Python-Notebook-Banners/Exercise.png"  style="display: block; margin-left: auto; margin-right: auto;";/>
</div>

# Exercise: The random forest
© ExploreAI Academy

In this exercise, we build, evaluate and compare random forest regression models.## Learning objectives

By the end of this train, you should be able to:
* Build a random forest regression model in Python.
* Experiment with different number of trees.
* Evaluate feature importance using a random forest. ## Exercises

In this exercise, we will be using the `Crop_yield` dataset that contains various factors that could influence the yield of a particular crop across different regions.### Import libraries and datasetimport numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score# Load dataset
df= pd.read_csv("https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Data/Python/Crop_yield.csv")
df.head(5)### Preparing the dataset

In the code below, we prepare our dataset for modeling by encoding categorical variables to convert them to a numeric format.# Dummy Variable Encoding for categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)### Exercise 1

Create a function named `train_rf_model` to train and evaluate a random forest regression model on the encoded dataset. 

The function should take in 3 parameters:
- DataFrame containing the encoded features
- A string containing the name of the target variable
- The number of estimators for the random forest 

It then returns: 
- The trained model object 
- The RMSE and R<sup>2</sup> scores of the model's performance on the test set. # Your solution here...

def train_rf_model(df, target_var, n_estimators):
    """
    Trains a random forest regression model on the given dataset.

    Parameters:
    - df: DataFrame containing the encoded features.
    - target_var: String containing the name of the target variable.
    - n_estimators: The number of estimators for the random forest.

    Returns:
    - model: The trained random forest model object.
    - rmse: The root mean squared error of the model on the test set.
    - r2: The R^2 score of the model on the test set.
    """

    # Splitting the dataset into training and testing sets
    X = df.drop(columns=[target_var])
    y = df[target_var]
     # Split the dataset into training and testing sets
    X = df.drop(columns=[target_var])
    y = df[target_var]
    # Split the dataset into training and testing sets
    X = df.drop(columns=[target_var])
    y = df[target_var]
    # Split the dataset into training and testing sets
   X = df.drop(columns=[target_var])
   y = df[target_var]
   # Split the dataset into training and testing sets
   X = df.drop(columns=[target_var])
   y = df[target_var]
   # Split the dataset into training and testing sets
   X = df.drop(columns=[target_var])
   y = df[target_var]
   # Split the dataset into training and testing sets
   X = df.drop(columns=[target_var])
   y = df[target_var]
   # Split the dataset into training and testing sets
   X = df.drop(columns=[target_var])
   y = df[target_var]
   # Split the dataset into training and testing sets
   X = df.drop(columns=[target

## Summary: C:\Users\moses_y\OneDrive\Desktop\ML Projects\alx\Data science\Machine Learning\Random Forests\the_random_forest_student_version.ipynb
<div align="center" style=" font-size: 80%; text-align: center; margin: 0 auto">
<img src="https://raw.githubusercontent.com/Explore-AI/Pictures/master/Python-Notebook-Banners/Code_challenge.png"  style="display: block; margin-left: auto; margin-right: auto;";/>
</div># Code challenge: Random forest regression
© ExploreAI AcademyIn this code challenge, we'll test our knowledge of how to create an ensemble model known as a random forest. We will train this new model using the world population data. ⚠️ **Note that this code challenge is graded and will contribute to your overall marks for this module. Submit this notebook for grading. Note that the names of the functions are different in this notebook. Transfer the code in your notebook to this submission notebook**

### Instructions

- **Do not add or remove cells in this notebook. Do not edit or remove the `### START FUNCTION` or `### END FUNCTION` comments. Do not add any code outside of the functions you are required to edit. Doing any of this will lead to a mark of 0%!**

- Answer the questions according to the specifications provided.

- Use the given cell in each question to see if your function matches the expected outputs.

- Do not hard-code answers to the questions.

- The use of StackOverflow, Google, and other online tools is permitted. However, copying a fellow student's code is not permissible and is considered a breach of the Honour code. Doing this will result in a mark of 0%.### Importsimport numpy as np
import pandas as pd
from numpy import array
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_errorpopulation_df = pd.read_csv('https://raw.githubusercontent.com/Explore-AI/Public-Data/master/AnalyseProject/world_population.csv', index_col='Country Code')
meta_df = pd.read_csv('https://raw.githubusercontent.com/Explore-AI/Public-Data/master/AnalyseProject/metadata.csv', index_col='Country Code')population_df.head()meta_df.head()### Question 1The world population data spans from 1960 to 2017. We'd like to build a predictive model that can give us the best guess at what the world population in a given year was. However, we want to compute this estimate for only _countries within a given income group_. 

First, however, we need to organise our data such that the sklearn's `RandomForestRegressor` class can train on our data. To do this, we will write a function that takes as input an income group and returns a 2-d numpy array that contains the year and the measured population.

_**Function Specifications:**_
* Should take a `str` argument, called `income_group_name` as input and return a numpy `array` type as output.
* Set the default argument of `income_group_name` to equal `'Low income'`.
* If the specified value of `income_group_name` does not match the given input, return False.
* If the specified value of `income_group_name` does not match the given input, return True.
* If the specified value of `income_group_name` does not match the given input, return False.
**Function Parameters:**
_**Input: The income group name.
_**Output: The year and measured population.
_**Input: The year and measured population.
_**Input: The year and measured population.
_**Input: The year and measured population.
_**Input: The year and measured population.
_**Input: The year and measured population.
_**Input: The year and measured population.
_**Input: The year and measured population.
_**Input: The year and measured population.
_**Input: The year and measured population.
_**Input: The year and measured population.
_**Input: The year and measured population.
_**Input: The year and measured population.
_**Input: The year and measured population.
_**Input: The year and measured population.
_**Input: The year and measured population.
_**Input: The year and measured population.
_**Input: The year and measured population.
_**Input: The year and measured population.
_**Input: The year and measured population.
_**Input: The year and measured population.


