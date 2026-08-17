import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load the data
data = pd.read_csv(r"C:\Users\moses_y\OneDrive\Desktop\ML Projects\football\cleaned_football_data.csv")

# Encode the categorical variables
le = LabelEncoder()
data['Div'] = le.fit_transform(data['Div'])
data['HomeTeam'] = le.fit_transform(data['HomeTeam'])
data['AwayTeam'] = le.fit_transform(data['AwayTeam'])
data['Referee'] = le.fit_transform(data['Referee'].astype(str))

# Predictors
X = data.drop(['FullTimeResult', 'HalfTimeResult', 'Date'], axis=1)

# Targets
y_half = le.fit_transform(data['HalfTimeResult'])
y_full = le.fit_transform(data['FullTimeResult'])

# Split the data into a training set and a test set for halftime and fulltime
X_train_half, X_test_half, y_train_half, y_test_half = train_test_split(X, y_half, test_size=0.25, random_state=42)
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X, y_full, test_size=0.25, random_state=42)

# Initialize the XGBoost models
xgb_half = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_full = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Train the models
xgb_half.fit(X_train_half, y_train_half)
xgb_full.fit(X_train_full, y_train_full)

# Make predictions
xgb_half_predictions = xgb_half.predict(X_test_half)
xgb_full_predictions = xgb_full.predict(X_test_full)

# Calculate the accuracy of the models
xgb_half_accuracy = accuracy_score(y_test_half, xgb_half_predictions)
xgb_full_accuracy = accuracy_score(y_test_full, xgb_full_predictions)

print('XGBoost Halftime Accuracy:', xgb_half_accuracy)
print('XGBoost Fulltime Accuracy:', xgb_full_accuracy)
