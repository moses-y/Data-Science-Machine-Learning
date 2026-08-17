# Imports
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC  
from sklearn.neural_network import MLPClassifier
import pandas as pd

# Load and preprocess data
data = pd.read_csv(r"C:\Users\moses_y\OneDrive\Desktop\ML Projects\football\cleaned_football_data.csv") 

le = LabelEncoder()
data['Div'] = le.fit_transform(data['Div'])
data['HomeTeam'] = le.fit_transform(data['HomeTeam'])
data['AwayTeam'] = le.fit_transform(data['AwayTeam'])
data['Referee'] = le.fit_transform(data['Referee'].astype(str))

X = data.drop(['FullTimeResult', 'HalfTimeResult', 'Date'], axis=1)
y_half = le.fit_transform(data['HalfTimeResult'])
y_full = le.fit_transform(data['FullTimeResult'])  
y = le.fit_transform(data['FullTimeResult'])

# XGBoost modeling
X_train_half, X_test_half, y_train_half, y_test_half = train_test_split(X, y_half, test_size=0.25, random_state=42)
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X, y_full, test_size=0.25, random_state=42)

xgb_half = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss') 
xgb_full = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
 
xgb_half.fit(X_train_half, y_train_half)
xgb_full.fit(X_train_full, y_train_full) 

xgb_half_pred = xgb_half.predict(X_test_half)
xgb_full_pred = xgb_full.predict(X_test_full)

xgb_half_acc = accuracy_score(y_test_half, xgb_half_pred)
xgb_full_acc = accuracy_score(y_test_full, xgb_full_pred)

print(xgb_half_acc)
print(xgb_full_acc)

# Additional models with cross-validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Additional models
rf = RandomForestClassifier(n_jobs=-1)
svm = SVC(probability=True)  
nn = MLPClassifier(max_iter=1000) 

# Define parameter grids
rf_params = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 20]
}

svm_params = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'] 
}

nn_params = {
   'hidden_layer_sizes':[(10,), (50,), (100,)],  
   'activation':['relu', 'tanh']
}

# Grid Search Cross-Validation 

rf_cv = GridSearchCV(rf, rf_params, cv=5) 
svm_cv = GridSearchCV(svm, svm_params, cv=5)
nn_cv = GridSearchCV(nn, nn_params, cv=5)

rf_cv.fit(X_train, y_train)
svm_cv.fit(X_train, y_train)  
nn_cv.fit(X_train, y_train)

# Evaluation
y_probs = rf_cv.predict_proba(X_test)
rf_auc = roc_auc_score(y_test, y_probs, multi_class="ovr")

y_probs = svm_cv.predict_proba(X_test)
svm_auc = roc_auc_score(y_test, y_probs, multi_class="ovr")

y_probs = nn_cv.predict_proba(X_test)
nn_auc = roc_auc_score(y_test, y_probs, multi_class="ovr")

# Print results
# Print XGBoost accuracy scores
print("XGBoost Halftime Accuracy:", xgb_half_acc)
print("XGBoost Fulltime Accuracy:", xgb_full_acc)

# Print Random Forest accuracy and AUC
print("RF AUC:", rf_auc) 

# Print SVM accuracy and AUC
print("SVM AUC:", svm_auc)

# Print Neural Network accuracy and AUC 
print("NN AUC:", nn_auc)

results = {
    "XGBoost": {
        "Halftime Accuracy": xgb_half_acc,
        "Fulltime Accuracy": xgb_full_acc
    },
    "Random Forest": {
        "AUC": rf_auc
    },
    "SVM": {
        "AUC": svm_auc
    },
    "Neural Network": {
        "AUC": nn_auc
    }
}