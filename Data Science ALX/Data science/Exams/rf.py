import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create the dataframe with the provided data
data = {
    'team': ['Oita Trinita', 'Roasso Kumamoto', 'Montedio Yamagata', 'Tokushima Vortis', 'IK Oddevold', 'Sandvikens IF', 'SC Preussen 06 Munster', 'Hannover 96', 'Eintracht Braunschweig', '1. FC Magdeburg', 'FC Schaffhausen', 'FC Vaduz', 'Odra Opole', 'Wisla Plock', 'KS Lechia Gdansk', 'Zagłębie Lubin', 'Estoril Praia', 'Santa Clara Azores', 'De Graafschap', 'FC Volendam', 'Odds BK', 'Sarpsborg 08', 'FK Austria Wien', 'Wolfsberger AC', 'KFUM Oslo', 'Rosenborg BK', 'Kongsvinger IL Toppfotball', 'Lyn 1896 FK', 'Royal Charleroi SC', 'KAA Gent', 'SC Farense', 'Moreirense FC', 'Lillestrom SK', 'Molde FK'],
    'away_wins': [2, 0, 3, 3, 1, 2, 1, 1, 1, 3, 2, 2, 3, 0, 1, 1, 2, 4, 1, 3, 1, 2, 3, 1, 2, 3, 2, 3, 1, 3, 2, 3, 1, 3],
    'away_draws': [0, 3, 1, 0, 1, 1, 2, 1, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 1, 1, 2, 2, 1, 1, 1, 0, 1, 2, 1, 1, 2, 2, 2, 1],
    'away_losses': [3, 2, 1, 2, 3, 2, 2, 3, 4, 1, 2, 2, 1, 3, 2, 2, 1, 1, 3, 1, 2, 1, 1, 3, 1, 2, 2, 0, 3, 1, 1, 0, 2, 1],
    'home_wins': [1, 1, 3, 3, 1, 1, 2, 3, 2, 2, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 4, 3, 3, 2, 3, 3, 2, 3, 1, 3, 3, 3, 2, 4],
    'home_draws': [2, 1, 2, 2, 2, 3, 2, 1, 1, 2, 0, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 2, 1, 0, 2, 1],
    'home_losses': [2, 3, 0, 0, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 0, 1, 2, 1, 3, 1, 0, 0, 1, 1, 1, 1, 1, 0, 2, 0, 1, 2, 1, 0],
    'head_to_head_wins': [3, 2, 1, 2, 3, 1, 0, 3, 0, 3, 2, 2, 2, 3, 1, 3, 1, 3, 0, 3, 2, 3, 3, 1, 0, 4, 2, 3, 2, 3, 2, 2, 1, 3],
    'head_to_head_draws': [1, 1, 1, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'head_to_head_losses': [1, 2, 3, 2, 1, 2, 4, 1, 2, 0, 2, 2, 2, 1, 3, 1, 3, 1, 2, 0, 0, 0, 1, 3, 4, 0, 2, 1, 2, 1, 2, 2, 3, 1],
    'outcome': [1, -1, 0, 1, 1, -1, -1, 1, -1, 1, 0, 0, 1, -1, 0, 1, 1, -1, -1, 1, 0, 0, 1, -1, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1]
}

df = pd.DataFrame(data)

# Split the data into features and target
X = df.drop(columns=['team', 'outcome'])
y = df['outcome']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest model with Grid Search for hyperparameter tuning
param_grid = {
    'n_estimators': [25, 50, 100],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [5, 8, 10],
    'min_samples_leaf': [1, 3, 5]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters
best_params = grid_search.best_params_
print(f'Best parameters: {best_params}')

# Train the Random Forest model with the best parameters
best_rf_model = grid_search.best_estimator_

# Make predictions
y_pred = best_rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Loss', 'Draw', 'Win'])
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Loss', 'Draw', 'Win'], yticklabels=['Loss', 'Draw', 'Win'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature importance
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

# Print feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{f + 1}. feature {features[indices[f]]} ({importances[indices[f]]})")

# Plot the feature importances
plt.figure(figsize=(12, 6))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

# Function to predict outcomes
def predict_outcomes(new_data, model):
    new_X = new_data.drop(columns=['team'])
    predictions = model.predict(new_X)
    outcome_mapping = {1: 'Win', 0: 'Draw', -1: 'Loss'}
    predicted_outcomes = [outcome_mapping[p] for p in predictions]
    return predicted_outcomes

# Example new match data
new_match_data = {
    'team': [
        'Oita Trinita vs. Roasso Kumamoto',
        'Montedio Yamagata vs. Tokushima Vortis',
        'IK Oddevold vs. Sandvikens IF',
        'SC Preussen 06 Munster vs. Hannover 96',
        'Eintracht Braunschweig vs. 1. FC Magdeburg',
        'FC Schaffhausen vs. FC Vaduz',
        'Odra Opole vs. Wisla Plock',
        'KS Lechia Gdansk vs. Zagłębie Lubin',
        'Estoril Praia vs. Santa Clara Azores',
        'De Graafschap vs. FC Volendam',
        'Odds BK vs. Sarpsborg 08',
        'FK Austria Wien vs. Wolfsberger AC',
        'KFUM Oslo vs. Rosenborg BK',
        'Kongsvinger IL Toppfotball vs. Lyn 1896 FK',
        'Royal Charleroi SC vs. KAA Gent',
        'SC Farense vs. Moreirense FC',
        'Lillestrom SK vs. Molde FK'
    ],
    'away_wins': [2, 3, 1, 1, 1, 2, 3, 1, 2, 1, 1, 3, 1, 2, 1, 2, 1],
    'away_draws': [0, 1, 1, 2, 0, 1, 1, 2, 2, 1, 2, 1, 2, 1, 1, 2, 2],
    'away_losses': [3, 1, 3, 2, 4, 2, 1, 2, 1, 3, 2, 1, 2, 2, 3, 1, 2],
    'home_wins': [1, 3, 1, 2, 2, 3, 3, 3, 1, 1, 4, 3, 3, 2, 1, 3, 2],
    'home_draws': [2, 2, 2, 2, 1, 0, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 2],
    'home_losses': [2, 0, 2, 1, 2, 2, 1, 0, 2, 3, 0, 1, 1, 1, 2, 1, 1],
    'head_to_head_wins': [3, 1, 3, 0, 0, 2, 2, 1, 1, 0, 2, 3, 0, 2, 2, 2, 1],
    'head_to_head_draws': [1, 1, 1, 1, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1],
    'head_to_head_losses': [1, 3, 1, 4, 2, 2, 2, 3, 3, 2, 0, 1, 4, 2, 2, 2, 3]
}

# Create DataFrame for new match data
new_data_df = pd.DataFrame(new_match_data)

# Predict outcomes for the new match data
predicted_outcomes = predict_outcomes(new_data_df, best_rf_model)

# Print the predicted outcomes
for match, outcome in zip(new_data_df['team'], predicted_outcomes):
    print(f'Match: {match}, Predicted Outcome: {outcome}')
