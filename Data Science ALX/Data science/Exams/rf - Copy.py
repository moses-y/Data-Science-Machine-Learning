import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

# Updated data with Elo ratings and outcomes
data = {
  'team': ['Flora Tallinn', 'Vikingur Reykjavik', 'Pyunik', 'Ordabasy', 'Servette', 'Braga', 'Vojvodina Novi Sad', 'Maribor'],
  'last_10_wins': [3, 4, 5, 6, 2, 3, 1, 4],
  'last_10_draws': [3, 2, 2, 1, 3, 2, 3, 1],
  'last_10_losses': [4, 4, 3, 3, 5, 5, 6, 5],
  'avg_goals_scored': [1.5, 2.0, 2.5, 1.8, 1.2, 1.6, 1.0, 1.4],
  'avg_goals_conceded': [2.0, 1.5, 1.0, 1.2, 2.5, 2.0, 2.2, 1.8],
  'elo_rating': [1450, 1470, 1430, 1440, 1460, 1480, 1420, 1455],  # Updated Elo ratings
  'outcome': [0, 0, 1, 1, 0, 1, 0, 1]  # Updated outcomes: 1 for win, 0 for draw
}

df = pd.DataFrame(data)

# Adjust outcome encoding
outcome_mapping = {1: 1, 0: 0}  # 1 for win, 0 for draw
df['outcome'] = df['outcome'].map(outcome_mapping)

# Prepare features and target
X = df.drop(columns=['team', 'outcome'])
y = df['outcome']

# Encode 'team' feature
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
team_encoded = encoder.fit_transform(df[['team']])
team_columns = encoder.get_feature_names_out(['team'])
X = pd.concat([X, pd.DataFrame(team_encoded, columns=team_columns)], axis=1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Draw', 'Win'], zero_division=1))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# New match data with updated Elo ratings
new_match_data = {
  'team': [
      'Flora Tallinn vs. Vikingur Reykjavik',
      'Pyunik vs. Ordabasy',
      'Servette vs. Braga',
      'Vojvodina Novi Sad vs. Maribor'
  ],
  'last_10_wins': [3, 5, 4, 3],
  'last_10_draws': [4, 3, 3, 4],
  'last_10_losses': [3, 2, 3, 3],
  'avg_goals_scored': [1.5, 2.0, 1.8, 1.6],
  'avg_goals_conceded': [1.2, 1.0, 1.5, 1.4],
  'elo_rating': [1450, 1470, 1430, 1440]  # Updated Elo ratings for new matches
}

# Create DataFrame for new match data
new_data_df = pd.DataFrame(new_match_data)

# Encode 'team' feature for new data
team_encoded_new = encoder.transform(new_data_df[['team']])
new_X = pd.concat([new_data_df.drop('team', axis=1), pd.DataFrame(team_encoded_new, columns=team_columns)], axis=1)

# Predict outcomes for the new match data
predicted_outcomes = model.predict(new_X)

# Map predictions to readable outcomes
outcome_mapping = {1: 'Win', 0: 'Draw'}
predicted_outcomes_readable = [outcome_mapping[p] for p in predicted_outcomes]

# Print the predicted outcomes
for match, outcome in zip(new_data_df['team'], predicted_outcomes_readable):
  print(f'Match: {match}, Predicted Outcome: {outcome}')