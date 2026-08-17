# ml_pipeline.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import numpy as np

# --- Configuration ---
FILE_PATH = 'Part_A_Dataset.xlsx'
TARGET_COLUMN = 'Attrition'
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_SAVE_PATH = 'attrition_model.joblib'
SCALER_SAVE_PATH = 'scaler.joblib' # Will save the preprocessor here
COLUMNS_SAVE_PATH = 'model_columns.joblib'

# --- Load Data ---
print("1. Loading data...")
try:
    df = pd.read_excel(FILE_PATH)
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

# --- Initial EDA & Feature Dropping ---
print("\n2. Initial EDA & Feature Dropping...")
# Convert target variable
df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(lambda x: 1 if x == 'Yes' else 0)
print(f"Target variable '{TARGET_COLUMN}' converted to numeric.")

# Identify columns to drop
columns_to_drop = []
potential_constant_cols = ['EmployeeCount', 'StandardHours', 'Over18']
identifier_col = 'EmployeeNumber'

if identifier_col in df.columns:
    columns_to_drop.append(identifier_col)
    print(f" - Dropping identifier column: {identifier_col}")

for col in potential_constant_cols:
    if col in df.columns:
        if df[col].nunique() == 1:
            columns_to_drop.append(col)
            print(f" - Dropping constant column: {col}")
        else:
             print(f" - Column '{col}' is not constant, keeping.")
    else:
        print(f" - Column '{col}' not found in dataframe.")


df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
print(f"Columns dropped. New shape: {df.shape}")


# --- Preprocessing ---
print("\n3. Preprocessing...")
# Separate features and target
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

print(f" - Numerical features: {len(numerical_features)}")
print(f" - Categorical features: {len(categorical_features)}")

# Create preprocessing pipelines for numerical and categorical features
# Note: OneHotEncoder handles unknown categories in test set if they appear
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features) # Use sparse_output=False for dense array
    ],
    remainder='passthrough' # Keep other columns (if any) - should not be needed here
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
print(f" - Data split into train/test sets. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Apply preprocessing (Fit on train, transform train and test)
# Fit the preprocessor and transform training data
X_train_processed = preprocessor.fit_transform(X_train)
# Transform test data
X_test_processed = preprocessor.transform(X_test)

# Get feature names after one-hot encoding
# Need to handle potential sparse matrix output if sparse_output=True was used
feature_names_out = preprocessor.get_feature_names_out()
print(f" - Preprocessing complete. Number of features after encoding: {len(feature_names_out)}")

# Save the fitted preprocessor (contains scaler and encoder)
joblib.dump(preprocessor, SCALER_SAVE_PATH) # Saving the whole preprocessor is often easier
print(f" - Preprocessor (scaler & encoder) saved to {SCALER_SAVE_PATH}")
# Save the columns list *before* preprocessing, as the preprocessor handles the transformation
model_columns = X_train.columns.tolist()
joblib.dump(model_columns, COLUMNS_SAVE_PATH)
print(f" - Original columns list saved to {COLUMNS_SAVE_PATH}")


# --- Model Training ---
print("\n4. Model Training (Logistic Regression)...")
model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000) # Increased max_iter for convergence
model.fit(X_train_processed, y_train)
print(" - Model training complete.")

# --- Model Evaluation ---
print("\n5. Model Evaluation...")
y_pred = model.predict(X_test_processed)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f" - Accuracy: {accuracy:.4f}")
print(" - Confusion Matrix:")
print(conf_matrix)
print(" - Classification Report:")
print(class_report)

# --- Model Saving ---
print("\n6. Model Saving...")
joblib.dump(model, MODEL_SAVE_PATH)
print(f" - Trained model saved to {MODEL_SAVE_PATH}")

print("\n--- ML Pipeline Complete ---")