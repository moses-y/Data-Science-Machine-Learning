# ml_pipeline_v2.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline # Keep Pipeline if needed later, but ColumnTransformer is key here
import joblib
import numpy as np
import warnings

# Try importing XGBoost, warn if not found
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("XGBoost library not found. XGBoost model will be skipped.")

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn') # Suppress OneHotEncoder warnings if needed

# --- Configuration ---
FILE_PATH = 'Part_A_Dataset.xlsx'
DATA_SHEET_NAME = 'WA_Fn-UseC_-HR-Employee-Attriti' # Explicitly use the data sheet name
TARGET_COLUMN = 'Attrition'
TEST_SIZE = 0.2
RANDOM_STATE = 42
# Define models to train
MODELS = {
    'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced'), # Added class_weight
    'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced') # Added class_weight
}
if XGB_AVAILABLE:
    # XGBoost uses scale_pos_weight for imbalance
    MODELS['XGBoost'] = XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss')

BEST_MODEL_SAVE_PATH = 'best_attrition_model.joblib'
PREPROCESSOR_SAVE_PATH = 'preprocessor.joblib' # Renamed from scaler.joblib
COLUMNS_SAVE_PATH = 'model_columns.joblib'

# Columns identified as ordinal from the 'Data Definitions' sheet
ORDINAL_COLS = [
    'Education', 'EnvironmentSatisfaction', 'JobInvolvement',
    'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction',
    'WorkLifeBalance'
]
# Other known categorical columns (object type)
OBJECT_COLS = [
    'BusinessTravel', 'Department', 'EducationField',
    'Gender', 'JobRole', 'MaritalStatus', 'OverTime'
]

# --- Load Data ---
print("1. Loading data...")
try:
    # Load the specific data sheet
    df = pd.read_excel(FILE_PATH, sheet_name=DATA_SHEET_NAME)
    print(f"Data loaded successfully from sheet '{DATA_SHEET_NAME}'. Shape: {df.shape}")
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

# Identify columns to drop (Identifiers and Constants)
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

# Update feature lists based on definitions and dtypes
all_categorical_features = []
numerical_features = []

for col in X.columns:
    if col in ORDINAL_COLS or col in OBJECT_COLS:
        all_categorical_features.append(col)
    elif pd.api.types.is_numeric_dtype(X[col]):
         # Ensure StockOptionLevel (often treated as categorical/ordinal) is handled correctly
         if col == 'StockOptionLevel':
             all_categorical_features.append(col)
         # Ensure JobLevel is handled correctly (often ordinal)
         elif col == 'JobLevel':
              all_categorical_features.append(col)
         else:
            numerical_features.append(col)
    else:
        # Fallback for any unexpected types - treat as categorical for safety
        all_categorical_features.append(col)
        print(f"Warning: Column '{col}' has unexpected type {X[col].dtype}, treating as categorical.")


print(f" - Numerical features identified: {len(numerical_features)}")
# print(f"   {numerical_features}") # Uncomment to see list
print(f" - Categorical features identified (incl. ordinal): {len(all_categorical_features)}")
# print(f"   {all_categorical_features}") # Uncomment to see list


# Create preprocessing pipelines
# OneHotEncoder for ALL identified categorical features (ordinal included)
# StandardScaler for numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), all_categorical_features)
    ],
    remainder='passthrough' # Should not have remainders if all columns are classified
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
print(f" - Data split into train/test sets. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Apply preprocessing (Fit on train, transform train and test)
print(" - Applying preprocessing (fitting on train data)...")
X_train_processed = preprocessor.fit_transform(X_train)
print(" - Transforming test data...")
X_test_processed = preprocessor.transform(X_test)

# Get feature names after one-hot encoding
try:
    feature_names_out = preprocessor.get_feature_names_out()
    print(f" - Preprocessing complete. Number of features after encoding: {len(feature_names_out)}")
except Exception as e:
    print(f"Could not get feature names: {e}")
    print(f" - Preprocessing complete. Shape of processed train data: {X_train_processed.shape}")


# Save the fitted preprocessor
joblib.dump(preprocessor, PREPROCESSOR_SAVE_PATH)
print(f" - Preprocessor saved to {PREPROCESSOR_SAVE_PATH}")
# Save the columns list *before* preprocessing
model_columns = X_train.columns.tolist()
joblib.dump(model_columns, COLUMNS_SAVE_PATH)
print(f" - Original columns list saved to {COLUMNS_SAVE_PATH}")

# Adjust XGBoost scale_pos_weight for imbalance if available
if XGB_AVAILABLE:
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    MODELS['XGBoost'] = XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
    print(f" - Calculated scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")


# --- Model Training & Evaluation ---
print("\n4. Model Training & Evaluation...")
results = {}
best_f1 = -1
best_model_name = None
best_model_object = None

for name, model in MODELS.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train_processed, y_train)
    print(f"--- Evaluating {name} ---")
    y_pred = model.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)
    # Calculate F1 score specifically for the positive class (Attrition=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    results[name] = {'accuracy': accuracy, 'f1_score_pos': f1, 'report': class_report, 'confusion_matrix': conf_matrix}

    print(f" - Accuracy: {accuracy:.4f}")
    print(f" - F1 Score (Class 1): {f1:.4f}")
    print(" - Confusion Matrix:")
    print(conf_matrix)
    print(" - Classification Report:")
    print(class_report)

    # Check if this model is the best based on F1 score for class 1
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model_object = model

print("\n--- Model Comparison ---")
print(f"{'Model':<20} | {'Accuracy':<10} | {'F1 Score (Class 1)':<18}")
print("-" * 55)
for name, metrics in results.items():
    print(f"{name:<20} | {metrics['accuracy']:.4f}{'':<3} | {metrics['f1_score_pos']:.4f}")

# --- Best Model Saving ---
print("\n5. Best Model Saving...")
if best_model_object:
    joblib.dump(best_model_object, BEST_MODEL_SAVE_PATH)
    print(f" - Best model ({best_model_name} with F1 Score {best_f1:.4f}) saved to {BEST_MODEL_SAVE_PATH}")
else:
    print(" - No models were trained successfully.")


print("\n--- ML Pipeline V2 Complete ---")