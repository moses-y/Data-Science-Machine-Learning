import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset, specifying that the header is on the 3rd row (index 2)
data = pd.read_excel(r'C:\Users\moses_y\OneDrive\Desktop\ML Projects\Pipeline\Quote Pipeline.xlsx', header=2)

# Strip leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Drop columns with more than 50% missing values
threshold = len(data) * 0.5
data = data.dropna(axis=1, thresh=threshold)

# Impute missing values in remaining columns
for col in data.columns:
    if data[col].isnull().sum() > 0:
        if col == 'Quote Date':
            data[col] = data[col].fillna(method='ffill')  # Forward fill for date
        elif data[col].dtype == 'object':
            data[col] = data[col].fillna('Unknown')  # Fill missing categorical data with 'Unknown'
        else:
            data[col] = data[col].fillna(data[col].mean())  # Fill missing numerical data with mean

# Ensure zip codes are treated as strings
data['Sell-To Zip'] = data['Sell-To Zip'].astype(str)

# Data preprocessing
data['Quote Date'] = pd.to_datetime(data['Quote Date'], errors='coerce')
data = data.dropna(subset=['Quote Date'])
data['Quote Month'] = data['Quote Date'].dt.month
data['Quote Year'] = data['Quote Date'].dt.year
data['Outcome'] = data['Stage'].apply(lambda x: 1 if x == 'Won' else 0)

# Feature selection
features = ['Quote Month', 'Quote Year', 'Sell-To State', 'Salesperson', 'Item', 'Stage', 
            'Customer', 'Market Channel', 'Div Mfr Rep Name', 'Internal Account Manager', 
            'Sell-To City', 'Sell-To Zip', 'Quote Type', 'Price Group', 'Total Value', 'Probability Value']

# Drop non-feature columns
non_feature_columns = ['CustomerNo', 'Sell-To Contact', 'Ship To Name', 'Sell-to Phone No', 'Quote #', 'Primary', 
                       'QuotePipeLine.Description', 'LeadSource', 'Follow-Up Needed', 'Follow-Up Date', 
                       'Document Date', 'Expiration Date', 'No of Archived Versions', 
                       'Project Name', 'Artwork Path']

data = data.drop(columns=non_feature_columns, errors='ignore')

# Define preprocessing for categorical features
categorical_features = ['Sell-To State', 'Salesperson', 'Item', 'Stage', 'Customer', 'Market Channel', 
                        'Div Mfr Rep Name', 'Internal Account Manager', 'Sell-To City', 'Quote Type', 'Price Group', 'Sell-To Zip']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), ['Total Value', 'Probability Value'])
    ],
    remainder='passthrough'
)

# Split data into features and target
X = data[features]
y = data['Outcome']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the pipeline
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier())])
model_pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("ROC AUC Score:", roc_auc)

# Feature Importance Analysis
feature_importances = model_pipeline.named_steps['classifier'].feature_importances_
categorical_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = list(categorical_feature_names) + ['Total Value', 'Probability Value', 'Quote Month', 'Quote Year']

feature_importance_df = pd.DataFrame({
    'feature': all_feature_names,
    'importance': feature_importances
}).sort_values(by='importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df.head(20), x='importance', y='feature')
plt.title('Top 20 Feature Importances')
plt.show()
