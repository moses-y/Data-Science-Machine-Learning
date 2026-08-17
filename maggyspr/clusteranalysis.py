# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Load the data from the Excel file
data_original = pd.read_excel("Timesheet.xlsx")

# Fill missing values in the specified columns with 0
columns_to_fill = ['START TIME', 'END TIME', 'HOURS CALCULATION', 'TOTAL HOURS', 'STAFF', 'COMMENT', 'Unnamed: 12']
data_filled = data_original.copy()
data_filled[columns_to_fill] = data_filled[columns_to_fill].fillna(0)

# Convert 'TOTAL HOURS' back to numeric for the calculations
data_filled['TOTAL HOURS'] = pd.to_numeric(data_filled['TOTAL HOURS'], errors='coerce')

# Select a subset of the data for the cluster analysis
data_cluster = data_filled[['CLIENT', 'PROJECT NAME', 'ACTIVITY CODE', 'TOTAL HOURS']].astype(str)

# One-hot encode the categorical variables
encoder = OneHotEncoder(sparse=False)
data_encoded = encoder.fit_transform(data_cluster[['CLIENT', 'PROJECT NAME', 'ACTIVITY CODE']])
data_encoded = pd.DataFrame(data_encoded, columns=encoder.get_feature_names_out(['CLIENT', 'PROJECT NAME', 'ACTIVITY CODE']))

# Normalize the numerical variable
scaler = StandardScaler()
data_encoded['TOTAL HOURS'] = scaler.fit_transform(data_cluster[['TOTAL HOURS']])

# Perform the cluster analysis
kmeans = KMeans(n_clusters=5, random_state=0)
data_encoded['cluster'] = kmeans.fit_predict(data_encoded)

# Add the cluster labels back to the original data
data_cluster['cluster'] = data_encoded['cluster']

# Create bar charts for each cluster showing the top 5 'CLIENT', 'PROJECT NAME', and 'ACTIVITY CODE' categories
fig, axes = plt.subplots(15, 1, figsize=(10, 60))

for i, cluster in enumerate(data_cluster['cluster'].unique()):
    for j, var in enumerate(['CLIENT', 'PROJECT NAME', 'ACTIVITY CODE']):
        ax = axes[3*i + j]
        data_cluster[data_cluster['cluster'] == cluster][var].value_counts().head(5).plot(kind='bar', ax=ax)
        ax.set_title(f'Cluster {cluster}: Top 5 {var} Categories')
        ax.set_xlabel(var)
        ax.set_ylabel('Count')

plt.tight_layout()
plt.show()
