import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings('ignore', category=FutureWarning) # Ignore n_init warning for KMeans

# --- Constants ---
DATA_FILE = 'Part_B_Dataset.csv'
SCALER_FILE = 'scaler_part_b.joblib'
MODEL_FILE = 'clustering_model_part_b.joblib'
ELBOW_PLOT_FILE = 'elbow_curve.png'
SILHOUETTE_PLOT_FILE = 'silhouette_plot.png'
MAX_K = 10 # Maximum number of clusters to test for K-Means/Agglomerative

# --- 1. Load Data ---
print(f"--- 1. Loading Data from {DATA_FILE} ---")
try:
    df = pd.read_csv(DATA_FILE, sep=',')
    print("Dataset loaded successfully.")
    print("Initial data shape:", df.shape)
    print("Columns:", df.columns.tolist())
except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# --- 2. Data Exploration and Preprocessing ---
print("\n--- 2. Data Exploration & Preprocessing ---")
print("\nData Info:")
df.info()

print("\nMissing values check:")
print(df.isnull().sum())
# No missing values found in this dataset based on initial check.

# Select features for clustering
features = ['total_items', 'discount%', 'weekday', 'hour',
            'Food%', 'Fresh%', 'Drinks%', 'Home%', 'Beauty%', 'Health%', 'Baby%', 'Pets%']
X = df[features].copy()

# Ensure all features are numeric (already checked by info(), but good practice)
X_numeric = X.select_dtypes(include=np.number)
if X_numeric.shape[1] != len(features):
    print("Warning: Non-numeric columns detected among features. Check data types.")
    # Handle non-numeric or drop if necessary - dataset seems clean here

# Impute missing values (using median) - Added for robustness, though none detected
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_numeric)
X_imputed_df = pd.DataFrame(X_imputed, columns=features) # Use original feature names

# Scale the features
print("\nScaling features using StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed_df)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)
print("Scaling complete.")
print("Scaled Data Head:\n", X_scaled_df.head())

# --- 3. Algorithm Comparison & Evaluation (Silhouette Score) ---
print(f"\n--- 3. Evaluating Clustering Algorithms (k=2 to {MAX_K}) ---")
results = {}

# Evaluate K-Means
print("\nEvaluating K-Means...")
kmeans_scores = {}
inertia = [] # For Elbow plot
k_range = range(2, MAX_K + 1)
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled_df)
    score = silhouette_score(X_scaled_df, labels)
    kmeans_scores[k] = score
    inertia.append(kmeans.inertia_)
    print(f"  K-Means (k={k}): Silhouette Score = {score:.4f}")
results['KMeans'] = kmeans_scores

# Plot the Elbow curve (still useful for visual intuition)
plt.figure(figsize=(10, 6))
plt.plot(range(1, MAX_K + 1), [KMeans(n_clusters=1, init='k-means++', n_init=10, random_state=42).fit(X_scaled_df).inertia_] + inertia, marker='o', linestyle='--') # Add inertia for k=1
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-cluster sum of squares)')
plt.title('Elbow Method for Optimal k (K-Means)')
plt.xticks(range(1, MAX_K + 1))
plt.grid(True)
plt.savefig(ELBOW_PLOT_FILE)
print(f"Elbow curve plot saved as {ELBOW_PLOT_FILE}")
plt.close()

# Evaluate Agglomerative Clustering
print("\nEvaluating Agglomerative Clustering...")
agg_scores = {}
for k in k_range:
    agg = AgglomerativeClustering(n_clusters=k, linkage='ward') # Ward linkage is common
    labels = agg.fit_predict(X_scaled_df)
    score = silhouette_score(X_scaled_df, labels)
    agg_scores[k] = score
    print(f"  Agglomerative (k={k}): Silhouette Score = {score:.4f}")
results['Agglomerative'] = agg_scores

# Evaluate DBSCAN (Requires parameter tuning - example with common values)
# Note: Silhouette score is less ideal if DBSCAN finds noise points (-1)
# We might need to exclude noise points for calculation or use other metrics.
# For simplicity here, we'll try a couple of `eps` values.
print("\nEvaluating DBSCAN...")
dbscan_scores = {}
# Heuristic for eps: Check distance to k-th nearest neighbor (e.g., k=2*dims)
# from sklearn.neighbors import NearestNeighbors
# k_neighbors = 2 * X_scaled_df.shape[1]
# nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(X_scaled_df)
# distances, indices = nbrs.kneighbors(X_scaled_df)
# distance_desc = sorted(distances[:, k_neighbors-1], reverse=True)
# plt.plot(list(range(1,len(distance_desc)+1)), distance_desc) # Look for 'elbow' in k-distance plot
# plt.show() # Requires manual inspection - let's try common values instead

for eps_val in [0.5, 1.0, 1.5, 2.0]: # Example eps values
    dbscan = DBSCAN(eps=eps_val, min_samples=5) # min_samples often default or 2*dims
    labels = dbscan.fit_predict(X_scaled_df)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"  DBSCAN (eps={eps_val}): Found {n_clusters} clusters and {n_noise} noise points.")
    if n_clusters > 1 and len(set(labels)) > 1: # Need at least 2 clusters (excluding noise) for silhouette
        # Calculate silhouette score excluding noise points
        valid_indices = np.where(labels != -1)[0]
        if len(valid_indices) > 1 and len(set(labels[valid_indices])) > 1:
             score = silhouette_score(X_scaled_df.iloc[valid_indices], labels[valid_indices])
             dbscan_scores[f'eps={eps_val}'] = score
             print(f"    Silhouette Score (excluding noise): {score:.4f}")
        else:
             print("    Not enough valid points/clusters to calculate Silhouette Score.")
             dbscan_scores[f'eps={eps_val}'] = -1 # Indicate failure
    else:
        print("    Not enough clusters found to calculate Silhouette Score.")
        dbscan_scores[f'eps={eps_val}'] = -1 # Indicate failure
results['DBSCAN'] = dbscan_scores


# --- 4. Select Best Model ---
print("\n--- 4. Selecting Best Model based on Silhouette Score ---")
best_algo = None
best_k = None
best_score = -1 # Silhouette score ranges from -1 to 1
best_params = {}

# Find best K-Means
kmeans_best_k = max(results['KMeans'], key=results['KMeans'].get)
kmeans_best_score = results['KMeans'][kmeans_best_k]
print(f"Best K-Means: k={kmeans_best_k}, Score={kmeans_best_score:.4f}")
if kmeans_best_score > best_score:
    best_score = kmeans_best_score
    best_algo = 'KMeans'
    best_k = kmeans_best_k
    best_params = {'n_clusters': best_k, 'init': 'k-means++', 'n_init': 10, 'random_state': 42}

# Find best Agglomerative
agg_best_k = max(results['Agglomerative'], key=results['Agglomerative'].get)
agg_best_score = results['Agglomerative'][agg_best_k]
print(f"Best Agglomerative: k={agg_best_k}, Score={agg_best_score:.4f}")
if agg_best_score > best_score:
    best_score = agg_best_score
    best_algo = 'Agglomerative'
    best_k = agg_best_k
    best_params = {'n_clusters': best_k, 'linkage': 'ward'}

# Find best DBSCAN (if any valid scores)
dbscan_valid_scores = {k: v for k, v in results['DBSCAN'].items() if v > -1}
if dbscan_valid_scores:
    dbscan_best_param_str = max(dbscan_valid_scores, key=dbscan_valid_scores.get)
    dbscan_best_score = dbscan_valid_scores[dbscan_best_param_str]
    print(f"Best DBSCAN: Params='{dbscan_best_param_str}', Score={dbscan_best_score:.4f}")
    if dbscan_best_score > best_score:
        # Note: Choosing DBSCAN might be complex due to noise handling & parameter sensitivity
        # For this example, we'll stick with K-Means/Agglomerative if they perform well.
        # If DBSCAN is significantly better, manual review is often needed.
        print("Considering DBSCAN (Manual review recommended due to noise/params)")
        # To actually select DBSCAN:
        # best_score = dbscan_best_score
        # best_algo = 'DBSCAN'
        # best_k = None # DBSCAN doesn't pre-specify k
        # eps_val = float(dbscan_best_param_str.split('=')[1]) # Extract eps
        # best_params = {'eps': eps_val, 'min_samples': 5}
else:
    print("No valid Silhouette Scores found for DBSCAN with tested parameters.")


print(f"\nSelected Best Algorithm: {best_algo}")
if best_k:
    print(f"Selected Optimal k: {best_k}")
print(f"Best Silhouette Score: {best_score:.4f}")
print(f"Parameters: {best_params}")

# --- 5. Train Final Model & Assign Clusters ---
print("\n--- 5. Training Final Model and Assigning Clusters ---")

if best_algo == 'KMeans':
    final_model = KMeans(**best_params)
elif best_algo == 'Agglomerative':
    final_model = AgglomerativeClustering(**best_params)
elif best_algo == 'DBSCAN':
     final_model = DBSCAN(**best_params)
else:
    print("Error: No suitable model selected.")
    exit()

final_labels = final_model.fit_predict(X_scaled_df)

# Add cluster labels back to the original (imputed but not scaled) data
df_clustered = X_imputed_df.copy()
df_clustered['cluster'] = final_labels
df_clustered['customer'] = df['customer'] # Add customer ID back

print(f"\nData clustered into {len(set(final_labels)) - (1 if -1 in final_labels else 0)} clusters (excluding noise if any).")
print("Cluster distribution:")
print(df_clustered['cluster'].value_counts())

# --- 6. Cluster Profiling ---
print(f"\n--- 6. Cluster Profiles (Mean Values for {best_algo}) ---")
# Exclude noise points (-1) if DBSCAN was chosen
if best_algo == 'DBSCAN' and -1 in df_clustered['cluster'].unique():
    cluster_profiles = df_clustered[df_clustered['cluster'] != -1].groupby('cluster')[features].mean()
    print("Note: Profiles exclude noise points (cluster -1).")
else:
    cluster_profiles = df_clustered.groupby('cluster')[features].mean()

print(cluster_profiles)

# --- 7. Generate Silhouette Plot for Final Model ---
print(f"\n--- 7. Generating Silhouette Plot for {best_algo} (k={best_k or 'N/A'}) ---")
try:
    from yellowbrick.cluster import SilhouetteVisualizer

    if best_algo != 'DBSCAN': # SilhouetteVisualizer works best with models that have n_clusters
        visualizer = SilhouetteVisualizer(final_model, colors='yellowbrick')
        visualizer.fit(X_scaled_df)
        visualizer.finalize() # Finalize before saving
        plt.savefig(SILHOUETTE_PLOT_FILE)
        print(f"Silhouette plot saved as {SILHOUETTE_PLOT_FILE}")
        plt.close()
    else:
        print("Silhouette plot generation skipped for DBSCAN (visualizer expects n_clusters).")

except ImportError:
    print("Yellowbrick library not found. Skipping Silhouette plot generation.")
    print("Install it using: pip install yellowbrick")


# --- 8. Persist Model and Scaler ---
print("\n--- 8. Saving Scaler and Final Model ---")
joblib.dump(scaler, SCALER_FILE)
print(f"Scaler saved to {SCALER_FILE}")
joblib.dump(final_model, MODEL_FILE)
print(f"Final clustering model ({best_algo}) saved to {MODEL_FILE}")


print("\n--- Analysis Complete ---")
print(f"Best algorithm selected: {best_algo}")
print(f"Scaler saved: {SCALER_FILE}")
print(f"Model saved: {MODEL_FILE}")
print("Review plots and cluster profiles for insights.")