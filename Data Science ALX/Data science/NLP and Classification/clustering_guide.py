#!/usr/bin/env python
# coding: utf-8

# ## Understanding Clustering: A Comprehensive Guide to K-Means and Hierarchical Clustering
# 
# Clustering is a fundamental technique in unsupervised machine learning that involves grouping data points into clusters based on their similarities. In this blog, we will delve into the concepts of K-Means and Hierarchical Clustering, explore their applications using a practical dataset, and understand various metrics to evaluate their performance. By the end, you'll have a solid grasp of how to implement and interpret these clustering techniques.

# ### Introduction to Clustering
# 
# Clustering helps in identifying patterns and structures within data without any prior labels. It is widely used in customer segmentation, image segmentation, anomaly detection, and more. We will focus on two popular clustering methods: K-Means and Hierarchical Clustering.

# ### The Dataset
# 
# We will use a dataset containing information about mall customers, including their gender, age, annual income, and spending score. This data is ideal for demonstrating clustering techniques aimed at segmenting customers based on their annual income and spending scores.
# 
# ```python
# import pandas as pd
# 
# # Load the dataset
# data = pd.read_csv('https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Data/unsupervised_sprint/mall_customers.csv', index_col=0)
# data.head()
# ```
# 

# ### Descriptive Statistics
# 
# Before diving into clustering, it's essential to understand the dataset's characteristics. Let's print the descriptive statistics.
# 
# ```python
# print(data.describe())
# ```
# **Insight**: The `Spending_Score` ranges from 1 to 100, confirming the variety of customer spending behaviors in the dataset.

# ### Number of Observations
# 
# To understand the scale of our data, let's find the total number of observations.
# 
# ```python
# print(f'Total number of observations: {data.shape[0]}')
# ```
# **Answer**: There are 200 observations in the dataset.

# ### Handling Null Values
# 
# Missing values can disrupt the clustering process. Let's check for null values and handle them if necessary.
# 
# ```python
# print(data.isnull().sum())
# ```
# **Insight**: The dataset contains no null values, so we can proceed without any imputation.

# ### Scatter Plot Visualization
# 
# To visualize the relationship between annual income and spending score, we generate a scatter plot.
# 
# ```python
# import matplotlib.pyplot as plt
# 
# plt.figure(figsize=(10, 6))
# plt.scatter(data['Annual_Income_(k$)'], data['Spending_Score'], c='blue', edgecolors='w', s=100)
# plt.title('Relationship between Annual Income and Spending Score')
# plt.xlabel('Annual Income (k$)')
# plt.ylabel('Spending Score')
# plt.grid(True)
# plt.show()
# ```
# **Insight**: The scatter plot shows potential clusters forming, particularly in the center of the plot.

# ### Scaling Data
# 
# Scaling is crucial for clustering algorithms to ensure that features contribute equally to the distance calculations. Let's understand why the provided code returns an error and correct it.
# 
# ```python
# from sklearn.preprocessing import MinMaxScaler
# 
# scaler = MinMaxScaler()
# try:
#     X_scaled = scaler.fit_transform(data)
# except ValueError as e:
#     print(e)
# ```
# **Error**: The MinMaxScaler cannot handle non-numeric data, such as the `Gender` column.

# ### Feature Selection and Scaling
# 
# We focus on `Annual_Income_(k$)` and `Spending_Score` for clustering.
# 
# ```python
# X = data[['Annual_Income_(k$)', 'Spending_Score']]
# X_scaled = scaler.fit_transform(X)
# ```
# 

# ### Determining the Optimal Number of Clusters
# 
# Various metrics help determine the optimal number of clusters. Here, we examine those that are typically used, such as within-cluster variation, between-cluster variation, and the CH index.
# 
# #### Elbow Method
# 
# The Elbow Method helps in identifying the optimal number of clusters by plotting the within-cluster sum of squares (WCSS) against the number of clusters.
# 
# ```python
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# 
# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, random_state=42)
#     kmeans.fit(X_scaled)
#     wcss.append(kmeans.inertia_)
# 
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 11), wcss, marker='o')
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()
# ```
# 

# ### Evaluating Silhouette Score
# 
# Silhouette score helps evaluate clustering quality. Let's compute it for `k=5`.
# 
# ```python
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# 
# kmeans = KMeans(n_clusters=5, random_state=42)
# kmeans.fit(X_scaled)
# labels = kmeans.labels_
# silhouette = silhouette_score(X_scaled, labels)
# print(f'Silhouette Score for k=5: {silhouette}')
# ```
# **Answer**: The silhouette score for `k=5` clusters is around 0.559, indicating reasonable clustering quality.

# ### Interpreting the Silhouette Score
# 
# A silhouette score close to 1 indicates well-clustered points. The score of 0.559 suggests good clustering, as it is greater than 0 and relatively close to 1.

# ### Next Steps After Poor Silhouette Score
# 
# If the silhouette score is unsatisfactory, we need to consider the next steps: increasing/decreasing the number of clusters, exploring other clustering algorithms, or enriching the dataset.

# ### Fixing Agglomerative Clustering Code
# 
# Hierarchical clustering can be a good alternative to K-Means. Let's apply agglomerative clustering.
# 
# ```python
# from sklearn.cluster import AgglomerativeClustering
# 
# cluster = AgglomerativeClustering(n_clusters=5, linkage='ward')
# labels = cluster.fit_predict(X_scaled)
# print(labels)
# ```
# **Answer**: Use `cluster.fit_predict(X_scaled)` to obtain cluster labels.

# ### Differences Between KMeans and Hierarchical Clustering
# 
# K-Means and hierarchical clustering differ in their approach to forming clusters. Hierarchical clustering does not require the number of clusters to be predefined, unlike K-Means.

# ### Interpreting Dendrograms
# 
# A dendrogram helps visualize the hierarchical relationship between clusters. Let's use it to determine the optimal number of clusters.
# 
# ```python
# import scipy.cluster.hierarchy as sch
# 
# plt.figure(figsize=(10, 7))
# dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
# plt.title('Dendrogram')
# plt.xlabel('Customers')
# plt.ylabel('Euclidean distances')
# plt.show()
# ```
# **Insight**: Based on the dendrogram, the most appropriate number of clusters appears to be 5.

# ### Analyzing Cluster Characteristics
# 
# Let's analyze the characteristics of each cluster by computing the mean values of features within each cluster.
# 
# ```python
# cluster_means = data.groupby('Cluster').mean()
# print(cluster_means)
# ```
# **Insight**: Cluster 3 includes customers with significantly low annual income and spending scores.

# ### Practical Interpretation of Clusters
# 
# Interpreting the characteristics of each cluster helps in understanding customer segments. For example, customers in Cluster 2, on average, earn more annually compared to the average earnings of all customers in the dataset.

# ### Visualizing Data Points Distribution
# 
# A bar chart helps visualize the distribution of data points among clusters.
# 
# ```python
# plt.figure(figsize=(10, 6))
# sns.countplot(x='Cluster', data=data, palette='viridis')
# plt.title('Number of Data Points per Cluster')
# plt.xlabel('Cluster')
# plt.ylabel('Number of Data Points')
# plt.show()
# ```
# **Insight**: The uneven distribution suggests potential issues with the clustering model's performance, such as overfitting.

# ### Understanding Davies-Bouldin Index
# 
# The Davies-Bouldin Index (DBI) is another metric for evaluating clustering quality. A lower DBI value indicates better clustering, where clusters are more compact and well-separated.
# 
# ```python
# from sklearn.metrics import davies_bouldin_score
# 
# dbi = davies_bouldin_score(X_scaled, labels)
# print(f'Davies-Bouldin Index: {dbi}')
# ```
# **Answer**: A lower DBI value indicates better clustering.

# ### Linkage Criteria in Hierarchical Clustering
# 
# Different linkage criteria in hierarchical clustering measure distances differently. Single linkage measures the shortest distance between any two points in the two clusters being merged.

# ### Next Steps Based on Clustering Evaluation
# 
# Based on the evaluation, the next steps may include refitting clustering models with different hyperparameters, performing further exploratory data analysis, and including additional relevant features.

# ### Conclusion
# 
# Clustering is a powerful tool for uncovering patterns and segments within data. By understanding and applying K-Means and Hierarchical Clustering, and evaluating their performance using various metrics, we can achieve meaningful insights and better decision-making. This comprehensive guide aims to equip you with the knowledge and skills to perform clustering effectively. Happy clustering!
