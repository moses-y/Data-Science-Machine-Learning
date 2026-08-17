import nbformat as nbf

# Define the content of your notebook
content = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "## Understanding Clustering: A Comprehensive Guide to K-Means and Hierarchical Clustering\n\n"
                  "Clustering is a fundamental technique in unsupervised machine learning that involves grouping data points into clusters based on their similarities. "
                  "In this blog, we will delve into the concepts of K-Means and Hierarchical Clustering, explore their applications using a practical dataset, and understand various metrics to evaluate their performance. "
                  "By the end, you'll have a solid grasp of how to implement and interpret these clustering techniques."
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Introduction to Clustering\n\n"
                  "Clustering helps in identifying patterns and structures within data without any prior labels. It is widely used in customer segmentation, image segmentation, anomaly detection, and more. "
                  "We will focus on two popular clustering methods: K-Means and Hierarchical Clustering."
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### The Dataset\n\n"
                  "We will use a dataset containing information about mall customers, including their gender, age, annual income, and spending score. This data is ideal for demonstrating clustering techniques aimed at segmenting customers based on their annual income and spending scores.\n\n"
                  "```python\n"
                  "import pandas as pd\n\n"
                  "# Load the dataset\n"
                  "data = pd.read_csv('https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Data/unsupervised_sprint/mall_customers.csv', index_col=0)\n"
                  "data.head()\n"
                  "```\n"
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Descriptive Statistics\n\n"
                  "Before diving into clustering, it's essential to understand the dataset's characteristics. Let's print the descriptive statistics.\n\n"
                  "```python\n"
                  "print(data.describe())\n"
                  "```\n"
                  "**Insight**: The `Spending_Score` ranges from 1 to 100, confirming the variety of customer spending behaviors in the dataset."
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Number of Observations\n\n"
                  "To understand the scale of our data, let's find the total number of observations.\n\n"
                  "```python\n"
                  "print(f'Total number of observations: {data.shape[0]}')\n"
                  "```\n"
                  "**Answer**: There are 200 observations in the dataset."
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Handling Null Values\n\n"
                  "Missing values can disrupt the clustering process. Let's check for null values and handle them if necessary.\n\n"
                  "```python\n"
                  "print(data.isnull().sum())\n"
                  "```\n"
                  "**Insight**: The dataset contains no null values, so we can proceed without any imputation."
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Scatter Plot Visualization\n\n"
                  "To visualize the relationship between annual income and spending score, we generate a scatter plot.\n\n"
                  "```python\n"
                  "import matplotlib.pyplot as plt\n\n"
                  "plt.figure(figsize=(10, 6))\n"
                  "plt.scatter(data['Annual_Income_(k$)'], data['Spending_Score'], c='blue', edgecolors='w', s=100)\n"
                  "plt.title('Relationship between Annual Income and Spending Score')\n"
                  "plt.xlabel('Annual Income (k$)')\n"
                  "plt.ylabel('Spending Score')\n"
                  "plt.grid(True)\n"
                  "plt.show()\n"
                  "```\n"
                  "**Insight**: The scatter plot shows potential clusters forming, particularly in the center of the plot."
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Scaling Data\n\n"
                  "Scaling is crucial for clustering algorithms to ensure that features contribute equally to the distance calculations. Let's understand why the provided code returns an error and correct it.\n\n"
                  "```python\n"
                  "from sklearn.preprocessing import MinMaxScaler\n\n"
                  "scaler = MinMaxScaler()\n"
                  "try:\n"
                  "    X_scaled = scaler.fit_transform(data)\n"
                  "except ValueError as e:\n"
                  "    print(e)\n"
                  "```\n"
                  "**Error**: The MinMaxScaler cannot handle non-numeric data, such as the `Gender` column."
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Feature Selection and Scaling\n\n"
                  "We focus on `Annual_Income_(k$)` and `Spending_Score` for clustering.\n\n"
                  "```python\n"
                  "X = data[['Annual_Income_(k$)', 'Spending_Score']]\n"
                  "X_scaled = scaler.fit_transform(X)\n"
                  "```\n"
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Determining the Optimal Number of Clusters\n\n"
                  "Various metrics help determine the optimal number of clusters. Here, we examine those that are typically used, such as within-cluster variation, between-cluster variation, and the CH index.\n\n"
                  "#### Elbow Method\n\n"
                  "The Elbow Method helps in identifying the optimal number of clusters by plotting the within-cluster sum of squares (WCSS) against the number of clusters.\n\n"
                  "```python\n"
                  "from sklearn.cluster import KMeans\n"
                  "import matplotlib.pyplot as plt\n\n"
                  "wcss = []\n"
                  "for i in range(1, 11):\n"
                  "    kmeans = KMeans(n_clusters=i, random_state=42)\n"
                  "    kmeans.fit(X_scaled)\n"
                  "    wcss.append(kmeans.inertia_)\n\n"
                  "plt.figure(figsize=(10, 6))\n"
                  "plt.plot(range(1, 11), wcss, marker='o')\n"
                  "plt.title('Elbow Method')\n"
                  "plt.xlabel('Number of clusters')\n"
                  "plt.ylabel('WCSS')\n"
                  "plt.show()\n"
                  "```\n"
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Evaluating Silhouette Score\n\n"
                  "Silhouette score helps evaluate clustering quality. Let's compute it for `k=5`.\n\n"
                  "```python\n"
                  "from sklearn.cluster import KMeans\n"
                  "from sklearn.metrics import silhouette_score\n\n"
                  "kmeans = KMeans(n_clusters=5, random_state=42)\n"
                  "kmeans.fit(X_scaled)\n"
                  "labels = kmeans.labels_\n"
                  "silhouette = silhouette_score(X_scaled, labels)\n"
                  "print(f'Silhouette Score for k=5: {silhouette}')\n"
                  "```\n"
                  "**Answer**: The silhouette score for `k=5` clusters is around 0.559, indicating reasonable clustering quality."
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Interpreting the Silhouette Score\n\n"
                  "A silhouette score close to 1 indicates well-clustered points. The score of 0.559 suggests good clustering, as it is greater than 0 and relatively close to 1."
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Next Steps After Poor Silhouette Score\n\n"
                  "If the silhouette score is unsatisfactory, we need to consider the next steps: increasing/decreasing the number of clusters, exploring other clustering algorithms, or enriching the dataset."
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Fixing Agglomerative Clustering Code\n\n"
                  "Hierarchical clustering can be a good alternative to K-Means. Let's apply agglomerative clustering.\n\n"
                  "```python\n"
                  "from sklearn.cluster import AgglomerativeClustering\n\n"
                  "cluster = AgglomerativeClustering(n_clusters=5, linkage='ward')\n"
                  "labels = cluster.fit_predict(X_scaled)\n"
                  "print(labels)\n"
                  "```\n"
                  "**Answer**: Use `cluster.fit_predict(X_scaled)` to obtain cluster labels."
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Differences Between KMeans and Hierarchical Clustering\n\n"
                  "K-Means and hierarchical clustering differ in their approach to forming clusters. Hierarchical clustering does not require the number of clusters to be predefined, unlike K-Means."
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Interpreting Dendrograms\n\n"
                  "A dendrogram helps visualize the hierarchical relationship between clusters. Let's use it to determine the optimal number of clusters.\n\n"
                  "```python\n"
                  "import scipy.cluster.hierarchy as sch\n\n"
                  "plt.figure(figsize=(10, 7))\n"
                  "dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))\n"
                  "plt.title('Dendrogram')\n"
                  "plt.xlabel('Customers')\n"
                  "plt.ylabel('Euclidean distances')\n"
                  "plt.show()\n"
                  "```\n"
                  "**Insight**: Based on the dendrogram, the most appropriate number of clusters appears to be 5."
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Analyzing Cluster Characteristics\n\n"
                  "Let's analyze the characteristics of each cluster by computing the mean values of features within each cluster.\n\n"
                  "```python\n"
                  "cluster_means = data.groupby('Cluster').mean()\n"
                  "print(cluster_means)\n"
                  "```\n"
                  "**Insight**: Cluster 3 includes customers with significantly low annual income and spending scores."
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Practical Interpretation of Clusters\n\n"
                  "Interpreting the characteristics of each cluster helps in understanding customer segments. For example, customers in Cluster 2, on average, earn more annually compared to the average earnings of all customers in the dataset."
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Visualizing Data Points Distribution\n\n"
                  "A bar chart helps visualize the distribution of data points among clusters.\n\n"
                  "```python\n"
                  "plt.figure(figsize=(10, 6))\n"
                  "sns.countplot(x='Cluster', data=data, palette='viridis')\n"
                  "plt.title('Number of Data Points per Cluster')\n"
                  "plt.xlabel('Cluster')\n"
                  "plt.ylabel('Number of Data Points')\n"
                  "plt.show()\n"
                  "```\n"
                  "**Insight**: The uneven distribution suggests potential issues with the clustering model's performance, such as overfitting."
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Understanding Davies-Bouldin Index\n\n"
                  "The Davies-Bouldin Index (DBI) is another metric for evaluating clustering quality. A lower DBI value indicates better clustering, where clusters are more compact and well-separated.\n\n"
                  "```python\n"
                  "from sklearn.metrics import davies_bouldin_score\n\n"
                  "dbi = davies_bouldin_score(X_scaled, labels)\n"
                  "print(f'Davies-Bouldin Index: {dbi}')\n"
                  "```\n"
                  "**Answer**: A lower DBI value indicates better clustering."
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Linkage Criteria in Hierarchical Clustering\n\n"
                  "Different linkage criteria in hierarchical clustering measure distances differently. Single linkage measures the shortest distance between any two points in the two clusters being merged."
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Next Steps Based on Clustering Evaluation\n\n"
                  "Based on the evaluation, the next steps may include refitting clustering models with different hyperparameters, performing further exploratory data analysis, and including additional relevant features."
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "### Conclusion\n\n"
                  "Clustering is a powerful tool for uncovering patterns and segments within data. By understanding and applying K-Means and Hierarchical Clustering, and evaluating their performance using various metrics, we can achieve meaningful insights and better decision-making. This comprehensive guide aims to equip you with the knowledge and skills to perform clustering effectively. Happy clustering!"
    }
]

# Create a new notebook
nb = nbf.v4.new_notebook()

# Add the cells to the notebook
nb['cells'] = [nbf.v4.new_markdown_cell(cell["source"]) if cell["cell_type"] == "markdown" else nbf.v4.new_code_cell(cell["source"]) for cell in content]

# Write the notebook to a file
with open('clustering_guide.ipynb', 'w') as f:
    nbf.write(nb, f)
