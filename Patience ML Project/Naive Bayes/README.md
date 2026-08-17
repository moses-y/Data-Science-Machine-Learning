# Naive Bayes Sentiment Analysis on IMDB Movie Reviews

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0rc0-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This project is a comprehensive implementation of a Naive Bayes classifier from scratch to perform sentiment analysis on the IMDB movie review dataset. The model classifies reviews as either "positive" or "negative" and includes a full pipeline for data preprocessing, feature extraction, training, evaluation, and visualization.

## Key Features

- **From-Scratch Implementation**: The Naive Bayes classifier is built using standard Python libraries to demonstrate a fundamental understanding of the algorithm.
- **Advanced Feature Engineering**: Utilizes a Bag-of-Words model with n-grams and a limited vocabulary size for optimal performance.
- **In-Depth Evaluation**: Generates a suite of metrics and visualizations to thoroughly assess model performance.
- **Rich Visualizations**: Automatically produces a confusion matrix, classification report, ROC curve, feature importance plots, and word clouds.
- **Error Analysis**: Identifies the most confidently misclassified reviews to provide insight into the model's limitations.
- **Experiment Tracking**: Includes a robust logging and versioning system that saves the model, logs, and all visual assets for each run in a uniquely timestamped folder.

## Final Results

The model achieves a respectable accuracy and F1-score, demonstrating a solid ability to classify sentiment on unseen data.

- **Accuracy**: ~81.1%
- **F1-Score**: ~81.1%

### Performance Visualizations

Here is a selection of the visual outputs generated during the final run.

| Classification Report | Confusion Matrix |
| :---: | :---: |
| ![Classification Report](assets/classification_report.png) | ![Confusion Matrix](assets/confusion_matrix.png) |

| ROC Curve | Feature Importance |
| :---: | :---: |
| ![ROC Curve](assets/roc_curve.png) | ![Feature Importance](assets/feature_importance.png) |

### Positive vs. Negative Word Clouds

These word clouds show the most influential words for each sentiment class.

| Top Positive Words | Top Negative Words |
| :---: | :---: |
| ![Positive Word Cloud](assets/wordcloud_positive.png) | ![Negative Word Cloud](assets/wordcloud_negative.png) |


## How to Run

1.  **Set up the environment**:
    It is highly recommended to use a virtual environment.
    ```bash
    # Create the virtual environment
    uv venv

    # Activate the virtual environment
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

2.  **Install dependencies**:
    ```bash
    uv pip install -r requirements.txt
    ```

3.  **Run the pipeline**:
    ```bash
    python main.py
    ```
    The script will train the model, run the evaluation, and save the model, logs, and all visualization assets to timestamped directories in the `models/` and `assets/` folders.

## Project Structure

```
├── assets/                # To store generated plots and images
├── data/                  # For storing datasets (if not downloaded)
├── logs/                  # Contains detailed logs for each run
├── models/                # Stores the trained model checkpoints
├── src/                   # Source code for the project
│   ├── __init__.py
│   ├── classifier.py      # Core Naive Bayes algorithm
│   ├── evaluation.py      # Evaluation metrics and plotting functions
│   ├── features.py        # Feature extraction (Bag-of-Words)
│   ├── preprocessing.py   # Text cleaning and preparation
│   └── utils.py           # Helper functions (data loading, model saving)
├── main.py                # Main script to run the entire pipeline
├── requirements.txt       # Project dependencies
└── README.md              # This file
```