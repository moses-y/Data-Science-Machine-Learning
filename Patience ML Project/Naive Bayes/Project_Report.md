# Project Report: Naive Bayes Sentiment Analysis

**Author**: Patience
**Date**: August 10, 2025

## 1. Project Overview

This project involved the implementation of a Naive Bayes classifier from scratch to perform sentiment analysis on the IMDB movie review dataset. The primary goal was to classify movie reviews as either "positive" or "negative" based on their text content.

The project followed a complete machine learning pipeline:
- **Data Loading and Preprocessing**: Loading the standard IMDB dataset and cleaning the text.
- **Feature Engineering**: Converting raw text into a numerical format suitable for the model using a Bag-of-Words approach.
- **Model Implementation**: Building the Naive Bayes algorithm from the ground up.
- **Training and Evaluation**: Training the model on 25,000 reviews and evaluating its performance on a separate set of 25,000 unseen reviews.
- **Error Analysis and Visualization**: Deeply analyzing the model's mistakes and creating a suite of visualizations to understand its performance.

## 2. The Process and Implementation Details

### 2.1. Data Preprocessing
Each review underwent a series of cleaning steps to prepare it for feature extraction. This included converting text to lowercase, removing HTML tags, punctuation, and standard English "stopwords" (common words like "the", "a", "in" that provide little semantic value).

### 2.2. Feature Extraction: Bag-of-Words
The core of the text-to-number conversion was the **Bag-of-Words** model. This model represents each review as a numerical vector. A vocabulary of the most common words is first built from the training data. Then, for each review, the vector indicates the presence or absence of each vocabulary word.

### 2.3. Naive Bayes Classifier
The classifier was implemented from scratch based on Bayes' theorem. It calculates the probability of a review belonging to a certain class (positive or negative) given the words it contains. During training, it learns two key things:
1.  **Class Priors**: The overall probability of any given review being positive or negative.
2.  **Feature Likelihoods**: The probability of a specific word appearing in a review, given that the review is positive or negative (e.g., P("excellent" | positive)).

## 3. Challenges Faced and Solutions

The implementation journey was not without its challenges. The most significant hurdles were related to computational performance and memory management, which required several iterations of optimization.

### Challenge 1: Extreme Computational Cost and Memory Usage

**The Problem**: The initial implementation attempted to build a vocabulary from every single unique word in the 25,000 training reviews. This resulted in a vocabulary of over **2.1 million unique words**. When creating the feature vectors, this led to an attempt to generate over **100 billion data points** (50,000 reviews * 2.1 million features). This approach was computationally infeasible, would have taken many hours to run, and would have certainly crashed on a standard machine with 16 GB of RAM due to memory exhaustion.

**The Solution**: The solution was to adopt a standard practice in natural language processing: **limiting the vocabulary size**. I modified the code to only consider the **top 15,000 most frequent words**. This dramatically reduced the size of the feature vectors and made the problem computationally tractable, bringing the number of data points down to a manageable 750 million.

### Challenge 2: Inefficient Feature Extraction Loop

**The Problem**: Even after limiting the vocabulary, the feature extraction process was still taking over 30 minutes, far longer than expected. Upon investigation, I discovered a major bottleneck in the code. For each of the 50,000 reviews, the program was iterating through the entire 15,000-word vocabulary and, for each word, searching through the list of words in that review. This nested loop structure resulted in an enormous number of redundant checks.

**The Solution**: The fix was simple but incredibly effective. Before processing a review, I converted its list of words into a Python `set`. Checking for the existence of an item in a `set` is an extremely fast, constant-time operation, whereas searching a `list` is much slower. This single change reduced the feature extraction time from over 30 minutes to **under 7 minutes**.

### Challenge 3: Slow Evaluation and Prediction

**The Problem**: After the model was trained, the evaluation step (making predictions on the 25,000 test reviews) was also unexpectedly slow. The issue was similar to the previous one: the code was processing the test reviews one by one in a Python loop, which is inherently inefficient for this kind of task.

**The Solution**: I refactored the prediction logic to be more aligned with how optimized libraries like scikit-learn operate. Instead of a one-by-one loop, I modified the code to transform the *entire* test set into a feature matrix first. This bulk matrix was then passed to the classifier to get all predictions in a single, highly optimized batch operation, significantly speeding up the evaluation phase.

## 4. Results and Performance Analysis

### 4.1. Model Performance Metrics
The final trained model achieved the following performance metrics on the test set of 25,000 unseen reviews:

- **Accuracy**: 81.11% (20,277 correct predictions out of 25,000 samples)
- **Precision**: 81.12%
- **Recall**: 81.11%
- **F1-Score**: 81.11%
- **Total Errors**: 4,723 misclassified reviews

### 4.2. Dataset Specifications
- **Training Set**: 25,000 IMDB movie reviews
- **Test Set**: 25,000 IMDB movie reviews
- **Vocabulary Size**: 15,000 most frequent words (reduced from 2.1+ million)
- **Feature Vector Dimensions**: 15,000 features per review
- **Total Data Points Processed**: 750 million (50,000 reviews × 15,000 features)

### 4.3. Error Analysis Insights
The model's misclassifications revealed interesting patterns:

1. **Sarcastic Reviews**: The model struggled with reviews that used negative words to express positive sentiment (e.g., calling a movie "hilariously bad" in a positive way).

2. **Complex Vocabulary**: Long, detailed reviews with sophisticated language often confused the model, particularly when they mixed positive and negative descriptors.

3. **Cult Film Reviews**: Reviews for "so bad it's good" movies were frequently misclassified, as they contained many negative words while expressing positive overall sentiment.

4. **Extreme Confidence in Errors**: Notably, the top 5 misclassifications all had confidence scores of 100% (1.0000), indicating the model was very certain about its incorrect predictions.

### 4.4. Computational Performance Achievements
Through optimization efforts, we achieved significant performance improvements:

- **Initial Implementation**: Would have required processing 100+ billion data points (infeasible)
- **Optimized Implementation**: Successfully processed 750 million data points
- **Feature Extraction Time**: Reduced from 30+ minutes to under 7 minutes
- **Memory Usage**: Reduced from potentially crashing a 16GB system to running smoothly


## 5. Final Conclusion

This project successfully demonstrates the from-scratch implementation of a Naive Bayes classifier for sentiment analysis. The final model achieved a solid accuracy of **81.11%** on 25,000 unseen movie reviews, correctly classifying over 20,000 reviews while making 4,723 errors.

The primary challenges were not in the machine learning theory itself, but in the practical aspects of software engineering: writing efficient, optimized code that can handle large datasets without succumbing to performance bottlenecks. The iterative process of identifying and solving these bottlenecks—reducing vocabulary size from 2.1 million to 15,000 words, optimizing data structures from lists to sets, and implementing batch processing—was a critical part of the project's success.

The model's limitations, particularly with sarcastic content and complex language, highlight the inherent challenges in natural language processing and provide clear directions for future improvements, such as incorporating more sophisticated feature extraction techniques or ensemble methods.
