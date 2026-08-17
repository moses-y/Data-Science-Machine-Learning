import math
from collections import defaultdict, Counter

class NaiveBayesClassifier:
    """
    Naive Bayes Classifier implemented from scratch for sentiment analysis.
    """
    
    def __init__(self, alpha=1.0):
        """
        Initialize Naive Bayes classifier.
        
        Args:
            alpha: Smoothing parameter for Laplace smoothing.
        """
        self.alpha = alpha
        self.class_priors = {}
        self.feature_likelihoods = {}
        self.classes = []
        self.vocabulary = set()
        
    def fit(self, X, y):
        """
        Train the Naive Bayes classifier.
        
        Args:
            X: List of feature dictionaries.
            y: List of class labels.
        """
        self.classes = list(set(y))
        n_samples = len(y)
        
        # Calculate class priors P(class)
        class_counts = Counter(y)
        for class_label in self.classes:
            self.class_priors[class_label] = class_counts[class_label] / n_samples
        
        # Build vocabulary from all features
        for features in X:
            self.vocabulary.update(features.keys())
        
        # Calculate feature likelihoods P(feature|class)
        self.feature_likelihoods = {class_label: defaultdict(float) for class_label in self.classes}
        
        for class_label in self.classes:
            class_features = [X[i] for i in range(len(X)) if y[i] == class_label]
            
            feature_counts = defaultdict(int)
            total_feature_values = 0
            
            for features in class_features:
                for feature, value in features.items():
                    # For BoW, value is 0 or 1. For TF, it's a float.
                    # We sum the values.
                    feature_counts[feature] += value
                    total_feature_values += value
            
            # Calculate likelihoods with Laplace smoothing
            vocab_size = len(self.vocabulary)
            for feature in self.vocabulary:
                numerator = feature_counts[feature] + self.alpha
                denominator = total_feature_values + self.alpha * vocab_size
                self.feature_likelihoods[class_label][feature] = numerator / denominator
    
    def _predict_log_proba(self, features):
        """Calculate log probabilities for a single feature vector."""
        class_scores = {}
        for class_label in self.classes:
            log_prob = math.log(self.class_priors[class_label])
            
            for feature, value in features.items():
                if feature in self.vocabulary:
                    # Get the likelihood, use a small default if not present for a class
                    likelihood = self.feature_likelihoods[class_label].get(feature, 1e-10)
                    if likelihood > 0:
                        log_prob += value * math.log(likelihood)
            
            class_scores[class_label] = log_prob
        return class_scores

    def predict(self, X):
        """
        Predict classes for given feature vectors.
        
        Args:
            X: List of feature dictionaries.
            
        Returns:
            List of predicted class labels.
        """
        predictions = []
        for features in X:
            log_probs = self._predict_log_proba(features)
            predicted_class = max(log_probs, key=log_probs.get)
            predictions.append(predicted_class)
        return predictions

    def predict_proba(self, X):
        """
        Predict class probabilities for given feature vectors.
        
        Args:
            X: List of feature dictionaries.
            
        Returns:
            List of dictionaries containing class probabilities.
        """
        probabilities = []
        for features in X:
            log_probs = self._predict_log_proba(features)
            
            # Convert log probabilities to probabilities
            max_log_prob = max(log_probs.values())
            exp_scores = {k: math.exp(v - max_log_prob) for k, v in log_probs.items()}
            total_exp_score = sum(exp_scores.values())
            
            probs = {k: v / total_exp_score for k, v in exp_scores.items()}
            probabilities.append(probs)
        return probabilities

def get_feature_importance(classifier):
    """
    Extracts feature importance (log probabilities) from the classifier.
    
    Returns a dictionary where keys are features and values are their
    log probability difference between positive and negative classes.
    """
    feature_importance = {}
    pos_probs = classifier.feature_likelihoods.get('positive', {})
    neg_probs = classifier.feature_likelihoods.get('negative', {})
    
    # Ensure we have both classes before proceeding
    if not pos_probs or not neg_probs:
        return {}

    for feature in classifier.vocabulary:
        # Using log to show the magnitude of difference
        pos_score = math.log(pos_probs.get(feature, 1e-10))
        neg_score = math.log(neg_probs.get(feature, 1e-10))
        feature_importance[feature] = pos_score - neg_score
        
    return feature_importance
