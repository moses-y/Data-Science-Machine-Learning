import numpy as np
import pandas as pd
import re
import string
import math
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset creation (you can replace with your own dataset)
def create_sample_dataset():
    """
    Creates a sample movie review dataset for demonstration.
    In practice, you would load from files like IMDB reviews, Amazon reviews, etc.
    """
    positive_reviews = [
        "This movie is absolutely fantastic and amazing",
        "I loved every moment of this incredible film",
        "Outstanding performance and brilliant storytelling",
        "Excellent cinematography and wonderful acting",
        "Best movie I have seen this year, highly recommended",
        "Masterpiece of cinema with great direction",
        "Phenomenal acting and superb screenplay",
        "Brilliant film with outstanding performances",
        "Amazing story and excellent character development",
        "Perfect blend of drama and entertainment"
    ]
    
    negative_reviews = [
        "This movie is terrible and boring",
        "Worst film I have ever watched, complete waste of time",
        "Poor acting and horrible storyline",
        "Disappointing movie with bad direction",
        "Awful screenplay and terrible performances",
        "Boring plot and uninteresting characters",
        "Poor quality film with bad acting",
        "Disappointing and poorly executed movie",
        "Terrible direction and horrible cinematography",
        "Worst movie experience, very disappointing"
    ]
    
    # Create dataset
    reviews = positive_reviews + negative_reviews
    labels = ['positive'] * len(positive_reviews) + ['negative'] * len(negative_reviews)
    
    return pd.DataFrame({
        'review': reviews,
        'sentiment': labels
    })

class TextPreprocessor:
    """
    Handles text preprocessing for sentiment analysis
    """
    
    def __init__(self):
        self.stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
            'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above', 
            'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
            'further', 'then', 'once'
        }
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text):
        """Tokenize text into words"""
        return text.split()
    
    def remove_stop_words(self, tokens):
        """Remove stop words from token list"""
        return [token for token in tokens if token not in self.stop_words]
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        filtered_tokens = self.remove_stop_words(tokens)
        return filtered_tokens

class FeatureExtractor:
    """
    Extracts features for Naive Bayes classifier
    """
    
    def __init__(self, feature_type='bag_of_words'):
        self.feature_type = feature_type
        self.vocabulary = set()
        self.word_freq = defaultdict(int)
        
    def build_vocabulary(self, processed_texts):
        """Build vocabulary from processed texts"""
        for tokens in processed_texts:
            for token in tokens:
                self.vocabulary.add(token)
                self.word_freq[token] += 1
        
        print(f"Vocabulary size: {len(self.vocabulary)}")
        return self.vocabulary
    
    def extract_bag_of_words(self, tokens):
        """Extract bag-of-words features"""
        features = {}
        for word in self.vocabulary:
            features[word] = 1 if word in tokens else 0
        return features
    
    def extract_tf_features(self, tokens):
        """Extract term frequency features"""
        features = {}
        token_count = Counter(tokens)
        total_tokens = len(tokens)
        
        for word in self.vocabulary:
            features[word] = token_count[word] / total_tokens if total_tokens > 0 else 0
        return features
    
    def extract_features(self, tokens):
        """Extract features based on specified type"""
        if self.feature_type == 'bag_of_words':
            return self.extract_bag_of_words(tokens)
        elif self.feature_type == 'term_frequency':
            return self.extract_tf_features(tokens)
        else:
            raise ValueError("Unsupported feature type")

class NaiveBayesClassifier:
    """
    Naive Bayes Classifier implemented from scratch for sentiment analysis
    """
    
    def __init__(self, alpha=1.0):
        """
        Initialize Naive Bayes classifier
        
        Args:
            alpha: Smoothing parameter for Laplace smoothing
        """
        self.alpha = alpha  # Laplace smoothing parameter
        self.class_priors = {}  # P(class)
        self.feature_likelihoods = {}  # P(feature|class)
        self.classes = []
        self.vocabulary = set()
        
    def fit(self, X, y):
        """
        Train the Naive Bayes classifier
        
        Args:
            X: List of feature dictionaries
            y: List of class labels
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
        self.feature_likelihoods = {class_label: {} for class_label in self.classes}
        
        for class_label in self.classes:
            # Get all feature vectors for this class
            class_features = [X[i] for i in range(len(X)) if y[i] == class_label]
            
            # Count feature occurrences for this class
            feature_counts = defaultdict(int)
            total_features = 0
            
            for features in class_features:
                for feature, value in features.items():
                    feature_counts[feature] += value
                    total_features += value
            
            # Calculate likelihoods with Laplace smoothing
            for feature in self.vocabulary:
                numerator = feature_counts[feature] + self.alpha
                denominator = total_features + self.alpha * len(self.vocabulary)
                self.feature_likelihoods[class_label][feature] = numerator / denominator
    
    def predict_proba(self, features):
        """
        Predict class probabilities for given features
        
        Args:
            features: Dictionary of features
            
        Returns:
            Dictionary of class probabilities
        """
        class_scores = {}
        
        for class_label in self.classes:
            # Start with log prior
            log_prob = math.log(self.class_priors[class_label])
            
            # Add log likelihoods
            for feature, value in features.items():
                if feature in self.feature_likelihoods[class_label] and value > 0:
                    log_prob += value * math.log(self.feature_likelihoods[class_label][feature])
            
            class_scores[class_label] = log_prob
        
        # Convert log probabilities to regular probabilities
        max_score = max(class_scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in class_scores.items()}
        total = sum(exp_scores.values())
        
        return {k: v / total for k, v in exp_scores.items()}
    
    def predict(self, X):
        """
        Predict classes for given feature vectors
        
        Args:
            X: List of feature dictionaries
            
        Returns:
            List of predicted class labels
        """
        predictions = []
        for features in X:
            probabilities = self.predict_proba(features)
            predicted_class = max(probabilities, key=probabilities.get)
            predictions.append(predicted_class)
        return predictions

class SentimentAnalyzer:
    """
    Complete sentiment analysis pipeline
    """
    
    def __init__(self, feature_type='bag_of_words', alpha=1.0):
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor(feature_type)
        self.classifier = NaiveBayesClassifier(alpha)
        self.feature_type = feature_type
        
    def train(self, texts, labels):
        """Train the sentiment analyzer"""
        print("Preprocessing texts...")
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        
        print("Building vocabulary...")
        self.feature_extractor.build_vocabulary(processed_texts)
        
        print("Extracting features...")
        X = [self.feature_extractor.extract_features(tokens) for tokens in processed_texts]
        
        print("Training classifier...")
        self.classifier.fit(X, labels)
        
        return X
    
    def predict(self, texts):
        """Predict sentiment for given texts"""
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        X = [self.feature_extractor.extract_features(tokens) for tokens in processed_texts]
        return self.classifier.predict(X)
    
    def predict_proba(self, texts):
        """Predict sentiment probabilities for given texts"""
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        X = [self.feature_extractor.extract_features(tokens) for tokens in processed_texts]
        return [self.classifier.predict_proba(features) for features in X]

def evaluate_model(y_true, y_pred, classes):
    """Evaluate model performance"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return accuracy, precision, recall, f1

def error_analysis(analyzer, X_test, y_test, test_texts):
    """Perform error analysis"""
    predictions = analyzer.predict(test_texts)
    probabilities = analyzer.predict_proba(test_texts)
    
    print("\n=== Error Analysis ===")
    errors = []
    
    for i, (true_label, pred_label, text, probs) in enumerate(
        zip(y_test, predictions, test_texts, probabilities)
    ):
        if true_label != pred_label:
            confidence = max(probs.values())
            errors.append({
                'index': i,
                'text': text,
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': confidence,
                'probabilities': probs
            })
    
    print(f"Total errors: {len(errors)}")
    print("\nTop 5 most confident misclassifications:")
    
    # Sort by confidence (most confident errors first)
    errors.sort(key=lambda x: x['confidence'], reverse=True)
    
    for i, error in enumerate(errors[:5]):
        print(f"\n{i+1}. Text: '{error['text']}'")
        print(f"   True: {error['true_label']}, Predicted: {error['predicted_label']}")
        print(f"   Confidence: {error['confidence']:.4f}")
        print(f"   Probabilities: {error['probabilities']}")

def main():
    """Main function to demonstrate the implementation"""
    print("=== Naive Bayes Sentiment Analysis Implementation ===\n")
    
    # Create sample dataset (replace with your own dataset)
    print("Creating sample dataset...")
    df = create_sample_dataset()
    print(f"Dataset size: {len(df)}")
    print(f"Class distribution:\n{df['sentiment'].value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'].tolist(), 
        df['sentiment'].tolist(), 
        test_size=0.3, 
        random_state=42,
        stratify=df['sentiment']
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Test different feature extraction methods
    feature_types = ['bag_of_words', 'term_frequency']
    results = {}
    
    for feature_type in feature_types:
        print(f"\n\n=== Testing {feature_type.upper()} Features ===")
        
        # Initialize and train analyzer
        analyzer = SentimentAnalyzer(feature_type=feature_type, alpha=1.0)
        analyzer.train(X_train, y_train)
        
        # Make predictions
        predictions = analyzer.predict(X_test)
        
        # Evaluate
        accuracy, precision, recall, f1 = evaluate_model(
            y_test, predictions, analyzer.classifier.classes
        )
        
        results[feature_type] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # Error analysis
        error_analysis(analyzer, X_test, y_test, X_test)
    
    # Compare results
    print("\n\n=== Results Comparison ===")
    results_df = pd.DataFrame(results).T
    print(results_df)
    
    # Test on new examples
    print("\n\n=== Testing on New Examples ===")
    new_texts = [
        "This movie is absolutely amazing and wonderful!",
        "Terrible film, completely boring and disappointing.",
        "Not bad, but could be better.",
        "Outstanding performance by the actors."
    ]
    
    # Use the best performing model
    best_feature_type = max(results.keys(), key=lambda k: results[k]['accuracy'])
    print(f"Using best model: {best_feature_type}")
    
    final_analyzer = SentimentAnalyzer(feature_type=best_feature_type, alpha=1.0)
    final_analyzer.train(X_train, y_train)
    
    predictions = final_analyzer.predict(new_texts)
    probabilities = final_analyzer.predict_proba(new_texts)
    
    for text, pred, probs in zip(new_texts, predictions, probabilities):
        print(f"\nText: '{text}'")
        print(f"Predicted: {pred}")
        print(f"Probabilities: {probs}")

if __name__ == "__main__":
    main()