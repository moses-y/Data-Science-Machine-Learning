from src.preprocessing import AdvancedTextPreprocessor
from src.features import AdvancedFeatureExtractor
from src.classifier import NaiveBayesClassifier, get_feature_importance
from src.evaluation import (
    evaluate_model, error_analysis, plot_classification_report, 
    plot_word_clouds, plot_feature_importance, plot_roc_curve
)
from src.utils import load_imdb_dataset, save_model, load_model
import os
import logging
from datetime import datetime

class SentimentAnalyzer:
    """
    Complete sentiment analysis pipeline integrating all modules.
    """
    
    def __init__(self, feature_types=['bag_of_words'], ngram_range=(1, 1), alpha=1.0, max_features=None):
        self.preprocessor = AdvancedTextPreprocessor()
        self.feature_extractor = AdvancedFeatureExtractor(feature_types, ngram_range, max_features)
        self.classifier = NaiveBayesClassifier(alpha)
        
    def train(self, texts, labels):
        """Train the sentiment analyzer."""
        print("\n1. Preprocessing texts...")
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        
        print("\n2. Building vocabulary and extracting features...")
        self.feature_extractor.build_vocabulary(processed_texts)
        X = [self.feature_extractor.extract_features(tokens) for tokens in processed_texts]
        
        print("\n3. Training classifier...")
        self.classifier.fit(X, labels)
        print("Training complete.")
        
    def predict(self, texts):
        """Predict sentiment for given texts."""
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        X = [self.feature_extractor.extract_features(tokens) for tokens in processed_texts]
        return self.classifier.predict(X)
    
    def predict_proba(self, texts):
        """Predict sentiment probabilities for given texts."""
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        X = [self.feature_extractor.extract_features(tokens) for tokens in processed_texts]
        return self.classifier.predict_proba(X)

def setup_logging(log_path='logs'):
    """Sets up logging to file and console."""
    os.makedirs(log_path, exist_ok=True)
    log_filename = os.path.join(log_path, f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging setup complete.")

def main():
    """Main function to run the sentiment analysis pipeline."""
    print("=== Naive Bayes Sentiment Analysis Pipeline ===\n")
    
    # --- Configuration ---
    MODELS_PATH = "models"
    ASSETS_PATH = "assets"
    MODEL_FILENAME = os.path.join(MODELS_PATH, "sentiment_analyzer.pkl")
    
    # Create directories if they don't exist
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(ASSETS_PATH, exist_ok=True)

    # --- Data Loading ---
    # Using the Keras IMDB dataset, which is already split.
    # Loading with the full vocabulary.
    X_train, X_test, y_train, y_test = load_imdb_dataset()
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # --- Model Training ---
    # Using Bag-of-Words with unigrams and bigrams, plus negation handling
    analyzer = SentimentAnalyzer(
        feature_types=['bag_of_words'], 
        ngram_range=(1, 2), # Use unigrams and bigrams
        alpha=1.0,
        max_features=15000
    )
    analyzer.train(X_train, y_train)
    
    # --- Model Evaluation ---
    predictions = analyzer.predict(X_test)
    probabilities = analyzer.predict_proba(X_test)
    
    evaluate_model(y_test, predictions, analyzer.classifier.classes, ASSETS_PATH)
    
    # --- Advanced Visualizations ---
    print("\n--- Generating Advanced Visualizations ---")
    feature_importance = get_feature_importance(analyzer.classifier)
    
    plot_classification_report(y_test, predictions, analyzer.classifier.classes, ASSETS_PATH)
    plot_word_clouds(feature_importance, ASSETS_PATH)
    plot_feature_importance(feature_importance, top_n=15, assets_path=ASSETS_PATH)
    plot_roc_curve(y_test, probabilities, analyzer.classifier.classes, ASSETS_PATH)
    
    # --- Error Analysis ---
    # Performing error analysis on the full test set.
    error_analysis(
        X_test, 
        y_test, 
        predictions, 
        probabilities
    )
    
    # --- Save Model ---
    save_model(analyzer, MODEL_FILENAME)
    
    # --- Test on New Examples ---
    print("\n\n=== Testing on New Examples with Saved Model ===")
    loaded_analyzer = load_model(MODEL_FILENAME)
    if not loaded_analyzer:
        return

    new_texts = [
        "This movie is absolutely amazing and wonderful!",
        "Terrible film, completely boring and disappointing.",
        "Not bad, but could be better.",
        "The acting was not good at all.",
        "Outstanding performance by the actors."
    ]
    
    new_predictions = loaded_analyzer.predict(new_texts)
    new_probabilities = loaded_analyzer.predict_proba(new_texts)
    
    for text, pred, probs in zip(new_texts, new_predictions, new_probabilities):
        print(f"\nText: '{text}'")
        print(f"Predicted: {pred}")
        print(f"Probabilities: {probs}")

if __name__ == "__main__":
    setup_logging()  # Setup logging before running the main pipeline
    main()
