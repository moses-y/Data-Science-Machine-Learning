import pandas as pd
import pickle
from tensorflow.keras.datasets import imdb

def load_imdb_dataset(num_words=None):
    """
    Loads the IMDB dataset from Keras, decodes it, and maps labels.
    Setting num_words=None loads the full vocabulary.
    """
    print("Loading IMDB dataset from Keras...")
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

    word_index = imdb.get_word_index()
    reverse_word_index = {v: k for k, v in word_index.items()}

    def decode_review(text_indices):
        # The indices are offset by 3 because 0, 1, and 2 are reserved indices for
        # "padding", "start of sequence", and "unknown".
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in text_indices])

    print("Decoding reviews...")
    X_train = [decode_review(review) for review in X_train]
    X_test = [decode_review(review) for review in X_test]
    
    # Map labels from 0/1 to negative/positive
    label_map = {0: 'negative', 1: 'positive'}
    y_train = [label_map[label] for label in y_train]
    y_test = [label_map[label] for label in y_test]
    
    print("IMDB dataset loaded and processed successfully.")
    return X_train, X_test, y_train, y_test

def load_dataset(file_path):
    """
    Load dataset from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from {file_path}")
        print(f"Dataset size: {len(df)}")
        print(f"Class distribution:\n{df['sentiment'].value_counts()}")
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None

def save_model(analyzer, file_path):
    """
    Serialize and save the trained model to a file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(analyzer, f)
    print(f"Model saved to {file_path}")

def load_model(file_path):
    """
    Load a serialized model from a file.
    """
    try:
        with open(file_path, 'rb') as f:
            analyzer = pickle.load(f)
        print(f"Model loaded from {file_path}")
        return analyzer
    except FileNotFoundError:
        print(f"Error: Model file not found at {file_path}")
        return None