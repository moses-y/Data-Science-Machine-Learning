from collections import defaultdict, Counter
import math

class AdvancedFeatureExtractor:
    """
    Extracts features for Naive Bayes classifier, supporting
    bag-of-words, TF, TF-IDF, and n-grams.
    """
    
    def __init__(self, feature_types=['bag_of_words'], ngram_range=(1, 1), max_features=None):
        self.feature_types = feature_types
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vocabulary = set()
        self.word_freq = defaultdict(int)
        self.doc_freq = defaultdict(int)
        self.num_docs = 0

    def _get_ngrams(self, tokens):
        """Generate n-grams from tokens."""
        all_ngrams = []
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                all_ngrams.append(' '.join(tokens[i:i+n]))
        return all_ngrams

    def build_vocabulary(self, processed_texts):
        """Build vocabulary from processed texts."""
        self.num_docs = len(processed_texts)
        all_features = []
        for tokens in processed_texts:
            features = self._get_ngrams(tokens)
            all_features.append(features)
            
            # For vocabulary and doc frequency
            unique_features_in_doc = set(features)
            for feature in unique_features_in_doc:
                self.doc_freq[feature] += 1

            # For word frequency
            for feature in features:
                 self.word_freq[feature] += 1

        # Create vocabulary from all features
        if self.max_features is not None:
            # Select top N features based on frequency
            sorted_word_freq = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
            self.vocabulary = {word for word, freq in sorted_word_freq[:self.max_features]}
        else:
            self.vocabulary = set(self.word_freq.keys())
        
        print(f"Vocabulary size: {len(self.vocabulary)}")
        return self.vocabulary

    def extract_bag_of_words(self, features):
        """Extract bag-of-words features."""
        bow_features = {}
        feature_set = set(features)
        for word in self.vocabulary:
            bow_features[word] = 1 if word in feature_set else 0
        return bow_features

    def extract_tf_features(self, features):
        """Extract term frequency features."""
        tf_features = {}
        feature_count = Counter(features)
        total_features = len(features)
        
        for word in self.vocabulary:
            tf_features[word] = feature_count[word] / total_features if total_features > 0 else 0
        return tf_features

    def extract_tfidf_features(self, features):
        """Extract TF-IDF features."""
        tfidf_features = {}
        tf_features = self.extract_tf_features(features)
        
        for word in self.vocabulary:
            tf = tf_features.get(word, 0)
            # Add 1 to num_docs to prevent division by zero for words not in doc_freq
            idf = math.log((self.num_docs + 1) / (self.doc_freq[word] + 1)) + 1
            tfidf_features[word] = tf * idf
        return tfidf_features

    def extract_features(self, tokens):
        """Extract features based on specified type."""
        features_dict = {}
        
        # First, generate all n-grams
        ngrams = self._get_ngrams(tokens)

        if 'bag_of_words' in self.feature_types:
            features_dict.update(self.extract_bag_of_words(ngrams))
        
        if 'term_frequency' in self.feature_types:
            features_dict.update(self.extract_tf_features(ngrams))

        if 'tfidf' in self.feature_types:
            features_dict.update(self.extract_tfidf_features(ngrams))
            
        if not features_dict:
             raise ValueError("Unsupported feature type specified")

        return features_dict
