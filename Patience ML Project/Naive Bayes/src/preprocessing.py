import re

class AdvancedTextPreprocessor:
    """
    Handles advanced text preprocessing for sentiment analysis,
    including negation handling.
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
        self.negation_words = {'not', 'no', 'never', "n't"}
        self.punctuation = {'.', '!', '?', ';', ','}

    def clean_text(self, text):
        """Clean and preprocess text."""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s\']', '', text) # Keep apostrophes for contractions
        text = ' '.join(text.split())
        return text

    def tokenize(self, text):
        """Tokenize text into words."""
        return text.split()

    def handle_negation(self, tokens):
        """Add NOT_ prefix to words following a negation word."""
        negated_tokens = []
        negation_active = False
        for token in tokens:
            if token in self.negation_words:
                negation_active = True
            elif any(punct in token for punct in self.punctuation):
                negation_active = False
            
            if negation_active and token not in self.negation_words:
                negated_tokens.append(f"NOT_{token}")
            else:
                negated_tokens.append(token)
        return negated_tokens

    def remove_stop_words(self, tokens):
        """Remove stop words from token list."""
        # Keep negation words in the text as they are important for sentiment
        return [token for token in tokens if token not in self.stop_words or token in self.negation_words]

    def preprocess(self, text):
        """Complete preprocessing pipeline."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens_no_stopwords = self.remove_stop_words(tokens)
        negated_tokens = self.handle_negation(tokens_no_stopwords)
        return negated_tokens
