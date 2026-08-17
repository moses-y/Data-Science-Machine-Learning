import os
from docx import Document
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from rake_nltk import Rake
import nltk
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Directory containing the docx files
transcriptions_directory = r'D:\Transcribe\001'  # Use raw string for the path

# List of additional stop words to ignore
additional_stopwords = {'speaker', 'yeah', 'c', 'b', 'okay', 'like'}

# Initialize RAKE with custom stop words
rake = Rake(stopwords=stopwords.words('english') + list(additional_stopwords))

# Function to read text from a docx file
def read_docx(file_path):
    doc = Document(file_path)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text

# Process each docx file
docx_files = [f for f in os.listdir(transcriptions_directory) if f.endswith('.docx')]

for file_name in docx_files:
    file_path = os.path.join(transcriptions_directory, file_name)
    text = read_docx(file_path)
    
    # Extract keywords using RAKE
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases()
    
    # Convert keywords list to a single string
    keyword_text = ' '.join(keywords)
    
    # Create and display a word cloud for keywords
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=additional_stopwords).generate(keyword_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Keyword Cloud of {file_name}')
    plt.show()
