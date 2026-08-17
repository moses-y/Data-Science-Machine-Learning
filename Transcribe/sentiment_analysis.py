import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from docx import Document

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Directory containing the docx files
transcriptions_directory = r'D:\Transcribe\001'  # Use raw string for the path

# Function to perform sentiment analysis
def analyze_sentiment(text):
    scores = sid.polarity_scores(text)
    return scores

# Summary file to hold sentiment results for all documents
summary_file_path = os.path.join(transcriptions_directory, 'sentiment_analysis_summary.txt')

# Process each transcription file
docx_files = [f for f in os.listdir(transcriptions_directory) if f.endswith('.docx')]

# Open the summary file in write mode
with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
    summary_file.write("Sentiment Analysis Summary for All Documents\n\n")

    for file_name in docx_files:
        file_path = os.path.join(transcriptions_directory, file_name)
        print(f"Analyzing sentiment for file: {file_path}")
        
        try:
            # Read the content of the docx file
            doc = Document(file_path)
            text = '\n'.join([para.text for para in doc.paragraphs])
            
            # Perform sentiment analysis
            sentiment_scores = analyze_sentiment(text)

            # Append the results to the summary file
            summary_file.write(f"Sentiment analysis for {file_name}\n")
            summary_file.write(f"Overall sentiment scores: {sentiment_scores}\n\n")

            print(f"Appended sentiment analysis for {file_name} to summary file")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

print("Finished analyzing sentiment for all files.")
print(f"Sentiment analysis summary saved to {summary_file_path}")
