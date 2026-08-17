import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
import os
from docx import Document
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Directory containing the docx files
transcriptions_directory = r'D:\Transcribe\001'  # Use raw string for the path

# Function to read text from a docx file
def read_docx(file_path):
    doc = Document(file_path)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text

# Combine text from all docx files
combined_text = ""
docx_files = [f for f in os.listdir(transcriptions_directory) if f.endswith('.docx')]

for file_name in docx_files:
    file_path = os.path.join(transcriptions_directory, file_name)
    combined_text += read_docx(file_path) + " "

# Create and display a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Transcriptions')
plt.show()
