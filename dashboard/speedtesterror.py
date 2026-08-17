
from speedtestOCR import SpeedTestOCR  # Assuming this is the name of your Python file
import streamlit as st
import pandas as pd
import plotly.express as px

import re
import pandas as pd
from PIL import Image
import pytesseract
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

class SpeedTestOCR:
    def __init__(self, image_path):
        self.image_path = image_path
        self.columns = ['Interval', 'Transfer', 'Bitrate', 'Jitter', 'Lost/Total Datagrams', 'Direction']
        self.data = []
        self.df = pd.DataFrame()

    def perform_ocr(self):
        try:
            image = Image.open(self.image_path)
            text = pytesseract.image_to_string(image)
            logging.info("OCR performed successfully.")
            return text
        except Exception as e:
            logging.error("Error performing OCR: {}".format(e))
            return ""

    def parse_ocr_text(self, text):
        pattern = re.compile(r'(\d+.\d+-\d+.\d+ sec)\s+(\d+ MBytes)\s+(\d+ Mbits/sec)\s+(\d+.\d+ ms)\s+(\d+/\d+ \(\d+%\))\s+(sender|receiver)')
        matches = pattern.finditer(text)
        self.data = [match.groups() for match in matches]
        logging.info("Text parsed successfully.")

    def load_to_dataframe(self):
        if self.data:
            self.df = pd.DataFrame(self.data, columns=self.columns)
            logging.info("Data loaded into DataFrame successfully.")
        else:
            logging.info("No data to load into DataFrame.")

    def save_to_csv(self, csv_file_path):
        if not self.df.empty:
            self.df.to_csv(csv_file_path, index=False)
            logging.info(f"DataFrame saved to CSV at {csv_file_path}.")
        else:
            logging.info("DataFrame is empty. No CSV file created.")

# Usage
image_path = 'speedtest.jpeg'
csv_file_path = 'RawOCRspeedtestfile.csv'

ocr_processor = SpeedTestOCR(image_path)
ocr_text = ocr_processor.perform_ocr()
ocr_processor.parse_ocr_text(ocr_text)
ocr_processor.load_to_dataframe()
ocr_processor.save_to_csv(csv_file_path)


