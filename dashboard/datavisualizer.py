import pytesseract
from PIL import Image
import pandas as pd
import re
import logging

logging.basicConfig(level=logging.INFO)

class DataVisualizer:
    def __init__(self, image_path, csv_path):
        self.image_path = image_path
        self.csv_path = csv_path
        self.pattern = r'(\d+.\d+-\d+.\d+ sec)\s+(\d+ MBytes)\s+(\d+ Mbits/sec)\s+(\d+.\d+ ms)\s+(\d+/\d+ \(\d+%\))\s+(sender|receiver)'
        self.columns = ['Interval', 'Transfer', 'Bitrate', 'Jitter', 'Lost/Total Datagrams', 'Direction']
        self.data = []
        self.df = pd.DataFrame()

    def ocr_to_dataframe(self):
        try:
            # Perform OCR on the image
            image = Image.open(self.image_path)
            text = pytesseract.image_to_string(image)
            logging.info("OCR performed successfully.")
            
            # Parse the extracted text
            matches = re.finditer(self.pattern, text)
            self.data = [match.groups() for match in matches]
            
            # Validate and load the data to a DataFrame
            self.df = pd.DataFrame(self.data, columns=self.columns)
            logging.info("Data loaded into DataFrame successfully.")

            # Save the DataFrame to a CSV file
            if not self.df.empty:
                self.df.to_csv(self.csv_path, index=False)
                logging.info(f"DataFrame saved to CSV at {self.csv_path}.")
            else:
                logging.warning("No data to load into DataFrame. CSV not created.")

        except FileNotFoundError:
            logging.error(f"The file {self.image_path} does not exist.")
        except Exception as e:
            logging.error(f"An error occurred: {e}")

# Usage
visualizer = DataVisualizer("speedtest.jpeg", 'RawOCRspeedtestfile.csv')
visualizer.ocr_to_dataframe()
