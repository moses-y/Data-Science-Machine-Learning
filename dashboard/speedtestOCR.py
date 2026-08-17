import os
import configparser
import cv2
from pero_ocr.core.layout import PageLayout
from pero_ocr.document_ocr.page_parser import PageParser
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

class SpeedTestOCR:
    def __init__(self, image_path, config_path):
        self.image_path = image_path
        self.config_path = config_path
        self.page_layout = None

    def load_config(self):
        config = configparser.ConfigParser()
        config.read(self.config_path)
        return config

    def perform_ocr(self):
        try:
            # Load the config
            config = self.load_config()
            
            # Initialize the OCR pipeline
            page_parser = PageParser(config, config_path=os.path.dirname(self.config_path))
            
            # Read the image
            image = cv2.imread(self.image_path, 1)
            
            # Initialize an empty page layout
            self.page_layout = PageLayout(id=os.path.basename(self.image_path), 
                                          page_size=(image.shape[0], image.shape[1]))
            
            # Process the image with OCR
            self.page_layout = page_parser.process_page(image, self.page_layout)
            
            logging.info("OCR performed successfully.")
            return self.page_layout
        except Exception as e:
            logging.error("Error performing OCR: {}".format(e))
            return None

    def export_to_xml(self, output_path):
        if self.page_layout:
            self.page_layout.to_pagexml(output_path)
            logging.info(f"Exported Page XML to {output_path}.")
        else:
            logging.warning("No page layout to export.")

# Usage
image_path = 'speedtest.jpeg'
config_path = './config_file.ini'
output_xml_path = 'output_page.xml'

ocr_processor = SpeedTestOCR(image_path, config_path)
page_layout = ocr_processor.perform_ocr()
if page_layout:
    ocr_processor.export_to_xml(output_xml_path)
