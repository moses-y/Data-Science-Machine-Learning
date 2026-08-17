# field_data_processor.py

import pandas as pd
from data_ingestion import create_db_engine, query_data, read_from_web_CSV
import logging

# Configuration parameters for the FieldDataProcessor
config_params = {
    "sql_query": """
        SELECT *
        FROM geographic_features
        LEFT JOIN weather_features USING (Field_ID)
        LEFT JOIN soil_and_crop_features USING (Field_ID)
        LEFT JOIN farm_management_features USING (Field_ID)
    """,
    "db_path": "sqlite:///Maji_Ndogo_farm_survey_small.db",
    "columns_to_rename": {
        'Annual_yield': 'Crop_type', 
        'Crop_type': 'Annual_yield'
    },
    "values_to_rename": {
        'cassaval': 'cassava', 
        'wheatn': 'wheat', 
        'teaa': 'tea'
    },
    "weather_csv_path": "https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_station_data.csv",
    "weather_mapping_csv": "https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_data_field_mapping.csv",
}

class FieldDataProcessor:
    """
    Processes field data by loading, cleaning, and merging with weather station data.

    Attributes:
        db_path (str): Database path for SQL query.
        sql_query (str): SQL query for data ingestion.
        columns_to_rename (dict): Columns to be renamed.
        values_to_rename (dict): Crop type values to be renamed.
        weather_map_data (str): Path to weather station mapping CSV.
    """

    def __init__(self, config_params, logging_level="INFO"):
        self.db_path = config_params['db_path']
        self.sql_query = config_params['sql_query']
        self.columns_to_rename = config_params['columns_to_rename']
        self.values_to_rename = config_params['values_to_rename']
        self.weather_map_data = config_params['weather_mapping_csv']
        self.initialize_logging(logging_level)
        self.df = None
        self.engine = None

    def initialize_logging(self, logging_level):
        logger_name = __name__ + ".FieldDataProcessor"
        self.logger = logging.getLogger(logger_name)
        self.logger.propagate = False
        if logging_level.upper() == "DEBUG":
            log_level = logging.DEBUG
        elif logging_level.upper() == "INFO":
            log_level = logging.INFO
        elif logging_level.upper() == "NONE":
            self.logger.disabled = True
            return
        else:
            log_level = logging.INFO
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def ingest_sql_data(self):
        self.engine = create_db_engine(self.db_path)
        self.df = query_data(self.engine, self.sql_query)
        self.logger.info("Successfully loaded data.")

    
    #def rename_columns(self):
     #   column1, column2 = (
    #        list(self.columns_to_rename.keys())[0],
    #        list(self.columns_to_rename.values())[0],
    #    )
    #    temp_name = "__temp_name_for_swap__"
    #    while temp_name in self.df.columns:
    #        temp_name += "_"
    #    self.df = self.df.rename(columns={column1: temp_name, column2: column1})
    #    self.df = self.df.rename(columns={temp_name: column2})
    #    self.logger.info(f"Swapped columns: {column1} with {column2}")

    def rename_columns(self):
        # Log initial columns for verification
        self.logger.info(f"Initial columns: {self.df.columns.tolist()}")

        # Expected column names before renaming
        expected_columns = ['Annual_yield', 'Crop_type']
        
        # Check if the expected columns exist in the DataFrame
        missing_columns = [col for col in expected_columns if col not in self.df.columns]
        if missing_columns:
            self.logger.error(f"Missing columns before renaming: {missing_columns}")
            # Handle missing columns appropriately, e.g., return or raise an exception
            return
        
        # Perform the column renaming
        column1, column2 = self.columns_to_rename.keys()
        temp_name = "__temp_name_for_swap__"
        self.df.rename(columns={column1: temp_name, column2: column1}, inplace=True)
        self.df.rename(columns={temp_name: column2}, inplace=True)
        
        # Log final columns for verification
        self.logger.info(f"Columns after renaming: {self.df.columns.tolist()}")


    #def apply_corrections(self):
        # Ensure non-negative values in the 'Elevation' column
    #    self.df['Elevation'] = self.df['Elevation'].abs()
        
    #    # Apply renaming corrections for crop types
    #    self.df['Crop_type'] = self.df['Crop_type'].apply(lambda crop: self.values_to_rename.get(crop, crop))
        
        # Strip leading and trailing spaces from 'Crop_type' column values
    #    self.df['Crop_type'] = self.df['Crop_type'].str.strip()
        
    #    self.logger.info("Applied corrections to data.")

    def apply_corrections(self):
        # Log unique 'Crop_type' values before applying corrections
        unique_crops_before = self.df['Crop_type'].unique()
        self.logger.info(f"Unique 'Crop_type' values before corrections: {unique_crops_before}")

        # Ensure non-negative values in the 'Elevation' column
        self.df['Elevation'] = self.df['Elevation'].abs()
        
        # Apply renaming corrections for crop types
        self.df['Crop_type'] = self.df['Crop_type'].apply(lambda crop: self.values_to_rename.get(crop.strip(), crop.strip()))
        
        # Strip leading and trailing spaces from 'Crop_type' column values
        # Note: This is now incorporated into the line above to strip spaces before applying renaming corrections
        
        # Log unique 'Crop_type' values after applying corrections
        unique_crops_after = self.df['Crop_type'].unique()
        self.logger.info(f"Unique 'Crop_type' values after corrections: {unique_crops_after}")

    def weather_station_mapping(self):
    # Fetch weather station mapping data
        weather_map_df = read_from_web_CSV(self.weather_map_data)
    
    # Merge the fetched data with the main DataFrame if it's already loaded
        if self.df is not None:
            self.df = self.df.merge(weather_map_df, on='Field_ID', how='left')
            # self.logger.info("Weather station data integrated successfully.")
        return weather_map_df
    
    def process(self):
        self.ingest_sql_data()
        #Insert your code here
        self.rename_columns()  # Step 2: Rename columns as needed
        self.apply_corrections()  # Step 3: Apply any necessary corrections to the data
        self.weather_station_mapping()
        #self.logger.info("Data processing complete.")  # Log the completion of the process
