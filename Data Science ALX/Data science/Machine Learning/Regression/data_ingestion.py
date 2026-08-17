from sqlalchemy import create_engine, text
import logging
import pandas as pd
# Name our logger so we know that logs from this module come from the data_ingestion module
logger = logging.getLogger('data_ingestion')
# Set a basic logging message up that prints out a timestamp, the name of our logger, and the message
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


### START FUNCTION

def create_db_engine(db_path):
    """
    Creates and tests a database engine connection.
    
    Parameters:
    - db_path (str): Database connection URL.
    
    Returns:
    - engine: A SQLAlchemy engine instance.
    
    Raises:
    - ImportError: If SQLAlchemy is not installed.
    - Exception: If connection to the database fails.
    """
    try:
        engine = create_engine(db_path)
        # Test connection
        with engine.connect():
            pass
        # test if the database engine was created successfully
        logger.info("Database engine created successfully.")
        return engine # Return the engine object if it all works well
    except ImportError: #If we get an ImportError, inform the user SQLAlchemy is not installed
        logger.error("SQLAlchemy is required to use this function. Please install it first.")
        raise e  # noqa: F821
    except Exception as e: # If we fail to create an engine inform the user
        logger.error(f"Failed to create database engine. Error: {e}")
        raise e
    
def query_data(engine, sql_query):
    """
    Executes an SQL query and returns the results as a DataFrame.
    
    Parameters:
    - engine: SQLAlchemy engine object.
    - sql_query (str): SQL query string.
    
    Returns:
    - df (DataFrame): Query results.
    
    Raises:
    - ValueError: If the query returns an empty DataFrame.
    - Exception: For other failures during the query execution.
    """
    try:
        with engine.connect() as connection:
            df = pd.read_sql_query(text(sql_query), connection)
        if df.empty:
            # Log a message or handle the empty DataFrame scenario as needed
            msg = "The query returned an empty DataFrame."
            logger.error(msg)
            raise ValueError(msg)
        logger.info("Query executed successfully.")
        return df
    except ValueError as e: 
        logger.error(f"SQL query failed. Error: {e}")
        raise e
    except Exception as e:
        logger.error(f"An error occurred while querying the database. Error: {e}")
        raise e
    
def read_from_web_CSV(URL):
    """
    Reads a CSV file from a web URL into a DataFrame.
    
    Parameters:
    - URL (str): The web URL pointing to a CSV file.
    
    Returns:
    - df (DataFrame): Data read from the CSV file.
    
    Raises:
    - pd.errors.EmptyDataError: If the URL does not point to a valid CSV file.
    - Exception: For other failures during CSV file reading.
    """
    try:
        df = pd.read_csv(URL)
        logger.info("CSV file read successfully from the web.")
        return df
    except pd.errors.EmptyDataError as e:
        logger.error("The URL does not point to a valid CSV file. Please check the URL and try again.")
        raise e
    except Exception as e:
        logger.error(f"Failed to read CSV from the web. Error: {e}")
        raise e
    
### END FUNCTION