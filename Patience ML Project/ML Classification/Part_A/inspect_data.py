# inspect_data.py
import pandas as pd

# Define the file path
file_path = 'Part_A_Dataset.xlsx'

try:
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Display the first 5 rows
    print("First 5 rows of the dataset:")
    print(df.head())

    # Display column names and data types
    print("\nColumn names and data types:")
    print(df.info())

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")