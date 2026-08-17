# inspect_sheets.py
import pandas as pd

file_path = 'Part_A_Dataset.xlsx'

try:
    # Get sheet names
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    print(f"Sheet names: {sheet_names}")

    # Assuming the second sheet contains definitions
    if len(sheet_names) > 1:
        definitions_sheet_name = sheet_names[1]
        print(f"\nReading definitions from sheet: '{definitions_sheet_name}'")
        # Read the sheet, assuming definitions might not have headers
        df_definitions = pd.read_excel(file_path, sheet_name=definitions_sheet_name, header=None)
        print("\n--- Data Definitions ---")
        # Print the content row by row, assuming two columns [Variable, Description]
        for index, row in df_definitions.iterrows():
            variable = row.iloc[0] if len(row) > 0 else 'N/A'
            description = row.iloc[1] if len(row) > 1 else 'N/A'
            print(f"- {variable}: {description}")
        print("--- End Definitions ---")
    else:
        print("Only one sheet found. No separate definitions sheet detected.")

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")