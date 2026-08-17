import pandas as pd
import pytest

# Load the DataFrames from CSV files
weather_df = pd.read_csv('sampled_weather_df.csv')
field_df = pd.read_csv('sampled_field_df.csv')

# Test the shape of the weather DataFrame
def test_read_weather_DataFrame_shape():
    assert weather_df.shape[0] > 0, "weather_df should have more than 0 rows"
    assert weather_df.shape[1] > 0, "weather_df should have more than 0 columns"

# Test the shape of the field DataFrame
def test_read_field_DataFrame_shape():
    assert field_df.shape[0] > 0, "field_df should have more than 0 rows"
    assert field_df.shape[1] > 0, "field_df should have more than 0 columns"

# Test the columns of the weather DataFrame
def test_weather_DataFrame_columns():
    expected_columns = ['Weather_station_ID', 'Message', 'Measurement', 'Value']
    missing_columns = [col for col in expected_columns if col not in weather_df.columns]
    assert not missing_columns, f"Missing expected columns in weather_df: {missing_columns}"

# Test the columns of the field DataFrame
def test_field_DataFrame_columns():
    expected_columns = ['Field_ID', 'Elevation', 'Crop_type', 'Annual_yield']  # Corrected column names to match actual DataFrame
    missing_columns = [col for col in expected_columns if col not in field_df.columns]
    assert not missing_columns, f"Missing expected columns in field_df: {missing_columns}"

# Test for non-negative elevation values in the field DataFrame
def test_field_DataFrame_non_negative_elevation():
    assert (field_df['Elevation'] >= 0).all(), "Found negative elevation values in field_df"


# Test for valid crop types in the field DataFrame
def test_crop_types_are_valid():
    valid_crops = ['potato', 'banana', 'coffee', 'rice', 'maize', 'wheat', 'tea', 'cassava']  # Updated list of valid crops; ensure there are no trailing spaces in crop names
    invalid_crops = field_df[~field_df['Crop_type'].isin(valid_crops)]
    assert invalid_crops.empty, f"Found invalid crops in field_df: {invalid_crops['Crop_type'].unique().tolist()}"


# Test for positive rainfall values in the weather DataFrame
def test_positive_rainfall_values():
    # Filter for rows where 'Measurement' is 'Rainfall', then check 'Value'
    invalid_rainfall = weather_df[(weather_df['Measurement'] == 'Rainfall') & (weather_df['Value'] <= 0)]
    assert invalid_rainfall.empty, "Found non-positive rainfall values in weather_df"
