#main_schedule_script.py

import pandas as pd
import requests
from datetime import datetime
import os

# Replace 'f1_analysis' with the actual path or way you import your analysis functions if it's part of a module
import f1_analysis

def fetch_data_from_api(url):
    response = requests.get(url)
    return response.json()

def get_or_refresh_data(file_path, url):
    if os.path.exists(file_path) and datetime.fromtimestamp(os.path.getmtime(file_path)).date() >= datetime.now().date():
        data = pd.read_csv(file_path)
    else:
        data_json = fetch_data_from_api(url)
        # Convert JSON to DataFrame as needed here, this is an example
        data = pd.DataFrame(data_json)
        data.to_csv(file_path, index=False)
    return data

def load_f1_schedule(year):
    schedule_path = f'cache/schedule_{year}.csv'
    schedule_url = f'http://ergast.com/api/f1/{year}.json'
    return get_or_refresh_data(schedule_path, schedule_url)

def load_detailed_data(year, data_types):
    base_url = f"http://ergast.com/api/f1/{year}/"
    data = {}
    for data_type in data_types:
        data_path = f'cache/{data_type}_{year}.csv'
        data_url = base_url + f"{data_type}.json"
        data[data_type] = get_or_refresh_data(data_path, data_url)
    return data

def is_race_week(race_date):
    race_date = datetime.strptime(race_date, '%Y-%m-%d')
    return datetime.now().isocalendar()[1] == race_date.isocalendar()[1]

def main():
    year = 2024
    schedule = load_f1_schedule(year)
    
    for index, row in schedule.iterrows():
        if is_race_week(row['date']):
            print(f"Race week for {row['raceName']} on {row['date']}. Fetching and analyzing data...")
            data_types = ['session_info', 'driver_info', 'session_status', 'track_status',
                          'extended_timing', 'timing_app', 'car_data', 'position_data',
                          'weather_data', 'race_control_messages', 'lap_count']
            detailed_data = load_detailed_data(year, data_types)
            f1_analysis.clean_and_analyze_data(detailed_data)
        else:
            print(f"No race this week for {row['raceName']} on {row['date']}.")

if __name__ == "__main__":
    main()
