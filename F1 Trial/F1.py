# F1.py

import os
import pandas as pd
import fastf1
from fastf1 import get_session, plotting
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import requests
from datetime import datetime

# Setup cache directory for FastF1
CACHE_PATH = r"C:\Users\moses_y\OneDrive\Desktop\ML Projects\F1 Trial\fastf1_cache"
if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH)
fastf1.Cache.enable_cache(CACHE_PATH)
plotting.setup_mpl()

# Function to fetch data from Ergast API
def fetch_ergast_data(endpoint):
    url = f"https://ergast.com/api/f1/{endpoint}.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Failed to fetch data: {e}")
        return None

# Fetch F1 schedule for the current year
def fetch_schedule(year):
    return fetch_ergast_data(f"{year}")

# Load and cache data using FastF1
def fetch_and_cache_session_data(year, grand_prix, session_type):
    try:
        session = get_session(year, grand_prix, session_type)
        session.load()
        return session
    except Exception as e:
        print(f"Error loading data for {grand_prix} - {session_type}: {e}")
        return None

# Format timedelta to a readable string
def format_timedelta(seconds):
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes}:{seconds:06.3f}"

# Clean and process data for plotting
def process_car_data(car_data):
    times = car_data['Time']
    times = (times - times.iloc[0]).dt.total_seconds()  # Convert to seconds
    speeds = car_data['Speed']
    return times, speeds

def plot_fastest_laps(year, grand_prix, sessions, drivers, use_plotly=True):
    for session_type, session_name in sessions:
        session = fetch_and_cache_session_data(year, grand_prix, session_type)
        if not session:
            continue

        if use_plotly:
            fig = go.Figure()
            max_time = 0
            for driver in drivers:
                fastest_lap = session.laps.pick_driver(driver).pick_fastest()
                if fastest_lap is not None and not fastest_lap.empty:
                    car_data = fastest_lap.get_car_data()
                    times = [t.total_seconds() for t in car_data['Time']]
                    max_time = max(max_time, max(times))
                    speeds = car_data['Speed']
                    lap_time = format_timedelta(fastest_lap['LapTime'].total_seconds())
                    fig.add_trace(go.Scatter(x=times, y=speeds, mode='lines', name=f'{driver} {session_name} Fastest Lap ({lap_time})'))

            # Define tick values for the x-axis
            tickvals_x = [i for i in range(0, int(max_time) + 10, 10)]
            ticktext_x = [format_timedelta(i) for i in tickvals_x]

            fig.update_layout(
                title=f'{", ".join(drivers)} Fastest {session_name} Laps - {year} {grand_prix} GP',
                xaxis=dict(title='Time [s]', tickmode='array', tickvals=tickvals_x, ticktext=ticktext_x),
                yaxis=dict(title='Speed [Km/h]'),
                template='plotly_dark'
            )
            fig.show()
        else:
            def plot_with_matplotlib(session, session_name, driver_codes):
                figsize = (20, 10)
                fig, ax = plt.subplots(figsize=figsize)
                max_time = 0
                for driver_code in driver_codes:
                    fast_lap = session.laps.pick_driver(driver_code).pick_fastest()
                    if fast_lap is not None and not fast_lap.empty:
                        car_data = fast_lap.get_car_data()
                        times = car_data['Time']
                        times = (times - times.iloc[0]).dt.total_seconds()  # Convert to seconds
                        max_time = max(max_time, max(times))
                        speeds = car_data['Speed']
                        lap_time_str = format_timedelta(fast_lap['LapTime'].total_seconds())
                        ax.plot(times, speeds, label=f'{driver_code} Fastest {session_name} Lap ({lap_time_str})')

                tickvals_x = range(0, int(max_time) + 10, 10)
                ax.set_xticks(tickvals_x)
                ax.set_xticklabels([format_timedelta(x) for x in tickvals_x])
                ax.set_yticks(range(0, 360, 20))
                
                ax.set_xlabel('Time [s]')
                ax.set_ylabel('Speed [Km/h]')
                ax.set_title(f'{", ".join(driver_codes)} Fastest {session_name} Lap Speed Comparison - {year} {grand_prix} GP')
                ax.legend()
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.tight_layout()
                plt.show()

            plot_with_matplotlib(session, session_name, drivers)

# Main function to fetch data and plot
def main():
    year = 2024
    schedule = fetch_schedule(year)
    
    if schedule:
        for race in schedule['MRData']['RaceTable']['Races']:
            grand_prix = race['raceName']
            race_date = datetime.strptime(race['date'], '%Y-%m-%d')
            
            if race_date <= datetime.now():
                print(f"Plotting data for: {grand_prix} on {race['date']}")
                try:
                    sessions = [
                        ('FP1', 'Free Practice 1'), 
                        ('FP2', 'Free Practice 2'), 
                        ('FP3', 'Free Practice 3'), 
                        ('Q', 'Qualifying'), 
                        ('R', 'Race'),
                        ('S', 'Sprint') # Removed 'S' session type due to its absence
                    ]
                    plot_fastest_laps(year, grand_prix, sessions, ['LEC', 'SAI'], use_plotly=True)
                    plot_fastest_laps(year, grand_prix, sessions, ['LEC', 'SAI'], use_plotly=False)
                except Exception as e:
                    print(f"Error plotting data for {grand_prix}: {e}")

if __name__ == "__main__":
    main()

