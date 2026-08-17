

import os
import pandas as pd
import fastf1
from fastf1 import get_session, plotting
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import datetime
import json
import time

# Set working directory
os.chdir('C:/Users/moses_y/OneDrive/Desktop/ML Projects/F1 Trial')

# Setup cache directory for FastF1
CACHE_PATH = r"C:\Users\moses_y\OneDrive\Desktop\ML Projects\F1 Trial\fastf1_cache"
if not os.path.exists(CACHE_PATH):
  os.makedirs(CACHE_PATH)
fastf1.Cache.enable_cache(CACHE_PATH)
plotting.setup_mpl()

# Define a cache file to store data
CACHE_FILE = 'f1_data_cache.json'

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

# Load data from cache or fetch and cache it
def get_f1_data():
  if os.path.exists(CACHE_FILE):
      with open(CACHE_FILE, 'r') as cache:
          data = json.load(cache)
          # Check if the data is stale (1 hour old)
          if time.time() - data['timestamp'] < 3600:
              return data['f1_data']
  
  # Fetch new data and cache it
  f1_data = fetch_ergast_data(f"current.json")
  with open(CACHE_FILE, 'w') as cache:
      json.dump({'timestamp': time.time(), 'f1_data': f1_data}, cache)
  return f1_data

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
              title=f'{", ".join(drivers)} Fastest Laps Comparison - {year} {grand_prix} GP',
              xaxis=dict(title='Time [s]', tickmode='array', tickvals=tickvals_x, ticktext=ticktext_x),
              yaxis=dict(title='Speed [Km/h]', tickmode='array', tickvals=list(range(0, 360, 20))),
              template='plotly_dark'
          )
          fig.show()
      else:
          plt.figure(figsize=(20, 10))
          max_time = 0

          for driver in drivers:
              fastest_lap = session.laps.pick_driver(driver).pick_fastest()
              if fastest_lap is not None and not fastest_lap.empty:
                  car_data = fastest_lap.get_car_data()
                  times = car_data['Time']
                  times = (times - times.iloc[0]).dt.total_seconds()  # Convert to seconds
                  max_time = max(max_time, max(times))
                  speeds = car_data['Speed']
                  lap_time_str = format_timedelta(fastest_lap['LapTime'].total_seconds())
                  plt.plot(times, speeds, label=f'{driver} Fastest {session_name} Lap ({lap_time_str})')

          tickvals_x = range(0, int(max_time) + 10, 10)
          plt.xticks(tickvals_x, [format_timedelta(x) for x in tickvals_x])
          plt.yticks(range(0, 360, 20))

          plt.xlabel('Time [s]')
          plt.ylabel('Speed [Km/h]')
          plt.title(f'{session_name} Lap Speed Comparison - {year} {grand_prix} GP')
          plt.legend()
          plt.grid(True, which='both', linestyle='--', linewidth=0.5)
          plt.tight_layout()
          plt.show()

# Define a function for plotting lap times using seaborn
def plot_lap_times(data, highlight_drivers, session):
  session_data = data[data['session'] == session]
  highlight_session_data = highlight_drivers[highlight_drivers['session'] == session]

  plt.figure()
  sns.lineplot(data=session_data, x='lap_number', y='lap_time', hue='driver', palette='gray', legend=False)
  sns.lineplot(data=highlight_session_data, x='lap_number', y='lap_time', hue='driver', 
               palette=['blue', 'red', 'green'], legend='full')

  plt.title(f'{session} Lap Times')
  plt.xlabel('Lap Number')
  plt.ylabel('Lap Time (seconds)')
  plt.legend(title='Driver')
  plt.gca().yaxis.set_minor_locator(plt.AutoMinorLocator())
  plt.grid(True, which='both', linestyle='--', linewidth=0.5)
  plt.tight_layout()
  plt.show()

# Define a function for plotting sector times using seaborn
def plot_sector_times(data, highlight_drivers, session):
  qual_data = data[data['session'] == session]
  highlight_qual_data = highlight_drivers[highlight_drivers['session'] == session]

  sectors = ['sector1', 'sector2', 'sector3']
  plt.figure(figsize=(15, 10))
  fig, axes = plt.subplots(3, 1, sharex=True)

  for i, sector in enumerate(sectors):
      ax = axes[i]
      sns.lineplot(data=qual_data, x='lap_number', y=sector, hue='driver', palette='gray', legend=False, ax=ax)
      sns.lineplot(data=highlight_qual_data, x='lap_number', y=sector, hue='driver', 
                   palette=['blue', 'red', 'green'], legend=False, ax=ax)
      
      ax.set_ylabel(f'{sector.capitalize()} Time (seconds)')
      ax.grid(True, which='both', linestyle='--', linewidth=0.5)
      ax.yaxis.set_minor_locator(plt.AutoMinorLocator())

  axes[-1].set_xlabel('Lap Number')
  axes[0].set_title(f'{session} Sector Times')
  plt.tight_layout()
  plt.show()

# Define a function for plotting time delta using seaborn
def plot_time_delta(data, highlight_drivers, session):
  qual_data = highlight_drivers[highlight_drivers['session'] == session]
  qual_data['time_delta'] = qual_data.groupby('driver')['lap_time'].diff()

  plt.figure(figsize=(15, 6))
  sns.lineplot(data=qual_data, x='lap_number', y='time_delta', hue='driver', 
               palette=['blue', 'red', 'green'], legend='full')

  plt.title(f'{session} Time Delta for Top Drivers')
  plt.xlabel('Lap Number')
  plt.ylabel('Time Delta (seconds)')
  plt.legend(title='Driver')
  plt.gca().yaxis.set_minor_locator(plt.AutoMinorLocator())
  plt.grid(True, which='both', linestyle='--', linewidth=0.5)
  plt.tight_layout()
  plt.show()

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

                  # Load and process data for seaborn plots
                  data = get_f1_data()
                  df = pd.json_normalize(data['MRData']['RaceTable']['Races'])
                  top_drivers = ['Piastri', 'Sainz', 'Leclerc']
                  highlight_drivers = df[df['driver'].isin(top_drivers)]

                  # Plot using seaborn
                  for session in ['FP1', 'FP2', 'FP3', 'Q']:
                      plot_lap_times(df, highlight_drivers, session)
                      if session == 'Q':
                          plot_sector_times(df, highlight_drivers, session)
                          plot_time_delta(df, highlight_drivers, session)
              except Exception as e:
                  print(f"Error plotting data for {grand_prix}: {e}")

if __name__ == "__main__":
  main()