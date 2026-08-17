import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to the data folder
data_folder = 'C:/Users/moses_y/OneDrive/Desktop/plots/data/'

# Load the GTFS files into DataFrames
agency = pd.read_csv(data_folder + 'agency.txt')
calendar = pd.read_csv(data_folder + 'calendar.txt')
calendar_dates = pd.read_csv(data_folder + 'calendar_dates.txt')
feed_info = pd.read_csv(data_folder + 'feed_info.txt')
frequencies = pd.read_csv(data_folder + 'frequencies.txt')
routes = pd.read_csv(data_folder + 'routes.txt')
shapes = pd.read_csv(data_folder + 'shapes.txt')
stops = pd.read_csv(data_folder + 'stops.txt')
trips = pd.read_csv(data_folder + 'trips.txt')

# Merge 'trips' and 'frequencies' DataFrames on 'trip_id'
trips = trips.merge(frequencies, on='trip_id')

# Convert 'start_time' and 'end_time' columns to datetime type
trips['start_time'] = pd.to_datetime(trips['start_time'], format='%H:%M:%S')
trips['end_time'] = pd.to_datetime(trips['end_time'], format='%H:%M:%S')

# Check if 'trip_duration' column exists in the DataFrame
if 'trip_duration' not in trips.columns:
    # Calculate trip duration
    trips['trip_duration'] = trips['end_time'] - trips['start_time']

# Convert trip_duration to numeric format
trips['trip_duration'] = trips['trip_duration'].dt.total_seconds() / 60

# Filter out invalid or missing data
trips = trips.dropna(subset=['route_id', 'trip_duration'])

# Check if any valid trip durations exist
if trips['trip_duration'].empty:
    print("No valid trip durations found.")
else:
    # Calculate average trip duration
    average_trip_duration = trips['trip_duration'].mean()
    print("Average Trip Duration:", average_trip_duration)

    # Plot the bar chart for average trip duration by route ID
    plt.figure(figsize=(16, 8))  # Adjust the figure size
    ax = sns.barplot(x='route_id', y='trip_duration', data=trips)
    plt.xlabel('Route ID')
    plt.ylabel('Trip Duration (minutes)')
    plt.title(f'Average Trip Duration by Route ID\n(Average: {average_trip_duration:.2f} minutes)')  # Include average in the plot title
    plt.xticks(rotation=60)  # Rotate the x-axis labels by 30 degrees
    plt.tight_layout()  # Adjust the spacing of the plot

    # Adjust the x-axis labels to start at 0
    ax.set_xlim(left=0)
    plt.ylim(0, None)

    plt.show()
