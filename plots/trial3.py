import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the DataFrames with the provided heads
routes = pd.DataFrame({'route_id': [1, 2, 3, 4, 5],
                       'agency_id': ['A1', 'A2', 'A1', 'A3', 'A2'],
                       'route_short_name': ['R1', 'R2', 'R3', 'R4', 'R5'],
                       'route_long_name': ['Route 1', 'Route 2', 'Route 3', 'Route 4', 'Route 5'],
                       'route_desc': ['Description 1', 'Description 2', 'Description 3', 'Description 4', 'Description 5'],
                       'route_type': [0, 1, 1, 2, 2],
                       'route_url': ['url1', 'url2', 'url3', 'url4', 'url5'],
                       'route_color': ['red', 'blue', 'green', 'yellow', 'orange'],
                       'route_text_color': ['white', 'black', 'white', 'black', 'white']})

trips = pd.DataFrame({'route_id': [1, 2, 3, 4, 5],
                      'service_id': ['S1', 'S2', 'S1', 'S3', 'S2'],
                      'trip_id': [101, 102, 103, 104, 105],
                      'trip_headsign': ['Headsign 1', 'Headsign 2', 'Headsign 3', 'Headsign 4', 'Headsign 5'],
                      'direction_id': [0, 1, 0, 1, 0],
                      'block_id': ['B1', 'B2', 'B3', 'B4', 'B5'],
                      'shape_id': [201, 202, 203, 204, 205]})

stops = pd.DataFrame({'stop_id': [1, 2, 3, 4, 5],
                      'stop_code': ['SC1', 'SC2', 'SC3', 'SC4', 'SC5'],
                      'stop_name': ['Stop 1', 'Stop 2', 'Stop 3', 'Stop 4', 'Stop 5'],
                      'stop_desc': ['Description 1', 'Description 2', 'Description 3', 'Description 4', 'Description 5'],
                      'stop_lat': [1.0, 2.0, 3.0, 4.0, 5.0],
                      'stop_lon': [10.0, 20.0, 30.0, 40.0, 50.0],
                      'zone_id': ['Z1', 'Z2', 'Z1', 'Z3', 'Z2'],
                      'stop_url': ['url1', 'url2', 'url3', 'url4', 'url5'],
                      'location_type': [0, 1, 0, 1, 0],
                      'parent_station': ['PS1', 'PS2', 'PS3', 'PS4', 'PS5']})

stop_times = pd.DataFrame({'trip_id': [101, 102, 103, 104, 105],
                           'arrival_time': ['10:00', '11:00', '12:00', '13:00', '14:00'],
                           'departure_time': ['10:05', '11:05', '12:05', '13:05', '14:05'],
                           'stop_id': [1, 2, 3, 4, 5],
                           'stop_sequence': [1, 2, 3, 4, 5],
                           'stop_headsign': ['Headsign 1', 'Headsign 2', 'Headsign 3', 'Headsign 4', 'Headsign 5'],
                           'pickup_type': [0, 1, 0, 1, 0],
                           'shape_dist_traveled': [0.0, 1.0, 2.0, 3.0, 4.0]})

shapes = pd.DataFrame({'shape_id': [201, 202, 203, 204, 205],
                       'shape_pt_lat': [1.0, 2.0, 3.0, 4.0, 5.0],
                       'shape_pt_lon': [10.0, 20.0, 30.0, 40.0, 50.0],
                       'shape_pt_sequence': [1, 2, 3, 4, 5],
                       'shape_dist_traveled': [0.0, 1.0, 2.0, 3.0, 4.0]})

calendar = pd.DataFrame({'service_id': ['S1', 'S2', 'S3', 'S4', 'S5'],
                         'monday': [1, 1, 0, 1, 0],
                         'tuesday': [1, 0, 1, 0, 1],
                         'wednesday': [0, 1, 0, 1, 0],
                         'thursday': [1, 0, 1, 0, 1],
                         'friday': [0, 1, 0, 1, 0],
                         'saturday': [1, 0, 1, 0, 1],
                         'sunday': [0, 1, 0, 1, 0],
                         'start_date': ['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01'],
                         'end_date': ['2023-12-31', '2023-12-31', '2023-12-31', '2023-12-31', '2023-12-31']})

frequencies = pd.DataFrame({'trip_id': [101, 102, 103, 104, 105],
                            'start_time': ['09:00', '10:00', '11:00', '12:00', '13:00'],
                            'end_time': ['10:00', '11:00', '12:00', '13:00', '14:00'],
                            'headway_secs': [600, 600, 600, 600, 600]})

agency = pd.DataFrame({'agency_id': ['A1', 'A2', 'A3'],
                       'agency_name': ['Agency 1', 'Agency 2', 'Agency 3'],
                       'agency_url': ['url1', 'url2', 'url3'],
                       'agency_timezone': ['Timezone 1', 'Timezone 2', 'Timezone 3'],
                       'agency_lang': ['Lang 1', 'Lang 2', 'Lang 3'],
                       'agency_phone': ['Phone 1', 'Phone 2', 'Phone 3'],
                       'agency_fare_url': ['fare_url1', 'fare_url2', 'fare_url3']})

calendar_dates = pd.DataFrame({'service_id': ['S1', 'S2', 'S3'],
                               'date': ['2023-01-01', '2023-02-01', '2023-03-01'],
                               'exception_type': [1, 2, 1]})

feed_info = pd.DataFrame({'feed_publisher_name': ['Publisher 1', 'Publisher 2', 'Publisher 3'],
                          'feed_publisher_url': ['publisher_url1', 'publisher_url2', 'publisher_url3'],
                          'feed_lang': ['Lang 1', 'Lang 2', 'Lang 3'],
                          'feed_start_date': ['2023-01-01', '2023-01-01', '2023-01-01'],
                          'feed_end_date': ['2023-12-31', '2023-12-31', '2023-12-31'],
                          'feed_version': ['Version 1', 'Version 2', 'Version 3']})


# Plot 1: Categorical Plot (Count)
sns.countplot(x='route_type', data=routes)
plt.xlabel('Route Type')
plt.ylabel('Count')
plt.title('Distribution of Route Types')
plt.show()

# Plot 2: Histogram
sns.histplot(stop_times['stop_sequence'], bins=20)
plt.xlabel('Stop Sequence')
plt.ylabel('Frequency')
plt.title('Distribution of Stop Sequences')
plt.show()



# Plot 4: Pie Chart
weekday_columns = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
weekday_counts = calendar[weekday_columns].sum()

plt.pie(weekday_counts, labels=weekday_columns, autopct='%1.1f%%')
plt.title('Distribution of Weekdays')
plt.axis('equal')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ... (Code to load the DataFrames)

# Exclude non-numeric columns from 'trips' DataFrame
numeric_columns = trips.select_dtypes(include=[float, int]).columns
numeric_trips = trips[numeric_columns]

# Plot 5: Correlation Heatmap
correlation_matrix = numeric_trips.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()

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


