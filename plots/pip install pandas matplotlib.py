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
stop_times = pd.read_csv(data_folder + 'stop_times.txt')
stops = pd.read_csv(data_folder + 'stops.txt')
trips = pd.read_csv(data_folder + 'trips.txt')

# Explore the data and perform necessary data manipulations
# For example, you can print the head of a DataFrame to check the data
print(routes.head())

# Create plots or perform any analysis using the loaded data
# For example, you can create a bar plot of the route types
route_type_counts = routes['route_type'].value_counts()
plt.bar(route_type_counts.index, route_type_counts.values)
plt.xlabel('Route Type')
plt.ylabel('Count')
plt.title('Distribution of Route Types')
plt.show()


sns.countplot(x='route_type', data=routes)
plt.xlabel('Route Type')
plt.ylabel('Count')
plt.title('Distribution of Route Types')
plt.show()


sns.distplot(trips['trip_length'], kde=False)
plt.xlabel('Trip Length')
plt.ylabel('Frequency')
plt.title('Distribution of Trip Lengths')
plt.show()


sns.barplot(x='route_type', y='trip_length', data=trips)
plt.xlabel('Route Type')
plt.ylabel('Trip Length')
plt.title('Average Trip Length by Route Type')
plt.show()

# Calculate counts for each route type
route_type_counts = routes['route_type'].value_counts()

# Plotting a pie chart
plt.pie(route_type_counts, labels=route_type_counts.index, autopct='%1.1f%%')
plt.title('Distribution of Route Types')
plt.axis('equal')
plt.show()

# Create a correlation matrix
correlation_matrix = trips.corr()

# Plotting the correlation matrix as a heatmap
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()
