# streamlit_app.py

import streamlit as st
import pandas as pd
import fastf1
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page config
st.set_page_config(page_title='F1 2024 Dashboard', layout='wide')

# Enable FastF1 cache
fastf1.Cache.enable_cache('fastf1_cache')

st.title('🏎️ Formula 1 2024 Season Analysis Dashboard')

st.markdown("""
Welcome to the Formula 1 2024 Season Analysis Dashboard. Explore race data, compare drivers, and dive into telemetry charts!
""")

# Load schedule data
@st.cache_data
def load_schedule(year):
  schedule = fastf1.get_event_schedule(year, include_testing=False)
  return schedule

schedule = load_schedule(2024)

# Sidebar selections
st.sidebar.header('Select Options')

# Grand Prix selection
grand_prix_list = schedule['EventName'].unique()
selected_grand_prix = st.sidebar.selectbox('Grand Prix', grand_prix_list)

# Session selection
session_options = ['FP1', 'FP2', 'FP3', 'Q', 'R']
selected_session = st.sidebar.selectbox('Session', session_options)

# Load session data based on selection
@st.cache_data
def load_session(year, gp_name, session_name):
  session = fastf1.get_session(year, gp_name, session_name)
  session.load()
  return session

try:
  session = load_session(2024, selected_grand_prix, selected_session)
except Exception as e:
  st.error(f"Error loading session data: {e}")
  st.stop()

# Driver selection
driver_list = list(session.driver_names.values())
selected_drivers = st.sidebar.multiselect('Drivers', driver_list, default=driver_list[:2])

# Plot selection
plot_options = ['Lap Times', 'Telemetry Data', 'Speed Traces']
selected_plot = st.sidebar.selectbox('Plot Type', plot_options)

# Tab layout
tab1, tab2 = st.tabs(['📊 Data Visualization', '📝 Notes'])

with tab1:
  if selected_plot == 'Lap Times':
      st.subheader('Lap Times Comparison')
      fig = go.Figure()
      for driver in selected_drivers:
          driver_code = [code for code, name in session.driver_names.items() if name == driver][0]
          driver_laps = session.laps.pick_driver(driver_code)
          fig.add_trace(go.Scatter(
              x=driver_laps['LapNumber'],
              y=driver_laps['LapTime'].dt.total_seconds(),
              mode='lines+markers',
              name=driver
          ))
      fig.update_layout(
          xaxis_title='Lap Number',
          yaxis_title='Lap Time (s)',
          hovermode='x unified'
      )
      st.plotly_chart(fig, use_container_width=True)

  elif selected_plot == 'Telemetry Data':
      st.subheader('Telemetry Data Comparison')
      lap_option = st.sidebar.selectbox('Select Lap', ['Fastest Lap', 'Average Lap'])
      fig = go.Figure()
      for driver in selected_drivers:
          driver_code = [code for code, name in session.driver_names.items() if name == driver][0]
          if lap_option == 'Fastest Lap':
              lap = session.laps.pick_driver(driver_code).pick_fastest()
          else:
              lap = session.laps.pick_driver(driver_code).pick_average()
          telemetry = lap.get_car_data().add_distance()
          fig.add_trace(go.Scatter(
              x=telemetry['Distance'],
              y=telemetry['Speed'],
              mode='lines',
              name=driver
          ))
      fig.update_layout(
          xaxis_title='Distance (m)',
          yaxis_title='Speed (km/h)',
          hovermode='x unified'
      )
      st.plotly_chart(fig, use_container_width=True)

  elif selected_plot == 'Speed Traces':
      st.subheader('Speed Traces')
      for driver in selected_drivers:
          driver_code = [code for code, name in session.driver_names.items() if name == driver][0]
          fastest_lap = session.laps.pick_driver(driver_code).pick_fastest()
          telemetry = fastest_lap.get_car_data().add_distance()
          fig = px.line(
              telemetry,
              x='Distance',
              y='Speed',
              title=f'Speed Trace - {driver}',
              labels={'Distance': 'Distance (m)', 'Speed': 'Speed (km/h)'}
          )
          st.plotly_chart(fig, use_container_width=True)

with tab2:
  # Example notes dictionary
  race_notes = {
      'Bahrain Grand Prix': 'Bahrain saw an incredible start to the season with...',
      'Saudi Arabian Grand Prix': 'A stunning performance by...',
      # Add notes for each Grand Prix
  }

  st.header(f'📝 Notes on {selected_grand_prix}')
  note = race_notes.get(selected_grand_prix, 'No notes available for this race.')
  st.write(note)