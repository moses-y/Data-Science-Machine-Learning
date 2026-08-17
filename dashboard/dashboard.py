import streamlit as st
import pandas as pd
import plotly.express as px

# Load the dataset
df = pd.read_csv('RawOCRspeedtestfile.csv')

# Convert necessary columns to numeric if they are not already
df['Bitrate'] = pd.to_numeric(df['Bitrate'].str.replace(' Mbits/sec', ''), errors='coerce')
df['Jitter'] = pd.to_numeric(df['Jitter'].str.replace(' ms', ''), errors='coerce')
df['Packet Loss'] = df['Lost/Total Datagrams'].str.extract(r'(\d+)%')[0].astype('float') / 100

# Streamlit app
st.title('Speed Test Data Dashboard')

# Bitrate over time chart
st.header('Bitrate Over Time')
bitrate_fig = px.line(df, x='Interval', y='Bitrate', title='Bitrate Over Time', markers=True)
st.plotly_chart(bitrate_fig)

# Packet loss chart
st.header('Packet Loss Over Time')
packet_loss_fig = px.bar(df, x='Interval', y='Packet Loss', title='Packet Loss Over Time')
st.plotly_chart(packet_loss_fig)

# Jitter chart
st.header('Jitter Over Time')
jitter_fig = px.scatter(df, x='Interval', y='Jitter', title='Jitter Over Time', size='Jitter', color='Jitter')
st.plotly_chart(jitter_fig)

# Transfer Distribution
st.header('Transfer Distribution')
transfer_fig = px.histogram(df, x='Transfer', title='Transfer Distribution')
st.plotly_chart(transfer_fig)

# Direction Proportion
st.header('Direction Proportion')
direction_fig = px.pie(df, names='Direction', title='Proportion of Direction')
st.plotly_chart(direction_fig)

# Save this code in a file, for example, `dashboard.py`
# Run the Streamlit app using the command `streamlit run dashboard.py`
