import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
from io import BytesIO
import math
import plotly.graph_objects as go
import yaml
from meter_reader import MeterReader

# Load configuration
def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

# Load config at startup
config = load_config()

def update_plot(data, placeholder):
    df_all = pd.DataFrame(data)
    marker_colors = ['red' if val > config['anomaly_threshold'] else 'blue' for val in df_all["Meter Reading"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(df_all))),
        y=df_all["Meter Reading"],
        mode='lines+markers',
        marker=dict(color=marker_colors),
        name='Meter Reading'
    ))
    fig.update_layout(
        title='Meter Reading Over Time',
        xaxis_title='Reading Number',
        yaxis_title='Meter Value',
        showlegend=True,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    placeholder.plotly_chart(fig, use_container_width=True)

# Initialize MeterReader
meter_reader = MeterReader()


# Load predefined video
video_path = "record1.mp4"  # Ensure this video exists in the working directory
cap = cv2.VideoCapture(video_path)

# UI Layout
st.title("Analogue to Digital Meter Reading using Computer Vision")
st.write("This tool converts analogue meter readings into digital readings in real-time using computer vision.")

if "playing" not in st.session_state:
    st.session_state.playing = False
if "frame_position" not in st.session_state:
    st.session_state.frame_position = 0
if "current_position" not in st.session_state:
    st.session_state.current_position = 0
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None
if "data" not in st.session_state:
    st.session_state.data = []
if "last_frame_bytes" not in st.session_state:  # Add this line
    st.session_state.last_frame_bytes = None

# Play and Pause Buttons
col1, col2 = st.columns(2)
if col1.button("Play Video"):
    st.session_state.playing = True
    # Set video to the stored position
    # print(f"position -- > {st.session_state.frame_position}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_position)

if col2.button("Pause Video"):
    st.session_state.playing = False
    st.session_state.frame_position = st.session_state.current_position
    # Store the current frame as bytes
    if st.session_state.last_frame is not None:
        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(st.session_state.last_frame, cv2.COLOR_RGB2BGR))
        if is_success:
            st.session_state.last_frame_bytes = buffer.tobytes()
    # print(f"Paused at position --> {st.session_state.frame_position}")

# DataFrame to store readings
data = []

# Session state to store last frame and data
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None
if "data" not in st.session_state:
    st.session_state.data = []

# Video and Table Layout
video_col, table_col = st.columns(2)
with video_col:
    st.subheader("Meter Video")
    frame_placeholder = st.empty()

with table_col:
    st.subheader("Live Meter Readings")
    table_placeholder = st.empty()

st.subheader("Meter Reading Time Series")
plot_placeholder = st.empty()

# Main loop for video playback
while st.session_state.playing and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.session_state.frame_position = 0
        st.session_state.current_position = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
        continue
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.session_state.last_frame = frame  # Store the last frame
        # Store frame as bytes
    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if is_success:
        st.session_state.last_frame_bytes = buffer.tobytes()
    frame_placeholder.image(frame, channels="RGB")

    # Store current position
    current_playing_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    st.session_state.current_position = current_playing_pos
    # print(f'current_playing pos --> {current_playing_pos}')
    
    # Process frame to get meter reading
    reading = meter_reader.process_frame(frame)
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.data.append({"Time": timestamp, "Meter Reading": reading})

    data.append({"Time": timestamp, "Meter Reading": reading})
    
    # Update table
    df = pd.DataFrame(data[-10:])  # Show last 10 readings
    table_placeholder.dataframe(df)

    update_plot(st.session_state.data, plot_placeholder)

    # time.sleep(0.01)

# Persist last frame and table when paused
# Display last frame and table when paused
# Display last frame when paused
if st.session_state.last_frame_bytes is not None:
    try:
        # Convert bytes back to image and display
        nparr = np.frombuffer(st.session_state.last_frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")
    except Exception as e:
        print(f"Error displaying last frame: {e}")
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.session_state.last_frame = frame
                is_success, buffer = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if is_success:
                    st.session_state.last_frame_bytes = buffer.tobytes()
                frame_placeholder.image(frame, channels="RGB")

if st.session_state.data:
    df = pd.DataFrame(st.session_state.data[-10:])
    table_placeholder.dataframe(df)
    update_plot(st.session_state.data, plot_placeholder)


# Download Button for Data
if data:
    df_all = pd.DataFrame(data)
    csv_buffer = BytesIO()
    df_all.to_csv(csv_buffer, index=False)
    st.download_button("Download Readings CSV", csv_buffer.getvalue(), "meter_readings.csv", "text/csv")