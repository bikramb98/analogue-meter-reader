import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
from io import BytesIO

# Placeholder function for meter reading conversion
def process_frame(frame):
    # Simulating meter reading detection
    detected_reading = np.random.uniform(10, 100)  # Replace with actual CV model output
    return round(detected_reading, 2)

# Load predefined video
video_path = "record1.mp4"  # Ensure this video exists in the working directory
cap = cv2.VideoCapture(video_path)

# UI Layout
st.title("Analog to Digital Meter Reading using Computer Vision")
st.write("This tool converts analogue meter readings into digital readings in real-time using computer vision.")

# Session state to control video play/pause
if "playing" not in st.session_state:
    st.session_state.playing = False
if "frame_position" not in st.session_state:
    st.session_state.frame_position = 0

# Play and Pause Buttons
col1, col2 = st.columns(2)
if col1.button("Play Video"):
    st.session_state.playing = True
    # Set video to the stored position
    print(f"position -- > {st.session_state.frame_position}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_position)

if col2.button("Pause Video"):
    st.session_state.playing = False
    # Store current position
    current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    # st.session_state.frame_position = current_pos
    st.session_state.frame_position = st.session_state.current_position
    print(f"paused position --> {st.session_state.frame_position}")
    # cap.release()  # Release the current capture
    # cap = cv2.VideoCapture(video_path)  # Reinitialize the capture

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

# Main loop for video playback
while st.session_state.playing and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.session_state.frame_position = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
        continue
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.session_state.last_frame = frame  # Store the last frame
    frame_placeholder.image(frame, channels="RGB")

    # Store current position
    current_playing_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    st.session_state.current_position = current_playing_pos
    print(f'current_playing pos --> {current_playing_pos}')
    
    # Process frame to get meter reading
    reading = process_frame(frame)
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.data.append({"Time": timestamp, "Meter Reading": reading})

    data.append({"Time": timestamp, "Meter Reading": reading})
    
    # Update table
    df = pd.DataFrame(data[-10:])  # Show last 10 readings
    table_placeholder.dataframe(df)
    
    # time.sleep(0.1)

# Persist last frame and table when paused
# Display last frame and table when paused
if st.session_state.last_frame is not None:
    frame_placeholder.image(st.session_state.last_frame, channels="RGB")

if st.session_state.data:
    df = pd.DataFrame(st.session_state.data[-10:])
    table_placeholder.dataframe(df)

# Download Button for Data
if data:
    df_all = pd.DataFrame(data)
    csv_buffer = BytesIO()
    df_all.to_csv(csv_buffer, index=False)
    st.download_button("Download Readings CSV", csv_buffer.getvalue(), "meter_readings.csv", "text/csv")
