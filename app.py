import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
from io import BytesIO
import math
import plotly.graph_objects as go

def update_plot(data, placeholder):
    df_all = pd.DataFrame(data)
    marker_colors = ['red' if val > 25 else 'blue' for val in df_all["Meter Reading"]]
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


# Placeholder function for meter reading conversion
def process_frame(frame):

    # Read the image
    img = frame
    if img is None:
        raise ValueError(f"Invalid frame")
    
    # Create a copy for visualization
    debug_img = img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # Detect circles (the gauge)
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=100, 
        param1=50, 
        param2=30, 
        minRadius=180, 
        maxRadius=205
    )
    
    if circles is None:
        raise ValueError("No gauge detected in the image")
    
    # Get the largest circle (should be the gauge)
    circles = np.uint16(np.around(circles))
    circle = circles[0][0]  # Get the first and presumably largest circle
    center_x, center_y, radius = circle

    # print(f"radius: {radius}")
    
    # if debug:
    #     # Draw the detected circle
    #     cv2.circle(debug_img, (center_x, center_y), radius, (0, 255, 0), 2)
    #     cv2.circle(debug_img, (center_x, center_y), 2, (0, 0, 255), 3)
    
    # Create a mask for the gauge
    mask = np.zeros_like(gray)
    cv2.circle(mask, (center_x, center_y), int(radius * 0.9), 255, -1)
    
    # Apply a threshold to detect the needle (darker than background)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

    # if debug:
    # cv2.imshow("threshold", thresh)
    # cv2.waitKey(1)
    
    # Apply Canny edge detection with the mask
    edges = cv2.Canny(thresh, 50, 150)
    edges = cv2.bitwise_and(edges, edges, mask=mask)
    
    # if debug:
    #     cv2.imshow("Edges", edges)
    #     cv2.waitKey(1)
    
    # Find lines using Hough Line Transform
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=30, 
        minLineLength=int(radius * 0.3),  # Needle should be a significant portion of radius
        maxLineGap=10
    )

    # print(len(lines))

    # Create a copy of the original image for line visualization
    # all_lines_img = img.copy()
    
    # Draw all detected lines in blue
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(all_lines_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Show the image with all detected lines
    # cv2.imshow("All Detected Lines", all_lines_img)
    # cv2.waitKey(1)
    
    if lines is None or len(lines) == 0:
        raise ValueError("No lines detected that could be the needle")
    
# Find which line is most likely to be the needle
    best_line = None
    best_score = 0
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Determine which end is closer to the center
        d1 = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
        d2 = np.sqrt((x2 - center_x)**2 + (y2 - center_y)**2)
        
        if d1 < d2:
            pivot_x, pivot_y = x1, y1
            needle_x, needle_y = x2, y2
        else:
            pivot_x, pivot_y = x2, y2
            needle_x, needle_y = x1, y1
        
        # The pivot should be close to the center, and the tip should be near the edge
        if d1 < radius * 0.3 or d2 < radius * 0.3:
            # Calculate line length
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if line_length > 90:

                # print(f"line length: {line_length}")
            
                # Higher score for longer lines with one end near center
                score = line_length * (1 - min(d1, d2) / radius)
                
                if score > best_score:
                    best_score = score
                    best_line = [pivot_x, pivot_y, needle_x, needle_y]
                    # print("best line detected")
    
    if best_line is None:
        pass
        # raise ValueError("Could not identify the needle")

    else:
    
        pivot_x, pivot_y, needle_x, needle_y = best_line
        
        # if debug:
        #     # Draw the detected needle
        #     cv2.line(debug_img, (pivot_x, pivot_y), (needle_x, needle_y), (0, 0, 255), 2)
        
        # Calculate angle
        dx = needle_x - center_x
        dy = center_y - needle_y  # Y is inverted in image coordinates
        
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        converted = False
        
        # Adjust angle to be in 0-360 range
        # if angle_deg < 0 and angle_deg < -48:

        #     converted = True

        #     angle_deg += 360

        # Adjust angle to be in 0-360 range
        if angle_deg < -48:
            angle_deg += 360

        # print(f"angle degrres: {angle_deg}")

        # angle_list.append(angle_deg)

        first_reading_val = 0
        first_reading_angle = 225
        second_reading_val = 10
        second_reading_angle = 136

        degree_per_val = (first_reading_angle - second_reading_angle) / (second_reading_val - first_reading_val)

        # print(degree_per_val)

        value = ((second_reading_val-first_reading_val)/(first_reading_angle-second_reading_angle))*(first_reading_angle-angle_deg)

        print(value)

        # value_list.append(value)

        # if converted == True:

        #     print(f"angle converted: {converted}, value = {value}")

        # else:

        #     print(f"angle : {angle_deg}, value = {value}")

        # Calculate gauge value based on angle
        # Handle the case where min_angle > max_angle (gauge wraps around)
        # if min_angle > max_angle:
        #     if angle_deg >= min_angle or angle_deg <= max_angle:
        #         # Normalize the angle to the 0-1 range
        #         if angle_deg >= min_angle:
        #             normalized_angle = (angle_deg - min_angle) / (360 - min_angle + max_angle)
        #         else:
        #             normalized_angle = (angle_deg + 360 - min_angle) / (360 - min_angle + max_angle)
                
        #         value = min_value + normalized_angle * (max_value - min_value)
        #     else:
        #         # Needle is outside the expected range
        #         value = None
        # else:
        #     # Normal case where min_angle < max_angle
        #     if min_angle <= angle_deg <= max_angle:
        #         normalized_angle = (angle_deg - min_angle) / (max_angle - min_angle)
        #         value = min_value + normalized_angle * (max_value - min_value)
        #     else:
        #         # Needle is outside the expected range
        #         if angle_deg < min_angle:
        #             value = min_value
        #         else:
        #             value = max_value
        
        # if debug:
        #     # Display angle and value information
        #     cv2.putText(
        #         debug_img, 
        #         f"Angle: {angle_deg:.1f}°", 
        #         (10, 30), 
        #         cv2.FONT_HERSHEY_SIMPLEX, 
        #         0.7, 
        #         (255, 0, 0), 
        #         2
        #     )
            
        #     if value is not None:
        #         cv2.putText(
        #             debug_img, 
        #             f"Value: {value:.1f}", 
        #             (10, 60), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 
        #             0.7, 
        #             (255, 0, 0), 
        #             2
        #         )
        #     else:
        #         cv2.putText(
        #             debug_img, 
        #             "Value: Out of range", 
        #             (10, 60), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 
        #             0.7, 
        #             (255, 0, 0), 
        #             2
        #         )
            
        #     # Display the debug image
        #     cv2.imshow("Gauge Reader Debug", debug_img)
        #     cv2.waitKey(1)
            # cv2.destroyAllWindows()
        # Simulating meter reading detection
        # detected_reading = np.random.uniform(10, 100)  # Replace with actual CV model output
        # return round(detected_reading, 2)

        return value

# Load predefined video
video_path = "record1.mp4"  # Ensure this video exists in the working directory
cap = cv2.VideoCapture(video_path)

# UI Layout
st.title("Analog to Digital Meter Reading using Computer Vision")
st.write("This tool converts analogue meter readings into digital readings in real-time using computer vision.")

# Session state to control video play/pause
# if "playing" not in st.session_state:
#     st.session_state.playing = False
# if "frame_position" not in st.session_state:
#     st.session_state.frame_position = 0

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
    print(f"position -- > {st.session_state.frame_position}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_position)

if col2.button("Pause Video"):
    st.session_state.playing = False
    st.session_state.frame_position = st.session_state.current_position
    # Store the current frame as bytes
    if st.session_state.last_frame is not None:
        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(st.session_state.last_frame, cv2.COLOR_RGB2BGR))
        if is_success:
            st.session_state.last_frame_bytes = buffer.tobytes()
    print(f"Paused at position --> {st.session_state.frame_position}")

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
    print(f'current_playing pos --> {current_playing_pos}')
    
    # Process frame to get meter reading
    reading = process_frame(frame)
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.data.append({"Time": timestamp, "Meter Reading": reading})

    data.append({"Time": timestamp, "Meter Reading": reading})
    
    # Update table
    df = pd.DataFrame(data[-10:])  # Show last 10 readings
    table_placeholder.dataframe(df)

    # Update plot
    # df_all = pd.DataFrame(st.session_state.data)
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(
    #     x=list(range(len(df_all))),  # Using index as x-axis
    #     y=df_all["Meter Reading"],
    #     mode='lines+markers',
    #     name='Meter Reading'
    # ))
    # fig.update_layout(
    #     title='Meter Reading Over Time',
    #     xaxis_title='Reading Number',
    #     yaxis_title='Meter Value',
    #     showlegend=True,
    #     height=400,
    #     margin=dict(l=20, r=20, t=40, b=20)
    # )
    # plot_placeholder.plotly_chart(fig, use_container_width=True)

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
