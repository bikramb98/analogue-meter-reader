import cv2
import numpy as np
import math
import pandas as pd

angle_list = []
value_list = []

def read_pressure_gauge_advanced(frame, min_value=0, max_value=30, 
                                min_angle=225, max_angle=315, debug=False):
    """
    A more customizable version of the pressure gauge reader that allows
    specifying the gauge's minimum and maximum values and angles.
    
    Args:
        image_path (str): Path to the image file
        min_value (float): Minimum value on the gauge scale
        max_value (float): Maximum value on the gauge scale
        min_angle (float): Angle in degrees where min_value is located
        max_angle (float): Angle in degrees where max_value is located
        debug (bool): If True, show debug visualizations
        
    Returns:
        float: The value where the needle is pointing
    """

    # angle_list = []
    # value_list = []

    # Read the image
    img = frame
    if img is None:
        raise ValueError(f"Invalid frame")
    
    # Create a copy for visualization
    debug_img = img.copy() if debug else None
    
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
    
    if debug:
        # Draw the detected circle
        cv2.circle(debug_img, (center_x, center_y), radius, (0, 255, 0), 2)
        cv2.circle(debug_img, (center_x, center_y), 2, (0, 0, 255), 3)
    
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

                print(f"line length: {line_length}")
            
                # Higher score for longer lines with one end near center
                score = line_length * (1 - min(d1, d2) / radius)
                
                if score > best_score:
                    best_score = score
                    best_line = [pivot_x, pivot_y, needle_x, needle_y]
                    print("best line detected")
    
    if best_line is None:
        raise ValueError("Could not identify the needle")
    
    pivot_x, pivot_y, needle_x, needle_y = best_line
    
    if debug:
        # Draw the detected needle
        cv2.line(debug_img, (pivot_x, pivot_y), (needle_x, needle_y), (0, 0, 255), 2)
    
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

    angle_list.append(angle_deg)

    first_reading_val = 0
    first_reading_angle = 225
    second_reading_val = 10
    second_reading_angle = 136

    degree_per_val = (first_reading_angle - second_reading_angle) / (second_reading_val - first_reading_val)

    # print(degree_per_val)

    value = ((second_reading_val-first_reading_val)/(first_reading_angle-second_reading_angle))*(first_reading_angle-angle_deg)

    value_list.append(value)

    if converted == True:

        print(f"angle converted: {converted}, value = {value}")

    else:

        print(f"angle : {angle_deg}, value = {value}")

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
    
    if debug:
        # Display angle and value information
        cv2.putText(
            debug_img, 
            f"Angle: {angle_deg:.1f}°", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 0, 0), 
            2
        )
        
        if value is not None:
            cv2.putText(
                debug_img, 
                f"Value: {value:.1f}", 
                (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 0, 0), 
                2
            )
        else:
            cv2.putText(
                debug_img, 
                "Value: Out of range", 
                (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 0, 0), 
                2
            )
        
        # Display the debug image
        cv2.imshow("Gauge Reader Debug", debug_img)
        cv2.waitKey(1)
        # cv2.destroyAllWindows()

        # print("---")
    
    return value, angle_list, value_list

# def main():
    # Example usage
    # image_path = "meter_images/meter3.png"  # Change to your image path
    
    # Method 1: Basic gauge reader with default settings
    # value = read_pressure_gauge(image_path, debug=True)
    # print(f"Gauge value: {value:.1f}")
    
    # Method 2: Advanced gauge reader with custom settings
    # Customize these values based on your specific gauge
    # value = read_pressure_gauge_advanced(
    #     image_path,
    #     min_value=0,    # Minimum value on gauge
    #     max_value=30,   # Maximum value on gauge
    #     min_angle=225,  # Angle where minimum value is located (degrees)
    #     max_angle=315,  # Angle where maximum value is located (degrees)
    #     debug=True
    # )

    #     value = read_pressure_gauge_advanced(
    #     image_path,
    #     min_value=0,    # Minimum value on gauge
    #     max_value=30,   # Maximum value on gauge
    #     min_angle=170,  # Angle where minimum value is located (degrees)
    #     max_angle=396,  # Angle where maximum value is located (degrees)
    #     debug=True
    # )



    # value = read_pressure_gauge_advanced(
    #     image_path,
    #     min_value=0,    # Minimum value on gauge
    #     max_value=30,   # Maximum value on gauge
    #     min_angle=225.23,  # Angle where minimum value is located (degrees)
    #     max_angle=402,  # Angle where maximum value is located (degrees)
    #     debug=True
    # )
    # print(f"Advanced gauge value: {value:.1f}")

def main():
    # Video capture setup
    cap = cv2.VideoCapture('record2.mp4')
    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream")
            break

        try:
            value, angle_list, value_list = read_pressure_gauge_advanced(
                frame,
                min_value=0,
                max_value=30,
                min_angle=225.23,
                max_angle=402,
                debug=True
            )
            print(f"Current gauge value: {value:.1f}")
        except Exception as e:
            print(f"Error processing frame: {e}")

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    print(angle_list)
    print(value_list)
    print(len(value_list))

    df = pd.DataFrame()
    df['angle'] = angle_list
    df['value'] = value_list
    df.to_csv('debug_df.csv')

if __name__ == "__main__":
    main()