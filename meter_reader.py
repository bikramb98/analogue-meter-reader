import cv2
import numpy as np
import math
import yaml

class MeterReader:
    def __init__(self, config_path='config.yaml'):
        self.config = self._load_config(config_path)

    def _load_config(self, config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def process_frame(self, frame):
        """Process a frame to detect and read the meter value."""
        if frame is None:
            raise ValueError("Invalid frame")
        
        # Create a copy for visualization
        debug_img = frame.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply GaussianBlur to reduce noise
        blur_config = self.config['gauge_detection']['gaussian_blur']
        blurred = cv2.GaussianBlur(gray, tuple(blur_config['kernel_size']), blur_config['sigma'])
        
        # Detect circles (the gauge)
        circle_config = self.config['gauge_detection']['hough_circles']
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=circle_config['dp'], 
            minDist=circle_config['min_dist'], 
            param1=circle_config['param1'], 
            param2=circle_config['param2'], 
            minRadius=circle_config['min_radius'], 
            maxRadius=circle_config['max_radius']
        )
        
        if circles is None:
            raise ValueError("No gauge detected in the image")
        
        # Get the largest circle (should be the gauge)
        circles = np.uint16(np.around(circles))
        circle = circles[0][0]  # Get the first and presumably largest circle
        center_x, center_y, radius = circle
        
        # Create a mask for the gauge
        mask = np.zeros_like(gray)
        cv2.circle(mask, (center_x, center_y), int(radius * 0.9), 255, -1)
        
        # Apply a threshold to detect the needle
        needle_config = self.config['needle_detection']
        _, thresh = cv2.threshold(blurred, needle_config['threshold_value'], 255, cv2.THRESH_BINARY_INV)
        
        # Apply Canny edge detection with the mask
        canny_config = needle_config['canny_edge']
        edges = cv2.Canny(thresh, canny_config['threshold1'], canny_config['threshold2'])
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        
        # Find lines using Hough Line Transform
        lines_config = needle_config['hough_lines']
        lines = cv2.HoughLinesP(
            edges, 
            rho=lines_config['rho'], 
            theta=np.pi/lines_config['theta_div'], 
            threshold=lines_config['threshold'], 
            minLineLength=int(radius * lines_config['min_line_length_ratio']),
            maxLineGap=lines_config['max_line_gap']
        )

        if lines is None or len(lines) == 0:
            raise ValueError("No lines detected that could be the needle")
        
        best_line = self._find_best_line(lines, center_x, center_y, radius)
        
        if best_line is None:
            return None

        return self._calculate_meter_value(best_line, center_x, center_y)

    def _find_best_line(self, lines, center_x, center_y, radius):
        """Find the line most likely to be the needle."""
        best_line = None
        best_score = 0
        
        center_proximity_ratio = self.config['gauge_calibration']['center_proximity_ratio']
        min_line_length = self.config['gauge_calibration']['min_line_length']
        
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
            
            if d1 < radius * center_proximity_ratio or d2 < radius * center_proximity_ratio:
                line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                if line_length > min_line_length:
                    score = line_length * (1 - min(d1, d2) / radius)
                    
                    if score > best_score:
                        best_score = score
                        best_line = [pivot_x, pivot_y, needle_x, needle_y]
        
        return best_line

    def _calculate_meter_value(self, best_line, center_x, center_y):
        """Calculate the meter value based on the needle position."""
        pivot_x, pivot_y, needle_x, needle_y = best_line
        
        # Calculate angle
        dx = needle_x - center_x
        dy = center_y - needle_y  # Y is inverted in image coordinates
        
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        # Adjust angle to be in 0-360 range
        if angle_deg < -48:
            angle_deg += 360

        # Get calibration values from config
        cal_config = self.config['gauge_calibration']
        first_reading_val = cal_config['first_reading']['value']
        first_reading_angle = cal_config['first_reading']['angle']
        second_reading_val = cal_config['second_reading']['value']
        second_reading_angle = cal_config['second_reading']['angle']

        # Calculate value
        value = np.abs(
            ((second_reading_val-first_reading_val)/(first_reading_angle-second_reading_angle))
            *(first_reading_angle-angle_deg)
        )

        return value 