import cv2
import torch
import time
import numpy as np
import mysql.connector
from datetime import datetime
from flask import Flask, Response, render_template
import pytesseract
import os

app = Flask(__name__)

# Load models only once
VEHICLE_MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 for vehicle detection
PLATE_DETECTOR_MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Placeholder for plate detector
PLATE_OCR_MODEL = pytesseract  # Using Tesseract for OCR

# Configure MySQL connection
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'traffic_violations'
}

class TrafficDetection:
    def __init__(self, video_path, speed_limit=60, meters_per_pixel=0.01):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open video source.")
        
        self.speed_limit = speed_limit
        self.meters_per_pixel = meters_per_pixel
        self.prev_frame_time = None
        self.vehicle_positions = {}
        self.signal_status = "red"
        self.reference_line = 300  # Reference line for red-light violations
        self.fps = 0
        self.current_time = ""
        os.makedirs("static/images", exist_ok=True)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

    def insert_violation(self, license_number, image_path, violation_type, speed, timestamp):
        if image_path.startswith('static/images/'):
            image_path = image_path.replace('static/images/', '')
        db = mysql.connector.connect(**db_config)
        cursor = db.cursor()
        query = """
        INSERT INTO nlcn_violations (vt_license_number, vt_images, vt_violations, vt_speed, vt_time)
        VALUES (%s, %s, %s, %s, %s)
        """
        values = (license_number, image_path, violation_type, speed, timestamp)
        cursor.execute(query, values)
        db.commit()
        db.close()  # Always close the connection after executing the query
        
    def read_license_plate(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        license_number = pytesseract.image_to_string(gray, config='--psm 8')
        return license_number.strip()
    
    def detect_license_plate(self, image):
        results = PLATE_DETECTOR_MODEL(image)  # Using YOLOv5 for plate detection
        plate_number = ""
        for *xyxy, conf, cls in results.xyxy[0]:
            if conf > 0.5 and int(cls) == 0:  # License plate class is 0
                x1, y1, x2, y2 = map(int, xyxy)
                cropped = image[y1:y2, x1:x2]
                plate_number = self.read_license_plate(cropped)  # Using OCR to extract plate number
        return plate_number

    def handle_violation(self, frame, license_number, speed):
        if speed > self.speed_limit:
            img_path = f"static/images/{license_number}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            cv2.imwrite(img_path, frame)
            violation_time = datetime.now()
            self.insert_violation(license_number, img_path, "Speeding", speed, violation_time)
        
    def handle_red_light_violation(self, frame, license_number):
        img_path = f"static/images/{license_number}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        cv2.imwrite(img_path, frame)
        violation_time = datetime.now()
        self.insert_violation(license_number, img_path, "Red Light Violation", 0, violation_time)
    
    def calculate_speed(self, prev_position, current_position, frame_rate, meters_per_pixel):
        """
        Tính tốc độ của xe (km/h) dựa trên khoảng cách di chuyển và thời gian giữa các khung hình.
        prev_position, current_position: tọa độ của xe trong các khung hình liên tiếp.
        frame_rate: tần suất khung hình (fps).
        meters_per_pixel: số mét tương ứng với một pixel.
        """
        # Tính khoảng cách Euclidean giữa hai vị trí
        distance_pixels = np.sqrt((current_position[0] - prev_position[0])**2 + (current_position[1] - prev_position[1])**2)
        
        # Chuyển khoảng cách từ pixel sang mét
        distance_meters = distance_pixels * meters_per_pixel
        
        # Tính thời gian giữa các khung hình (thời gian giữa hai frame)
        time_seconds = 1 / frame_rate  # Thời gian giữa hai khung hình (giây)
        
        # Tính tốc độ (m/s)
        speed_mps = distance_meters / time_seconds
        
        # Chuyển từ m/s sang km/h
        speed = speed_mps * 3.6
        
        return speed


    def generate_frames(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (820, 500))
            current_time = time.time()
            self.fps = 1 / (current_time - self.prev_frame_time) if self.prev_frame_time else 0
            self.prev_frame_time = current_time
            self.current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Traffic light simulation (green/yellow/red)
            if int(current_time) % 93 < 30:
                self.signal_status = "green"
                cv2.putText(frame, "GREEN LIGHT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif int(current_time) % 93 < 33:
                self.signal_status = "yellow"
                cv2.putText(frame, "YELLOW LIGHT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                self.signal_status = "red"
                cv2.putText(frame, "RED LIGHT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw the reference line for red-light violations
            cv2.line(frame, (20, self.reference_line), (frame.shape[1], self.reference_line), (0, 255, 0), 2)
            vehicle_results = VEHICLE_MODEL(frame)
            detected_vehicles = []

            for *xyxy, conf, cls in vehicle_results.xyxy[0]:
                if conf > 0.5 and int(cls) == 2:  # Vehicle class is 2
                    x1, y1, x2, y2 = map(int, xyxy)
                    vehicle_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    detected_vehicles.append((vehicle_center, x1, y1, x2, y2))

                    # Draw bounding box around vehicle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Detect license plate number and speed
                    plate_number = self.detect_license_plate(frame[y1:y2, x1:x2]) or "unknow"
                    speed = 0

                    if vehicle_center in self.vehicle_positions:
                        prev_time, prev_position = self.vehicle_positions[vehicle_center]
                        speed = self.calculate_speed(prev_position, (x1, y1), self.fps, 0.00245)

                        if speed > self.speed_limit:
                            violation_image = frame[y1:y2, x1:x2]
                            cv2.putText(frame, f"Speed: {int(speed)} km/h", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            self.handle_violation(violation_image, plate_number or "unknow", speed)

                    # Display plate number and speed on the frame
                    cv2.putText(frame, f"Plate: {plate_number}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    if speed > 0:
                        cv2.putText(frame, f"Speed: {int(speed)} km/h", (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    self.vehicle_positions[vehicle_center] = (current_time, (x1, y1))

                    # Check for red light violation
                    if self.signal_status == "red" and y2 > self.reference_line:
                        violation_image = frame[y1:y2, x1:x2]
                        self.handle_red_light_violation(violation_image, plate_number or "unknow")
            
            # Show video feed
            cv2.imshow('Traffic Detection', frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    db = mysql.connector.connect(**db_config)
    cursor = db.cursor()
    cursor.execute("SELECT * FROM nlcn_violations")
    violations = cursor.fetchall()
    db.close()
    
    return render_template('index.html', violations=violations)

@app.route('/video_feed')
def video_feed():
    video_path = "./static/videos/test_6.mp4"  # Replace with your video source
    traffic_detector = TrafficDetection(video_path)
    return Response(traffic_detector.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
