import cv2
import torch
import time
import numpy as np
import mysql.connector
from datetime import datetime
from flask import Flask, Response, render_template
from flask_socketio import SocketIO, emit
import pytesseract
import os

app = Flask(__name__)
socketio = SocketIO(app)
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
        
    def preprocess_plate_image(self, image):
        """
        Preprocess image for better OCR accuracy.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def read_license_plate(self, image):
        """
        Read license plate using OCR.
        """
        preprocessed_image = self.preprocess_plate_image(image)
        license_number = pytesseract.image_to_string(preprocessed_image, config='--psm 8')
        return ''.join(filter(str.isalnum, license_number.strip()))  # Remove invalid characters
    
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
            self.insert_violation(license_number, img_path, "Speeding", float(speed), violation_time)
        
    def handle_red_light_violation(self, frame, license_number, speed):
        img_path = f"static/images/{license_number}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        cv2.imwrite(img_path, frame)
        violation_time = datetime.now()
        self.insert_violation(license_number, img_path, "Red Light Violation", float(speed), violation_time)
    
    def calculate_speed(self, prev_position, current_position, frame_rate, meters_per_pixel):
        """
        Calculate the speed of a vehicle (km/h).
        """
        # Ensure positions are valid
        if not prev_position or not current_position:
            return 0

        # Euclidean distance in pixels
        distance_pixels = np.sqrt((current_position[0] - prev_position[0])**2 +
                                (current_position[1] - prev_position[1])**2)
        # Convert to meters
        distance_meters = distance_pixels * meters_per_pixel

        # Time interval between frames
        time_seconds = 1 / frame_rate

        # Speed in m/s
        speed_mps = distance_meters / time_seconds

        # Convert to km/h
        speed_kmh = speed_mps * 3.6
        return speed_kmh





    def generate_frames(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (820, 500))
            current_time = time.time()
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if not self.fps or self.fps <= 0:
                self.fps = 30  # Default FPS if unavailable

            
            self.prev_frame_time = current_time
            self.current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Traffic light simulation (green/yellow/red)
            cv2.putText(frame, "Traffic Signal: ", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if int(current_time) % 93 < 30:
                self.signal_status = "green"
                cv2.putText(frame, "GREEN", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif int(current_time) % 93 < 33:
                self.signal_status = "yellow"
                cv2.putText(frame, "YELLOW", (200,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                self.signal_status = "red"
                cv2.putText(frame, "RED", (190, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"FPS: {int(self.fps)}", (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
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
                    plate_number = self.detect_license_plate(frame[y1:y2, x1:x2])
                    speed = 0

                    if vehicle_center in self.vehicle_positions:
                        prev_time, prev_position = self.vehicle_positions[vehicle_center]
                        speed = self.calculate_speed(prev_position, (x1, y1), self.meters_per_pixel)

                        if speed >= self.speed_limit:
                            violation_image = frame[y1:y2, x1:x2]
                            cv2.putText(frame, f"S: {float(speed)} km/h", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            self.handle_violation(violation_image, plate_number, float(speed)*10000)

                    # Display plate number and speed on the frame
                    cv2.putText(frame, f"P: {plate_number}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, f"S: {speed} km/h", (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    self.vehicle_positions[vehicle_center] = (current_time, (x1, y1))

                    # Check for red light violation
                    if self.signal_status == "red" and y2 > self.reference_line:
                        violation_image = frame[y1:y2, x1:x2]
                        self.handle_red_light_violation(violation_image, plate_number, float(speed)*10000)
            
            # Show video feed
            cv2.imshow('Traffic Detection', frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    db = mysql.connector.connect(**db_config)
    cursor = db.cursor()
    cursor.execute("SELECT * FROM nlcn_violations ORDER BY vt_id DESC")
    violations = cursor.fetchall()
    db.close()
    
    return render_template('index.html', violations=violations)
def update_violations():
    while True:
        violations = get_violations()
        socketio.emit('update_violations', violations)
        time.sleep(2)  # Update every 5 seconds
        
@app.route('/video_feed')
def video_feed():
    video_path = "./static/videos/test_8.mp4"  # Replace with your video source
    traffic_detector = TrafficDetection(video_path)
    return Response(traffic_detector.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    print("Client connected")
    
if __name__ == '__main__':
    app.run(debug=True)