import cv2
import os
import torch
import time
import numpy as np
from datetime import datetime
import mysql.connector
from flask import Flask, Response, render_template, url_for
import function.utils_rotate as utils_rotate
import function.helper as helper
import logging

app = Flask(__name__)

# Load models only once
VEHICLE_MODEL = torch.hub.load('yolov5', 'custom', path='model/Vehicle_detect_1.pt', force_reload=True, source='local')
PLATE_DETECTOR_MODEL = torch.hub.load('yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True, source='local')
PLATE_OCR_MODEL = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')

# Configure logging
logging.basicConfig(level=logging.INFO)

class TrafficDetection:
    def __init__(self, video_path, speed_limit=60, db_config=None, meters_per_pixel=0.01):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open video source.")

        # Initialize variables
        self.speed_limit = speed_limit
        self.meters_per_pixel = meters_per_pixel
        self.prev_frame_time = None
        self.vehicle_positions = {}
        self.db_config = db_config
        self.signal_status = "red"
        self.reference_line = 300  # Reference line for red-light violations
        self.fps = 0
        self.current_time = ""
        os.makedirs("static/images", exist_ok=True)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

    def save_violation(self, image, plate_number, violation_type, speed):
        timestamp = datetime.now()
        filename = f'./static/images/violation_{timestamp.strftime("%Y%m%d_%H%M%S")}.jpg'
        cv2.imwrite(filename, image)

        db = mysql.connector.connect(**self.db_config)
        cursor = db.cursor()
        query = """
            INSERT INTO nlcn_violations (vt_license_number, vt_images, vt_violation, vt_speed, vt_time)
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(query, (plate_number, filename, violation_type, speed, timestamp))
        db.commit()
        db.close()

    def detect_license_plate(self, image):
        plates = PLATE_DETECTOR_MODEL(image, size=640)
        list_plates = plates.pandas().xyxy[0].values.tolist()
        list_read_plates = set()
        
        for plate in list_plates:
            x, y = int(plate[0]), int(plate[1])
            w, h = int(plate[2] - plate[0]), int(plate[3] - plate[1])
            crop_img = image[y:y+h, x:x+w]
            cv2.rectangle(image, (x, y), (x+w, y+h), color=(0, 0, 225), thickness=2)
            
            for cc in range(2):
                for ct in range(2):
                    deskewed_img = utils_rotate.deskew(crop_img, cc, ct)
                    lp = helper.read_plate(PLATE_OCR_MODEL, deskewed_img)
                    if lp != "unknown":
                        list_read_plates.add(lp)
                        cv2.putText(image, lp, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                        break
                if lp != "unknown":
                    break
        
        return list_read_plates.pop() if list_read_plates else "Unknown"

    def calculate_speed(self, prev_position, current_position, fps):
        pixel_distance = np.linalg.norm(np.array(current_position) - np.array(prev_position))
        distance_in_meters = pixel_distance * self.meters_per_pixel
        speed_in_kmh = distance_in_meters * fps * 3.6
        return speed_in_kmh

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

            if int(current_time) % 93 < 30:
                self.signal_status = "green"
                cv2.putText(frame, "GREEN LIGHT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif int(current_time) % 93 < 33:
                self.signal_status = "yellow"
                cv2.putText(frame, "YELLOW LIGHT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                self.signal_status = "red"
                cv2.putText(frame, "RED LIGHT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.line(frame, (20, self.reference_line), (frame.shape[1], self.reference_line), (0, 255, 0), 2)
            vehicle_results = VEHICLE_MODEL(frame)
            detected_vehicles = []

            for *xyxy, conf, cls in vehicle_results.xyxy[0]:
                if conf > 0.5 and int(cls) == 2:
                    x1, y1, x2, y2 = map(int, xyxy)
                    vehicle_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    detected_vehicles.append((vehicle_center, x1, y1, x2, y2))

                    if vehicle_center in self.vehicle_positions:
                        prev_time, prev_position = self.vehicle_positions[vehicle_center]
                        speed = self.calculate_speed(prev_position, (x1, y1), self.fps)

                        if speed > self.speed_limit:
                            cv2.putText(frame, f"Speed: {int(speed)} km/h", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            violation_image = frame[y1:y2, x1:x2]
                            plate_number = self.detect_license_plate(violation_image)
                            self.save_violation(violation_image, plate_number, "Speeding", speed)

                    self.vehicle_positions[vehicle_center] = (current_time, (x1, y1))

                    cv2.circle(frame, vehicle_center, 5, (0, 255, 0), -1)
                    cv2.putText(frame, f"Speed: {int(speed)} km/h", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                    if self.signal_status == "red" and y2 > self.reference_line:
                        violation_image = frame[y1:y2, x1:x2]
                        plate_number = self.detect_license_plate(violation_image)
                        self.save_violation(violation_image, plate_number, "Red Light Violation", speed)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    def get_signal_status(self):
        return self.signal_status

    def get_current_time(self):
        return self.current_time
    
    def get_fps(self):
        return self.fps



@app.route('/')
def index():
    db = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='traffic_violations'
    )

    cursor = db.cursor()
    cursor.execute("SELECT * FROM nlcn_violations")
    violations = cursor.fetchall()
    db.close()

    # Get video feed parameters for display
    video_path = './static/videos/test_7.mp4'
    vehicle_model_path = 'model/Vehicle_detect_1.pt'
    plate_reader_model_path = 'model/LP_detector_nano_61.pt'
    plate_ocr_model_path = 'model/LP_ocr_nano_62.pt'
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': '',
        'database': 'traffic_violations'
    }

    traffic_violation = TrafficDetection(video_path, db_config=db_config)
    return render_template(
        "index.html",
        violations=violations,
        video_feed=url_for('video_feed'),
        fps=traffic_violation.get_fps(),
        signal=traffic_violation.get_signal_status(),
        time=traffic_violation.get_current_time()
    )


@app.route('/vd')
def video_feed():
    video_path = './static/videos/test_7.mp4'
    vehicle_model_path = 'model/Vehicle_detect_1.pt'
    plate_reader_model_path = 'model/LP_detector_nano_61.pt'
    plate_ocr_model_path = 'model/LP_ocr_nano_62.pt'
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': '',
        'database': 'traffic_violations'
    }

    traffic_violation = TrafficDetection(video_path, db_config=db_config)
    return Response(traffic_violation.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
