CREATE DATABASE traffic_violations;

USE traffic_violations;

CREATE TABLE nlcn_violations (
    vt_id INT AUTO_INCREMENT PRIMARY KEY,
    vt_license_number VARCHAR(20),
    vt_images VARCHAR(255),
    vt_violation VARCHAR(50),
    vt_speed FLOAT,
    vt_time DATETIME
);
