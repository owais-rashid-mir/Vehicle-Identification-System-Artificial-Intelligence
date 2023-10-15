from ultralytics import YOLO
import cv2  # OpenCV library.

import util
from util import get_car, read_license_plate, write_csv

# Sort is a simple online and realtime tracking algorithm for 2D multiple object tracking in video sequences.
from sort.sort import *

results = {}

# For tracking all the vehicles
mot_tracker = Sort()

# load models
# The '.pt' extension commonly indicates PyTorch model weights.
# yolov8n - 'n' here means we are using the Yolov8 Nano model.
coco_model = YOLO('yolov8n.pt')
# license_plate_detection_best.pt - After training the model at 32 epochs.
license_plate_detector = YOLO('./models/license_plate_detection_best.pt')

# load video
cap = cv2.VideoCapture('./sample2.mp4')

# 2, 3, 5, 7 - These are the class IDs of vehicles(car, motorbike, bus and truck) in the Coco model dataset, and we will be detecting these first, and then license plates.
vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True

while ret:
    frame_nmr += 1
    ret, frame = cap.read()

    if ret:
        results[frame_nmr] = {}

        # Detect Vehicles
        detections = coco_model(frame)[0]
        detections_ = []  # For saving the bounding boxes of all the vehicles that we are going to detect.

        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection

            if int(class_id) in vehicles:
                # If we have a detected a vehicle, we will append the bounding box and the confidence score to the 'detections_' variable.
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        # track_ids will contain the bounding box of detected vehicles and the vehicle ID. This ID will represent the detected vehicles througout the video.
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car. get_car is defined in util.py
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate - Converting the license plate images to gray scale. This image will then be passed to EasyOCR.
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # cv2.imshow('original_crop', license_plate_crop)
                # cv2.imshow('threshold', license_plate_crop_thresh)
                # cv2.waitKey(0)

                # read license plate number. read_license_plate is defined in util.py
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                # If license plate is detected, we will read all the info of the vehicle and its license plate.
                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

# write results. write_csv - defined in util.py
# results will be stored in test.csv of 'info_csv' folder
# write_csv(results, './info_csv/test.csv')
write_csv(results, './test.csv')
