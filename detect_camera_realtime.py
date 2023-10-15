import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort
import easyocr
import csv
import util  # Import your existing util.py file
import os

# Initialize the SORT tracker
mot_tracker = Sort()

# Load YOLOv8 model for vehicle detection (coco_model) and license plate detection (license_plate_detector)
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detection_best.pt')

# Initialize EasyOCR reader.
# ['en'] specifies that we want to recognize text in English only.
reader = easyocr.Reader(['en'], gpu=False)

output_height = 720  # Desired height
output_width = 1280  # Desired width

# Create the output folder for saving CSV and cropped grayscale license images, if it doesn't exist
output_folder = 'output_results_realtime'
os.makedirs(output_folder, exist_ok=True)

# Define the output video path
output_video_path = os.path.join(output_folder, 'camera_video_output.avi')

# Create the video writer for the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec as needed
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (output_width, output_height))


def save_results_to_csv(results, output_path):
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['car_id', 'vehicle_bbox', 'license_plate_bbox', 'license_plate_text', 'license_plate_text_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for car_id, data in results.items():
            writer.writerow({
                'car_id': car_id,
                'vehicle_bbox': data['vehicle_bbox'],
                'license_plate_bbox': data['license_plate_bbox'],
                'license_plate_text': data['license_plate_text'],
                'license_plate_text_score': data['license_plate_text_score']
            })


def detect_vehicle_and_license_plate_in_realtime():
    # Open the webcam
    cap = cv2.VideoCapture(0)  # Use 0 to select the default webcam (you can change this if needed)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open the webcam.")
        return

    prev_detections = []
    prev_track_ids = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame for display
        frame = cv2.resize(frame, (output_width, output_height))

        # Detect vehicles using YOLOv8
        detections = coco_model(frame)[0]
        detections_ = []

        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection

            # Filter by class IDs for vehicles (you can modify this based on your class IDs)
            if int(class_id) in [2, 3, 5, 7]:
                # Ensure coordinates are integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                detections_.append([x1, y1, x2, y2, score])

        # Track vehicles using SORT
        if len(detections_) > 0:
            # Update the SORT tracker with the current detections
            track_ids = mot_tracker.update(np.asarray(detections_))

            # Store the current detections and track IDs
            prev_detections = detections_
            prev_track_ids = track_ids
        else:
            # If there are no detections in this frame, use the previous detections
            detections_ = prev_detections
            track_ids = prev_track_ids

        # Detect license plates using YOLOv8
        license_plates = license_plate_detector(frame)[0]

        results = {}

        # Process license plates
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Get the corresponding vehicle information
            xcar1, ycar1, xcar2, ycar2, car_id = util.get_car(license_plate, track_ids)

            # Crop the license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

            # Convert license plate image to grayscale
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # Read license plate number
            license_plate_text, license_plate_text_score = util.read_license_plate(license_plate_crop_thresh)

            if license_plate_text is not None:
                results[car_id] = {
                    'vehicle_bbox': [xcar1, ycar1, xcar2, ycar2],
                    'license_plate_bbox': [x1, y1, x2, y2],
                    'license_plate_text': license_plate_text,
                    'license_plate_text_score': license_plate_text_score
                }

                # Save the grayscale cropped license plate image in the output folder
                filename_base = f'realtime_output_license_{car_id}'
                output_license_plate_filename = os.path.join(output_folder, f'{filename_base}.png')
                cv2.imwrite(output_license_plate_filename, license_plate_crop_gray)

                # Save the results to a CSV file in the output folder with the same filename base
                output_csv_filename = os.path.join(output_folder, f'{filename_base}.csv')
                save_results_to_csv({car_id: results[car_id]}, output_csv_filename)  # Save individual result

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0),
                              4)  # Green bounding box for license plates

                # Display OCR license plate text and its background above the license plate bounding box
                license_plate_display_text = f"License Plate: {license_plate_text}"
                text_size, _ = cv2.getTextSize(license_plate_display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_x = int((x1 + x2 - text_size[0]) / 2)  # Center the OCR license plate text horizontally
                text_y = int(y1) - text_size[
                    1] - 10  # Position OCR license plate text above the license plate bounding box

                # Calculate the background rectangle dimensions for the OCR license plate text in red
                rect_x1 = text_x - 5
                rect_x2 = text_x + text_size[0] + 5
                rect_y1 = text_y - text_size[1] - 5
                rect_y2 = text_y + 5

                # Draw background rectangle for the OCR license plate text in red
                cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), -1)

                # Display OCR license plate text in green with smaller font
                cv2.putText(frame, license_plate_display_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Display confidence text and its background below the OCR license plate text
                confidence_text = f"Confidence: {license_plate_text_score:.2f}"
                text_size, _ = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_x = int((x1 + x2 - text_size[0]) / 2)  # Center the confidence text horizontally
                text_y = int(y2) + text_size[1] + 10  # Position confidence score below the OCR license plate text

                # Calculate the background rectangle dimensions for the confidence score in red
                rect_x1 = text_x - 5
                rect_x2 = text_x + text_size[0] + 5
                rect_y1 = text_y - text_size[1] - 5
                rect_y2 = text_y + 5

                # Draw background rectangle for the confidence score in red
                cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), -1)

                # Display confidence score in green with smaller font
                cv2.putText(frame, confidence_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw green bounding box for the license plate
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

            # Draw green bounding box for the vehicle
            cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0),
                          4)  # Green bounding box for vehicles

        # Add the frame with detections to the output video
        out.write(frame)

        # Display the frame with detections
        cv2.imshow('Real-time Detections', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and writer objects
    cap.release()
    out.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Call the function to detect vehicles and license plates in real-time
    detect_vehicle_and_license_plate_in_realtime()


