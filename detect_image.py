# press 'q' to exit the output window - detect_image.py

import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort
import easyocr
import csv
import util  # Import your existing util.py file

# Initialize the SORT tracker
mot_tracker = Sort()

# Load YOLOv8 model for vehicle detection (coco_model) and license plate detection (license_plate_detector)
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detection_best.pt')

# Initialize EasyOCR reader.
# ['en'] specifies that we want to recognize text in English only.
reader = easyocr.Reader(['en'], gpu=False)

output_height = 1000  # Desired height for the License Plate Info window
output_width = 800  # Desired width for the License Plate Info window


# Function to display license plate information for all detected vehicles
def display_license_plate_info(frame, detection_results):
    output_height = 1000  # Desired height for the License Plate Info window
    output_width = 800  # Desired width for the License Plate Info window

    output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)  # Create a plain background image

    y_offset = 100  # Initial Y offset for displaying information

    for car_id, data in detection_results.items():
        x1_lp, y1_lp, x2_lp, y2_lp = data['license_plate_bbox']
        license_plate_text = data['license_plate_text']

        # Crop the license plate region and convert it to grayscale
        license_plate_crop = frame[int(y1_lp):int(y2_lp), int(x1_lp):int(x2_lp), :]
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

        # Convert the grayscale image to a three-channel image
        license_plate_crop_color = cv2.cvtColor(license_plate_crop_gray, cv2.COLOR_GRAY2BGR)

        # Display "Actual License Plate Text"
        actual_text = f"Actual License Plate Text:"
        cv2.putText(output_image, actual_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Insert the grayscale cropped license plate image below "Actual License Plate Text"
        output_image[y_offset + 40:y_offset + 40 + license_plate_crop_color.shape[0],
        20:20 + license_plate_crop_color.shape[1]] = license_plate_crop_color

        # Display "Predicted License Plate Text"
        predicted_text = f"Predicted License Plate Text:"
        cv2.putText(output_image, predicted_text, (10, y_offset + 60 + license_plate_crop_color.shape[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the OCR generated license plate number
        cv2.putText(output_image, license_plate_text, (10, y_offset + 100 + license_plate_crop_color.shape[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        y_offset += 160 + license_plate_crop_color.shape[0]  # Increase the Y offset for the next vehicle's information

    return output_image


def detect_vehicle_and_license_plate_in_image(image_path):
    # Load the image
    frame = cv2.imread(image_path)
    image_with_detections = frame.copy()  # Create a copy to draw detections

    # Detect vehicles using YOLOv8
    detections = coco_model(frame)[0]
    detections_ = []

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection

        # Filter by class IDs for vehicles (you can modify this based on your class IDs)
        if int(class_id) in [2, 3, 5, 7]:
            detections_.append([x1, y1, x2, y2, score])

    # Track vehicles
    track_ids = mot_tracker.update(np.asarray(detections_))

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

            # Save the grayscale cropped license plate image in the local directory.
            output_license_plate_filename = f'output_results/image_output_license_{car_id}.png'
            cv2.imwrite(output_license_plate_filename, license_plate_crop_gray)

            # Save the results to a CSV file with the same name as the image
            output_csv_filename = f'output_results/image_output_license_{car_id}.csv'
            save_results_to_csv({car_id: results[car_id]}, output_csv_filename)  # Save individual result

            # Draw bounding box for the license plate
            # Defining the desired bounding box color (in BGR format)
            bounding_box_color = (0, 255, 0)  # Green color

            # Change the bounding box thickness to a desired value (e.g., 4)
            bounding_box_thickness = 6

            cv2.rectangle(image_with_detections, (int(x1), int(y1)), (int(x2), int(y2)), bounding_box_color,
                          bounding_box_thickness)

            # Display OCR license plate text and its background above the license plate bounding box
            license_plate_display_text = f"License Plate: {license_plate_text}"
            text_size, _ = cv2.getTextSize(license_plate_display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            text_x = int((x1 + x2 - text_size[0]) / 2)  # Center the OCR license plate text horizontally
            text_y = int(y1) - text_size[1] - 20  # Position OCR license plate text above the license plate bounding box

            # Calculate the background rectangle dimensions for the OCR license plate text
            rect_x1 = text_x - 10
            rect_x2 = text_x + text_size[0] + 10
            rect_y1 = text_y - text_size[1] - 10
            rect_y2 = text_y + 10

            # Draw background rectangle for the OCR license plate text
            cv2.rectangle(image_with_detections, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255),
                          -1)  # Filled background

            # Display OCR license plate text
            cv2.putText(image_with_detections, license_plate_display_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)  # Display OCR license plate text in green

            # Display confidence text and its background below the OCR license plate text
            confidence_text = f"Confidence: {license_plate_text_score:.2f}"
            text_size, _ = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            text_x = int((x1 + x2 - text_size[0]) / 2)  # Center the confidence text horizontally
            text_y = int(y2) + text_size[1] + 20  # Position confidence score below the OCR license plate text

            # Calculate the background rectangle dimensions for the confidence score
            rect_x1 = text_x - 10
            rect_x2 = text_x + text_size[0] + 10
            rect_y1 = text_y - text_size[1] - 10
            rect_y2 = text_y + 10

            # Draw background rectangle for the confidence score
            cv2.rectangle(image_with_detections, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255),
                          -1)  # Filled background

            # Display confidence score
            cv2.putText(image_with_detections, confidence_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)  # Display confidence score in green

            cv2.rectangle(image_with_detections, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0),
                          4)  # Increase bounding box thickness

    return results, image_with_detections  # Return the results and the image with detections


# For displaying content in License Info Window
def create_output_image(frame, detection_results):
    output_image = frame.copy()

    for car_id, data in detection_results.items():
        x1, y1, x2, y2 = data['vehicle_bbox']
        x1_lp, y1_lp, x2_lp, y2_lp = data['license_plate_bbox']
        license_plate_text = data['license_plate_text']

        # Crop the license plate region and convert it to grayscale
        license_plate_crop = frame[int(y1_lp):int(y2_lp), int(x1_lp):int(x2_lp), :]
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

        # Convert the grayscale image to a three-channel image
        license_plate_crop_color = cv2.cvtColor(license_plate_crop_gray, cv2.COLOR_GRAY2BGR)

        # Display actual and predicted license plate numbers along with the cropped color license plate image
        cv2.putText(output_image, f"Actual License plate number: ", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0), 3)
        cv2.putText(output_image, f"Predicted license plate number: {license_plate_text}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        output_image[150:150 + license_plate_crop_color.shape[0],
        10:10 + license_plate_crop_color.shape[1]] = license_plate_crop_color

    return output_image


# Function to save results to a CSV file
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


# Main code
if __name__ == "__main__":
    # Specify the path to the input image
    input_image_path = 'multimedia_samples/18 copy.jpg'

    # Load the original image
    original_image = cv2.imread(input_image_path)

    # Detect vehicles and license plates in the input image and get the detection results
    detection_results, image_with_detections = detect_vehicle_and_license_plate_in_image(input_image_path)

    # Create the License Plate Info output image with license plate information for all detected vehicles
    output_image = display_license_plate_info(original_image.copy(), detection_results)

    # Specify the path to save the CSV file
    #output_csv_path = 'output_results/image_output_results.csv'

    # Save the results to a CSV file
    #save_results_to_csv(detection_results, output_csv_path)  # Call the function here

    # Set the desired window size (adjust these values as needed)
    window_width = 1280  # Width of the window
    window_height = 720  # Height of the window

    # Create windows for the Original Image, License Plate Info, and Detections
    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original Image', window_width, window_height)

    cv2.namedWindow('License Plate Info', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('License Plate Info', output_width, output_height)

    cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detections', window_width, window_height)

    # Display the Original Image window
    cv2.imshow('Original Image', original_image)

    # Display the License Plate Info window
    cv2.imshow('License Plate Info', output_image)

    # Display the Detections window
    cv2.imshow('Detections', image_with_detections)

    # Wait for a key press and close the windows when a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()
