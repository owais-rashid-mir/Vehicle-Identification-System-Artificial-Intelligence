detect_image.py : For detecting on images.


detect_video.py : For detecting on videos.


detect_camera_realtime.py : For detecting in real-time via camera.


 main.py:
   - Loads YOLOv8 models for vehicle detection and license plate detection.
   - Reads a video file frame by frame.
   - Detects vehicles using YOLOv8 and tracks them using the SORT (Simple Online and Realtime Tracking) algorithm.
   - Detects license plates on the detected vehicles using the license plate detection model.
   - Reads the license plate numbers using EasyOCR.
   - Saves the results in a CSV file (test.csv).


util.py:
   - Contains utility functions for writing results to a CSV file, checking license plate format, formatting license plate text, and reading license plates from images.

   
visualize.py:
   - Visualizes the results by drawing bounding boxes around vehicles and license plates.
   - Combines the license plate image with the original frame and adds the license plate number as text.
   - Saves the processed frames as a video (out.mp4).


add_missing_data.py:
   - Interpolates missing data in the CSV file to ensure that each frame has consistent information for each vehicle.
   - It uses linear interpolation to estimate bounding boxes and other details for frames where data is missing.
   - Writes the updated data to a new CSV file (test_interpolated.csv).

