from ultralytics import YOLO
import cv2

# Load your trained YOLO model
model = YOLO(r"C:\Users\lenovo\Desktop\yolo\best.pt")  # Path to your YOLO model

# Open the default camera (0 = default camera)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Unable to access the camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    # Perform prediction using the YOLO model
    results = model.predict(source=frame, conf=0.25)  # Adjust confidence threshold as needed

    # Annotate the frame with prediction results
    annotated_frame = results[0].plot()

    # Example values (replace with actual calibration data)
    focal_length = 800  # Focal length in pixels
    real_distance = 1000  # Distance from the camera to the object in mm

    # Extract bounding box information
    for box in results[0].boxes:
        # Get the coordinates of the bounding box
        x1, y1, x2, y2 = box.xyxy[0]  # Bounding box corners (top-left and bottom-right)
        pixel_width = x2 - x1  # Calculate width (longueur)
        pixel_height = y2 - y1  # Calculate height (largeur)

        # Convert coordinates to integers for display
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        pixel_width, pixel_height = int(pixel_width), int(pixel_height)

        # Convert to real-world dimensions
        real_width = (pixel_width * real_distance) / focal_length
        real_height = (pixel_height * real_distance) / focal_length

        # Display the real-world dimensions on the video feed
        text = f"Longueur: {real_width:.2f} mm, Largeur: {real_height:.2f} mm"
        cv2.putText(annotated_frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the real-time video stream with annotations
    cv2.imshow('YOLO Detection', annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()