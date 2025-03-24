import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("fire.pt")  # Replace with the path to your YOLO model

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set a confidence threshold
CONFIDENCE_THRESHOLD = 0.5

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Run the YOLO model
        results = model(frame)

        # Filter results for fire detections
        fire_detections = []
        for detection in results[0].boxes:
            if detection.cls == 0 and float(detection.conf) > CONFIDENCE_THRESHOLD:
                fire_detections.append(detection)

        # Annotate the frame with fire detections
        for detection in fire_detections:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            confidence = float(detection.conf)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for fire
            label = f"Fire: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Fire Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()