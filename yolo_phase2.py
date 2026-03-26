from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # Load pre-trained YOLOv8 nano model
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Run YOLO on frame

    annotated_frame = results[0].plot()  # Draw boxes on frame

    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()