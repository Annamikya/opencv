from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = None, None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if canvas is None:
        canvas = np.zeros_like(frame)  # black canvas

    results = model(frame)[0]

    for box in results.boxes:
        cls = int(box.cls[0])

        # 0 = person class in YOLO
        if cls == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Draw current point
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            if prev_x is not None and prev_y is not None:
                cv2.line(canvas, (prev_x, prev_y), (cx, cy), (0, 0, 255), 5)

            prev_x, prev_y = cx, cy
            break

    # If nothing detected → reset
    else:
        prev_x, prev_y = None, None

    cv2.imshow("Canvas", canvas)
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()