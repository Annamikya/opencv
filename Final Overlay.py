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
        canvas = np.zeros_like(frame)

    results = model(frame)[0]

    if len(results.boxes) > 0:
        box = results.boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Draw tracking dot
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # Draw line on canvas
        if prev_x is not None:
            cv2.line(canvas, (prev_x, prev_y), (cx, cy), (0, 0, 255), 8)

        prev_x, prev_y = cx, cy

    # 🔥 Merge canvas + webcam properly
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    mask_inv = cv2.bitwise_not(mask)

    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)

    final = cv2.add(frame_bg, canvas_fg)

    cv2.imshow("Air Writing", final)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()