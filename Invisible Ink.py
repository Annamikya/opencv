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

        # Green dot
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # 🔥 FORCE DRAW (key fix)
        if prev_x is not None:
            cv2.line(canvas, (prev_x, prev_y), (cx, cy), (0, 0, 255), 10)

        prev_x, prev_y = cx, cy

    # ❌ DO NOT reset prev_x here (this was breaking your line)

    # 🔥 SHOW MERGED OUTPUT (important)
    combined = cv2.add(frame, canvas)

    cv2.imshow("Air Draw", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()