import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = None, None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror view

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            # Index finger tip = landmark 8
            h, w, _ = frame.shape
            cx = int(handLms.landmark[8].x * w)
            cy = int(handLms.landmark[8].y * h)

            # Draw dot
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)

            # Draw line
            if prev_x is not None:
                cv2.line(canvas, (prev_x, prev_y), (cx, cy), (0, 0, 255), 6)

            prev_x, prev_y = cx, cy

    else:
        prev_x, prev_y = None, None

    # Merge
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)

    final = cv2.add(frame_bg, canvas_fg)

    cv2.imshow("Finger Drawing", final)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()