import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time

# ---------------- MEDIAPIPE SETUP ----------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)

# ---------------- CAMERA ----------------

cap = cv2.VideoCapture(0)

start_time = time.time()

trail = []
portal_active = False
portal_x = 0
portal_y = 0
portal_radius = 60

pinch_before = False


def finger_up(hand, tip, pip):
    return hand[tip].y < hand[pip].y


def draw_portal(frame, x, y, radius):
    overlay = frame.copy()

    # glowing rings
    for r in range(radius, radius + 35, 7):
        cv2.circle(
            overlay,
            (x, y),
            r,
            (255, 0, 255),
            3
        )

    # inner rings
    for r in range(15, radius, 15):
        cv2.circle(
            overlay,
            (x, y),
            r,
            (255, 255, 0),
            2
        )

    # rotating sparks
    t = time.time()

    for i in range(30):
        angle = (i * 12) + (t * 120)

        px = int(
            x + math.cos(math.radians(angle)) * radius
        )

        py = int(
            y + math.sin(math.radians(angle)) * radius
        )

        cv2.circle(
            overlay,
            (px, py),
            4,
            (0, 255, 255),
            -1
        )

    cv2.addWeighted(
        overlay,
        0.7,
        frame,
        0.3,
        0,
        frame
    )


# ---------------- MAIN LOOP ----------------

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp_ms = int(
        (time.time() - start_time) * 1000
    )

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        # Index finger tip
        index = hand[8]

        index_x = int(index.x * width)
        index_y = int(index.y * height)

        # Thumb
        thumb = hand[4]

        thumb_x = int(thumb.x * width)
        thumb_y = int(thumb.y * height)

        # ---------------- GLOWING TRAIL ----------------

        trail.append((index_x, index_y))

        if len(trail) > 25:
            trail.pop(0)

        for i in range(1, len(trail)):

            thickness = int(i / 4) + 1

            cv2.line(
                frame,
                trail[i - 1],
                trail[i],
                (255, 0, 255),
                thickness
            )

        cv2.circle(
            frame,
            (index_x, index_y),
            10,
            (255, 255, 255),
            -1
        )

        # ---------------- FINGER STATES ----------------

        index_up = finger_up(hand, 8, 6)
        middle_up = finger_up(hand, 12, 10)
        ring_up = finger_up(hand, 16, 14)
        pinky_up = finger_up(hand, 20, 18)

        # Pinch distance
        pinch_distance = math.sqrt(
            (index_x - thumb_x) ** 2 +
            (index_y - thumb_y) ** 2
        )

        # ---------------- PINCH = CREATE PORTAL ----------------

        if pinch_distance < 45:

            if not pinch_before:
                portal_active = True
                portal_x = index_x
                portal_y = index_y
                portal_radius = 60

                pinch_before = True

        else:
            pinch_before = False

        # ---------------- OPEN PALM = EXPAND ----------------

        open_palm = (
            index_up
            and middle_up
            and ring_up
            and pinky_up
        )

        if open_palm and portal_active:

            if portal_radius < 150:
                portal_radius += 3

            cv2.putText(
                frame,
                "PORTAL EXPANDING",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        # ---------------- FIST = CLOSE ----------------

        fist = (
            not index_up
            and not middle_up
            and not ring_up
            and not pinky_up
        )

        if fist:

            portal_active = False
            trail.clear()

            cv2.putText(
                frame,
                "PORTAL CLOSED",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

    # ---------------- DRAW PORTAL ----------------

    if portal_active:

        draw_portal(
            frame,
            portal_x,
            portal_y,
            portal_radius
        )

        cv2.putText(
            frame,
            "MAGIC PORTAL",
            (
                portal_x - 80,
                portal_y - portal_radius - 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # ---------------- INSTRUCTIONS ----------------

    cv2.putText(
        frame,
        "Index Finger = Magic Trail",
        (20, height - 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Pinch = Create Portal | Open Palm = Expand",
        (20, height - 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Fist = Close Portal | Q = Exit",
        (20, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Hand Controlled Magic Portal",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
landmarker.close()
cv2.destroyAllWindows()