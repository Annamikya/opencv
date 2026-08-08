import cv2
import mediapipe as mp
import random
import math
import time

# ---------------- MEDIAPIPE TASKS ----------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)

# ---------------- CAMERA ----------------

cap = cv2.VideoCapture(0)

# ---------------- FIREWORK ----------------

particles = []

colors = [
    (0, 0, 255),       # Red
    (0, 255, 255),     # Yellow
    (255, 0, 255),     # Pink
    (255, 255, 0),     # Cyan
    (0, 165, 255),     # Orange
    (255, 255, 255)    # White
]


def create_firework(x, y):

    color = random.choice(colors)

    for i in range(60):

        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(3, 10)

        dx = math.cos(angle) * speed
        dy = math.sin(angle) * speed

        particles.append([
            float(x),
            float(y),
            dx,
            dy,
            color,
            35
        ])


# Prevent continuous explosions while holding pinch
pinched_before = False

start_time = time.time()

# ---------------- MAIN LOOP ----------------

while True:

    success, frame = cap.read()

    if not success:
        print("Camera not detected")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    # BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Timestamp must keep increasing
    timestamp_ms = int((time.time() - start_time) * 1000)

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    # ---------------- HAND DETECTION ----------------

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        # Thumb tip = landmark 4
        thumb = hand[4]

        # Index finger tip = landmark 8
        index = hand[8]

        thumb_x = int(thumb.x * width)
        thumb_y = int(thumb.y * height)

        index_x = int(index.x * width)
        index_y = int(index.y * height)

        # Draw thumb
        cv2.circle(
            frame,
            (thumb_x, thumb_y),
            10,
            (0, 255, 255),
            -1
        )

        # Draw index finger
        cv2.circle(
            frame,
            (index_x, index_y),
            10,
            (255, 255, 255),
            -1
        )

        # Line between thumb and index
        cv2.line(
            frame,
            (thumb_x, thumb_y),
            (index_x, index_y),
            (255, 255, 255),
            2
        )

        # Calculate distance
        distance = math.sqrt(
            (thumb_x - index_x) ** 2 +
            (thumb_y - index_y) ** 2
        )

        # ---------------- PINCH ----------------

        if distance < 45:

            if not pinched_before:

                create_firework(index_x, index_y)

                pinched_before = True

            cv2.putText(
                frame,
                "BOOM!",
                (index_x - 50, index_y - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                3
            )

        else:
            pinched_before = False

    else:
        pinched_before = False

    # ---------------- PARTICLES ----------------

    for particle in particles[:]:

        x, y, dx, dy, color, life = particle

        x += dx
        y += dy

        # Gravity
        dy += 0.15

        life -= 1

        particle[0] = x
        particle[1] = y
        particle[2] = dx
        particle[3] = dy
        particle[5] = life

        if life > 0:

            cv2.circle(
                frame,
                (int(x), int(y)),
                4,
                color,
                -1
            )

        else:
            particles.remove(particle)

    # Instructions
    cv2.putText(
        frame,
        "Pinch thumb + index finger",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to Exit",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Hand Controlled Fireworks",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
landmarker.close()
cv2.destroyAllWindows()