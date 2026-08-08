import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time

# =========================
# MEDIAPIPE SETUP
# =========================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

start_time = time.time()

sparks = []
energy_particles = []

# =========================
# HELPERS
# =========================

def finger_up(hand, tip, pip):
    return hand[tip].y < hand[pip].y


def get_point(hand, landmark_id, width, height):
    p = hand[landmark_id]
    return int(p.x * width), int(p.y * height)


def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


# =========================
# MAGIC CIRCLE
# =========================

def draw_magic_circle(frame, center, radius, angle):

    x, y = center

    overlay = frame.copy()

    # outer glow
    for extra in [0, 8, 16]:
        cv2.circle(
            overlay,
            center,
            radius + extra,
            (0, 140, 255),
            2
        )

    # inner circles
    cv2.circle(
        overlay,
        center,
        int(radius * 0.75),
        (0, 220, 255),
        2
    )

    cv2.circle(
        overlay,
        center,
        int(radius * 0.45),
        (255, 180, 0),
        2
    )

    # rotating spokes
    for i in range(12):

        a = math.radians(angle + i * 30)

        x1 = int(x + math.cos(a) * radius * 0.45)
        y1 = int(y + math.sin(a) * radius * 0.45)

        x2 = int(x + math.cos(a) * radius)
        y2 = int(y + math.sin(a) * radius)

        cv2.line(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 180, 255),
            1
        )

    # rotating rune dots
    for i in range(20):

        a = math.radians(-angle + i * 18)

        px = int(x + math.cos(a) * radius * 0.88)
        py = int(y + math.sin(a) * radius * 0.88)

        cv2.circle(
            overlay,
            (px, py),
            3,
            (0, 255, 255),
            -1
        )

    cv2.addWeighted(
        overlay,
        0.75,
        frame,
        0.25,
        0,
        frame
    )


# =========================
# SPARK SYSTEM
# =========================

def add_sparks(x, y, count=8):

    for _ in range(count):

        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 7)

        sparks.append([
            float(x),
            float(y),
            math.cos(angle) * speed,
            math.sin(angle) * speed,
            random.randint(15, 30)
        ])


def update_sparks(frame):

    for s in sparks[:]:

        x, y, dx, dy, life = s

        x += dx
        y += dy

        dy += 0.08
        life -= 1

        s[0] = x
        s[1] = y
        s[2] = dx
        s[3] = dy
        s[4] = life

        if life > 0:
            size = max(1, life // 8)

            cv2.circle(
                frame,
                (int(x), int(y)),
                size,
                (0, 180, 255),
                -1
            )
        else:
            sparks.remove(s)


# =========================
# ENERGY BALL
# =========================

def draw_energy_ball(frame, center, radius):

    x, y = center

    overlay = frame.copy()

    # glow circles
    for r in range(radius, radius + 35, 7):

        cv2.circle(
            overlay,
            center,
            r,
            (255, 100, 255),
            2
        )

    # core
    cv2.circle(
        overlay,
        center,
        max(8, radius // 4),
        (255, 255, 255),
        -1
    )

    # rotating particles
    t = time.time()

    for i in range(25):

        a = t * 4 + i * (2 * math.pi / 25)

        rr = radius + random.randint(-8, 8)

        px = int(x + math.cos(a) * rr)
        py = int(y + math.sin(a) * rr)

        cv2.circle(
            overlay,
            (px, py),
            3,
            (255, 0, 255),
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


# =========================
# TWO HAND PORTAL
# =========================

def draw_portal(frame, center, radius):

    x, y = center

    overlay = frame.copy()

    t = time.time() * 120

    # glow rings
    for extra in range(0, 35, 7):
        cv2.circle(
            overlay,
            center,
            radius + extra,
            (255, 0, 180),
            2
        )

    # spiral
    for i in range(50):

        angle = math.radians(t + i * 18)

        rr = radius * (i / 50)

        px = int(x + math.cos(angle) * rr)
        py = int(y + math.sin(angle) * rr)

        cv2.circle(
            overlay,
            (px, py),
            3,
            (255, 80, 255),
            -1
        )

    # center dark hole
    cv2.circle(
        overlay,
        center,
        int(radius * 0.35),
        (10, 10, 20),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.75,
        frame,
        0.25,
        0,
        frame
    )


# =========================
# MAIN LOOP
# =========================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp_ms = int(
        (time.time() - start_time) * 1000
    )

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    angle = int(time.time() * 100) % 360

    hands_detected = result.hand_landmarks

    # =========================
    # ONE OR TWO HANDS
    # =========================

    if hands_detected:

        # draw magic circles for every detected hand
        for hand in hands_detected:

            wrist = get_point(
                hand,
                0,
                width,
                height
            )

            index_tip = get_point(
                hand,
                8,
                width,
                height
            )

            middle_tip = get_point(
                hand,
                12,
                width,
                height
            )

            ring_tip = get_point(
                hand,
                16,
                width,
                height
            )

            pinky_tip = get_point(
                hand,
                20,
                width,
                height
            )

            thumb_tip = get_point(
                hand,
                4,
                width,
                height
            )

            index_up = finger_up(hand, 8, 6)
            middle_up = finger_up(hand, 12, 10)
            ring_up = finger_up(hand, 16, 14)
            pinky_up = finger_up(hand, 20, 18)

            open_palm = (
                index_up and
                middle_up and
                ring_up and
                pinky_up
            )

            pinch = (
                distance(
                    index_tip,
                    thumb_tip
                ) < 45
            )

            # OPEN PALM -> MAGIC CIRCLE
            if open_palm:

                palm_x = int(
                    (
                        wrist[0]
                        + index_tip[0]
                        + pinky_tip[0]
                    ) / 3
                )

                palm_y = int(
                    (
                        wrist[1]
                        + index_tip[1]
                        + pinky_tip[1]
                    ) / 3
                )

                draw_magic_circle(
                    frame,
                    (palm_x, palm_y),
                    95,
                    angle
                )

                add_sparks(
                    palm_x,
                    palm_y,
                    3
                )

            # PINCH -> ENERGY BALL
            if pinch:

                ball_x = int(
                    (
                        index_tip[0]
                        + thumb_tip[0]
                    ) / 2
                )

                ball_y = int(
                    (
                        index_tip[1]
                        + thumb_tip[1]
                    ) / 2
                )

                draw_energy_ball(
                    frame,
                    (ball_x, ball_y),
                    28
                )

                add_sparks(
                    ball_x,
                    ball_y,
                    2
                )

        # =========================
        # TWO HANDS -> BIG PORTAL
        # =========================

        if len(hands_detected) == 2:

            hand1 = hands_detected[0]
            hand2 = hands_detected[1]

            p1 = get_point(
                hand1,
                9,
                width,
                height
            )

            p2 = get_point(
                hand2,
                9,
                width,
                height
            )

            center_x = int(
                (p1[0] + p2[0]) / 2
            )

            center_y = int(
                (p1[1] + p2[1]) / 2
            )

            hand_distance = distance(
                p1,
                p2
            )

            portal_radius = int(
                max(
                    60,
                    min(
                        hand_distance / 2,
                        180
                    )
                )
            )

            draw_portal(
                frame,
                (center_x, center_y),
                portal_radius
            )

            add_sparks(
                center_x,
                center_y,
                4
            )

            cv2.putText(
                frame,
                "TWO HAND PORTAL",
                (
                    center_x - 100,
                    center_y - portal_radius - 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

    # =========================
    # UPDATE SPARKS
    # =========================

    update_sparks(frame)

    # =========================
    # UI TEXT
    # =========================

    cv2.putText(
        frame,
        "DOCTOR STRANGE MAGIC SYSTEM",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 180, 255),
        2
    )

    cv2.putText(
        frame,
        "Open Palm = Magic Circle",
        (20, height - 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "Pinch = Energy Ball",
        (20, height - 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "Two Hands = Portal",
        (20, height - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "Press Q to Exit",
        (width - 170, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.imshow(
        "Doctor Strange Magic",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
landmarker.close()
cv2.destroyAllWindows()