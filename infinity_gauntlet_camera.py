import cv2
import mediapipe as mp
import numpy as np
import math
import time


# ==========================================
# MEDIAPIPE SETUP
# ==========================================

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


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

start_time = time.time()


# ==========================================
# COLORS - BGR
# ==========================================

SPACE = (255, 80, 20)       # Blue
MIND = (0, 255, 255)        # Yellow
REALITY = (40, 40, 255)     # Red
POWER = (255, 50, 180)      # Purple
TIME = (50, 255, 50)        # Green
SOUL = (0, 140, 255)        # Orange

GOLD = (20, 150, 220)
DARK_GOLD = (10, 70, 110)


# ==========================================
# HELPERS
# ==========================================

def point(hand, index, width, height):

    p = hand[index]

    return (
        int(p.x * width),
        int(p.y * height)
    )


def glow_circle(frame, center, radius, color):

    overlay = frame.copy()

    # Outer glow
    for extra in range(30, 0, -6):

        alpha = 0.04 + (
            (30 - extra) / 30
        ) * 0.05

        cv2.circle(
            overlay,
            center,
            radius + extra,
            color,
            -1
        )

        cv2.addWeighted(
            overlay,
            alpha,
            frame,
            1 - alpha,
            0,
            frame
        )

        overlay = frame.copy()

    # Main stone
    cv2.circle(
        frame,
        center,
        radius,
        color,
        -1
    )

    # Bright inner area
    bright = tuple(
        min(255, int(c * 1.7))
        for c in color
    )

    cv2.circle(
        frame,
        center,
        max(3, radius // 2),
        bright,
        -1
    )

    # White highlight
    cv2.circle(
        frame,
        (
            center[0] - radius // 3,
            center[1] - radius // 3
        ),
        max(2, radius // 5),
        (255, 255, 255),
        -1
    )


# ==========================================
# GAUNTLET
# ==========================================

def draw_gauntlet(frame, hand, width, height):

    # Important landmarks

    wrist = point(hand, 0, width, height)

    thumb_base = point(hand, 2, width, height)

    index_base = point(hand, 5, width, height)

    middle_base = point(hand, 9, width, height)

    ring_base = point(hand, 13, width, height)

    pinky_base = point(hand, 17, width, height)

    # Create palm polygon

    palm_points = np.array(
        [
            wrist,
            thumb_base,
            index_base,
            middle_base,
            ring_base,
            pinky_base
        ],
        np.int32
    )

    overlay = frame.copy()

    # Main golden glove

    cv2.fillPoly(
        overlay,
        [palm_points],
        GOLD
    )

    cv2.addWeighted(
        overlay,
        0.55,
        frame,
        0.45,
        0,
        frame
    )

    # Palm border

    cv2.polylines(
        frame,
        [palm_points],
        True,
        (80, 220, 255),
        3
    )

    # --------------------------------------
    # FINGER ARMOR
    # --------------------------------------

    finger_sets = [

        [5, 6, 7, 8],

        [9, 10, 11, 12],

        [13, 14, 15, 16],

        [17, 18, 19, 20]

    ]

    for finger in finger_sets:

        pts = []

        for landmark in finger:

            pts.append(
                point(
                    hand,
                    landmark,
                    width,
                    height
                )
            )

        for i in range(len(pts) - 1):

            cv2.line(
                frame,
                pts[i],
                pts[i + 1],
                GOLD,
                16
            )

            cv2.line(
                frame,
                pts[i],
                pts[i + 1],
                (80, 220, 255),
                2
            )

            # Armor joints

            cv2.circle(
                frame,
                pts[i],
                8,
                DARK_GOLD,
                -1
            )

    # Thumb armor

    thumb_points = [

        point(hand, 1, width, height),

        point(hand, 2, width, height),

        point(hand, 3, width, height),

        point(hand, 4, width, height)

    ]

    for i in range(len(thumb_points) - 1):

        cv2.line(
            frame,
            thumb_points[i],
            thumb_points[i + 1],
            GOLD,
            16
        )

        cv2.line(
            frame,
            thumb_points[i],
            thumb_points[i + 1],
            (80, 220, 255),
            2
        )

    # --------------------------------------
    # PALM ARMOR DETAILS
    # --------------------------------------

    center = point(
        hand,
        9,
        width,
        height
    )

    cv2.circle(
        frame,
        center,
        28,
        DARK_GOLD,
        4
    )

    cv2.circle(
        frame,
        center,
        18,
        GOLD,
        3
    )


# ==========================================
# STONES
# ==========================================

def draw_stones(
        frame,
        hand,
        width,
        height,
        pulse
):

    # Stone positions

    stone_locations = [

        # Thumb
        (4, SPACE),

        # Index
        (7, MIND),

        # Middle
        (11, REALITY),

        # Ring
        (15, POWER),

        # Pinky
        (19, SOUL),

        # Palm / Time stone
        (9, TIME)

    ]

    for landmark, stone_color in stone_locations:

        x, y = point(
            hand,
            landmark,
            width,
            height
        )

        if landmark == 9:

            radius = int(
                14 + pulse * 4
            )

        else:

            radius = int(
                10 + pulse * 3
            )

        glow_circle(
            frame,
            (x, y),
            radius,
            stone_color
        )


# ==========================================
# ENERGY PARTICLES
# ==========================================

def draw_energy_particles(
        frame,
        hand,
        width,
        height,
        timer
):

    palm = point(
        hand,
        9,
        width,
        height
    )

    for i in range(20):

        angle = (
            timer * 3
            + i * (2 * math.pi / 20)
        )

        radius = (
            35
            + 10 * math.sin(
                timer * 4 + i
            )
        )

        x = int(
            palm[0]
            + math.cos(angle) * radius
        )

        y = int(
            palm[1]
            + math.sin(angle) * radius
        )

        cv2.circle(
            frame,
            (x, y),
            2,
            (180, 230, 255),
            -1
        )


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = cap.read()

    if not success:

        print("Camera not detected")

        break

    # mirror image

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

    timestamp = int(
        (time.time() - start_time)
        * 1000
    )

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    current_time = time.time()

    pulse = (
        math.sin(
            current_time * 5
        ) + 1
    ) / 2


    # ======================================
    # HAND DETECTED
    # ======================================

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        # Gauntlet

        draw_gauntlet(
            frame,
            hand,
            width,
            height
        )

        # Infinity stones

        draw_stones(
            frame,
            hand,
            width,
            height,
            pulse
        )

        # Energy around palm

        draw_energy_particles(
            frame,
            hand,
            width,
            height,
            current_time
        )

        cv2.putText(
            frame,
            "INFINITY GAUNTLET ACTIVE",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "SHOW YOUR HAND",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )


    # ======================================
    # INFO
    # ======================================

    cv2.putText(
        frame,
        "Open your hand toward camera",
        (20, height - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to Exit",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    cv2.imshow(
        "Infinity Gauntlet AR",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


cap.release()

landmarker.close()

cv2.destroyAllWindows()