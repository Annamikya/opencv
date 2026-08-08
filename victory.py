import cv2
import mediapipe as mp
import time

# ---------------- MEDIAPIPE SETUP ----------------

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

start_time = time.time()

# Floating text variables
floating_texts = []

victory_before = False


def finger_up(hand, tip, pip):
    return hand[tip].y < hand[pip].y


# ---------------- MAIN LOOP ----------------

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp_ms = int((time.time() - start_time) * 1000)

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    # ---------------- HAND DETECTION ----------------

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        # Finger states
        index_up = finger_up(hand, 8, 6)
        middle_up = finger_up(hand, 12, 10)
        ring_up = finger_up(hand, 16, 14)
        pinky_up = finger_up(hand, 20, 18)

        # Victory sign:
        # Index + Middle UP
        # Ring + Pinky DOWN

        victory = (
            index_up
            and middle_up
            and not ring_up
            and not pinky_up
        )

        # Index finger position
        x = int(hand[8].x * width)
        y = int(hand[8].y * height)

        if victory:

            cv2.putText(
                frame,
                "VICTORY SIGN!",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                3
            )

            # Only create once per gesture
            if not victory_before:

                floating_texts.append([
                    x,
                    y,
                    50
                ])

                victory_before = True

        else:
            victory_before = False

    else:
        victory_before = False

    # ---------------- FLOATING TEXT ----------------

    for text in floating_texts[:]:

        x, y, life = text

        # Move upward
        y -= 3

        life -= 1

        text[1] = y
        text[2] = life

        cv2.putText(
            frame,
            "VICTORY",
            (x - 80, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 255),
            3
        )

        cv2.putText(
            frame,
            "✨",
            (x + 80, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        if life <= 0:
            floating_texts.remove(text)

    # ---------------- INSTRUCTIONS ----------------

    cv2.putText(
        frame,
        "Show Victory Sign",
        (20, height - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to Exit",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Victory Sign Animation",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
landmarker.close()
cv2.destroyAllWindows()