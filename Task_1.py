import cv2
import mediapipe as mp
import time

# ─── MediaPipe Pose Setup — FAST MODE ─────────────────────────────────────────

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

drawing_utils = mp.solutions.drawing_utils
landmark_style = drawing_utils.DrawingSpec(
    color=(0, 255, 0), thickness=1, circle_radius=1
)
connection_style = drawing_utils.DrawingSpec(
    color=(0, 128, 255), thickness=1, circle_radius=1
)

# ─── Webcam Setup ──────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# ─── Main Loop ────────────────────────────────────────────────────────────────

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style,
        )

    elapsed = max(time.time() - start_time, 1e-9)
    fps = int(1 / elapsed)
    cv2.putText(
        frame,
        f"FPS: {fps}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Fast Pose Detection — MediaPipe", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
pose.close()
cv2.destroyAllWindows()
