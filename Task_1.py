import os
import cv2
import psutil
import shutil
import mediapipe as mp
import subprocess
import time

# ─── System Cleanup ───────────────────────────────────────────────────────────

def cleanup_system():
    blacklist = ['firefox.exe', 'OneDrive.exe', 'Teams.exe',
                 'Spotify.exe', 'YourPhone.exe', 'Telegram.exe']
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.info['name'] in blacklist:
                psutil.Process(proc.info['pid']).terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    temp_dirs = [os.getenv('TEMP'), os.getenv('TMP'), 'C:\\Windows\\Temp']
    for temp in temp_dirs:
        try:
            shutil.rmtree(temp, ignore_errors=True)
            os.makedirs(temp, exist_ok=True)
        except:
            pass

    try:
        subprocess.call(['PowerShell', '-Command', 'Clear-RecycleBin -Force'], stdout=subprocess.DEVNULL)
    except:
        pass

cleanup_system()

# ─── MediaPipe Pose Setup — FAST MODE ─────────────────────────────────────────

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,                   # Fastest version
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

drawing_utils = mp.solutions.drawing_utils
landmark_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
connection_style = drawing_utils.DrawingSpec(color=(0, 128, 255), thickness=1, circle_radius=1)

# ─── Webcam Setup (480p or lower) ─────────────────────────────────────────────

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

    # Skip optional filters to reduce latency
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style
        )

    # Show FPS
    fps = int(1 / (time.time() - start_time))
    cv2.putText(frame, f'FPS: {fps}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Fast Pose Detection — MediaPipe", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
