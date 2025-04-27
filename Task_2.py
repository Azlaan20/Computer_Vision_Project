import os
import cv2
import psutil
import shutil
import mediapipe as mp
import subprocess
import time
import torch
import numpy as np

# ─── System Cleanup ────────────────────────────────────────────────────────────

def cleanup_system():
    blacklist = ['firefox.exe', 'OneDrive.exe', 'Teams.exe',
                 'Spotify.exe', 'YourPhone.exe', 'Telegram.exe']
    for proc in psutil.process_iter(['pid','name']):
        try:
            if proc.info['name'] in blacklist:
                psutil.Process(proc.info['pid']).terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    for temp in [os.getenv('TEMP'), os.getenv('TMP'), 'C:\\Windows\\Temp']:
        try:
            shutil.rmtree(temp, ignore_errors=True)
            os.makedirs(temp, exist_ok=True)
        except:
            pass

    try:
        subprocess.call(
            ['PowerShell','-Command','Clear-RecycleBin -Force'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except:
        pass

cleanup_system()

# ─── Load MiDaS Model (cached) ─────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_midas():
    model_dir = os.path.join(torch.hub.get_dir(),'checkpoints'); os.makedirs(model_dir,exist_ok=True)
    fn = 'dpt_swin2_tiny_256.pt'; path = os.path.join(model_dir, fn)
    if not os.path.exists(path):
        print("[INFO] Downloading MiDaS_small model...")
        m = torch.hub.load("intel-isl/MiDaS","MiDaS_small")
    else:
        print("[INFO] Loading MiDaS_small model from cache.")
        m = torch.hub.load("intel-isl/MiDaS","MiDaS_small", skip_validation=True)
    return m.to(device).eval()
midas = load_midas()
transforms = torch.hub.load("intel-isl/MiDaS","transforms", skip_validation=True)
depth_transform = transforms.small_transform  # [1,3,H',W']

# ─── MediaPipe Pose Setup ─────────────────────────────────────────────────────

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=0,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
draw_util = mp.solutions.drawing_utils
lspec = draw_util.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
cspec = draw_util.DrawingSpec(color=(0,128,255), thickness=1, circle_radius=1)

# ─── Classical Detectors ───────────────────────────────────────────────────────

sift = cv2.SIFT_create()
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ─── UI Controls ──────────────────────────────────────────────────────────────

def nothing(x): pass
cv2.namedWindow("Control",cv2.WINDOW_NORMAL)
cv2.createTrackbar("Brightness","Control",50,100,nothing)
cv2.createTrackbar("Contrast","Control",50,100,nothing)
low_light = False
occlusion = False
multi_person = False

# ─── Webcam Setup ─────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360)
if not cap.isOpened(): raise RuntimeError("Could not open webcam")

# ─── Main Loop ─────────────────────────────────────────────────────────────────

while True:
    t_start = time.time()
    ret, frame = cap.read()
    if not ret: break

    # UI Toggles
    key = cv2.waitKey(1) & 0xFF
    if key==ord('l'): low_light=not low_light
    elif key==ord('o'): occlusion=not occlusion
    elif key==ord('m'): multi_person=not multi_person
    elif key==ord('q'): break

    # Brightness/Contrast
    b = cv2.getTrackbarPos("Brightness","Control")/50.0  # 0–2
    c = cv2.getTrackbarPos("Contrast","Control")/50.0    # 0–2
    frame = cv2.convertScaleAbs(frame, alpha=c, beta=int((b-1)*100))

    # Simulated scenarios
    if low_light:
        frame = (frame * 0.3).astype(np.uint8)
    if occlusion:
        h,w = frame.shape[:2]
        cv2.rectangle(frame, (w//3,h//3),(w*2//3,h*2//3),(0,0,0),-1)

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Task 1: 2D Pose Detection
    t0 = time.time()
    results = pose.process(rgb)
    if results.pose_landmarks:
        draw_util.draw_landmarks(frame, results.pose_landmarks,
                                 mp_pose.POSE_CONNECTIONS,
                                 lspec, cspec)
    t1 = time.time()

    # Task 2: Depth Estimation
    input_batch = depth_transform(rgb).to(device)
    with torch.no_grad():
        pred = midas(input_batch)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=(h,w),
            mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()
    t2 = time.time()

    # Task 3 & 4: 3D Pose + Gesture
    pose3d = {}
    gesture = None
    if results.pose_landmarks:
        for idx,lm in enumerate(results.pose_landmarks.landmark):
            px,py = int(lm.x*w), int(lm.y*h)
            if 0<=px<w and 0<=py<h:
                z = float(pred[py,px]); x=lm.x*w; y=lm.y*h
                pose3d[idx] = (x,y,z)
                if idx in (11,13,15):
                    name={11:"R-Shoulder",13:"R-Elbow",15:"R-Wrist"}[idx]
                    cv2.putText(frame,f"{name} Z={z:.1f}",
                                (int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,(255,255,255),1,cv2.LINE_AA)
        # Gesture: Hands Up
        if all(k in pose3d for k in (11,12,15,16)):
            _,yrs,_=pose3d[11]; _,yls,_=pose3d[12]
            _,yrw,_=pose3d[15]; _,ylw,_=pose3d[16]
            if yrw<yrs and ylw<yls: gesture="Hands Up"
        # Gesture: Right Point
        if all(k in pose3d for k in (11,13,15)):
            xrs,_ ,_ = pose3d[11]; xre,_,_=pose3d[13]; xrw,y_rw,_=pose3d[15]
            if xrw>xre+50 and xre>xrs+50 and abs(y_rw-yrs)<50:
                gesture="Right Point"
    if multi_person:
        # HOG person detection in classical mode
        rects,_ = hog.detectMultiScale(frame, winStride=(8,8))
    t3 = time.time()

    # Task 5: Classical keypoints & HOG
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp = sift.detect(gray,None)
    frame = cv2.drawKeypoints(frame,kp,None,(255,0,0),flags=0)
    if multi_person:
        for x,y,ww,hh in rects:
            cv2.rectangle(frame,(x,y),(x+ww,y+hh),(0,0,255),2)
    t4 = time.time()

    # Overlay performance & counts
    cv2.putText(frame,
        f"SIFT:{len(kp)} HOG:{len(rects) if multi_person else 0}",
        (10,50),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2,cv2.LINE_AA)
    cv2.putText(frame,
        f"Pose:{(t1-t0)*1000:.1f}ms Depth:{(t2-t1)*1000:.1f}ms "
        f"Classical:{(t4-t3)*1000:.1f}ms",
        (10,75),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2,cv2.LINE_AA)
    if gesture:
        cv2.putText(frame,f"Gesture:{gesture}",(10,h-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2,cv2.LINE_AA)

    # Combine with depth map for UI
    depth_norm = cv2.normalize(pred,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm,cv2.COLORMAP_MAGMA)
    combined = np.hstack((frame, cv2.resize(depth_color,(w,h))))
    fps = 1.0/(time.time()-t_start)
    cv2.putText(combined,f"FPS:{fps:.1f}",(10,25),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)

    cv2.imshow("Pose + Depth + Classical + UI", combined)

cap.release()
cv2.destroyAllWindows()
