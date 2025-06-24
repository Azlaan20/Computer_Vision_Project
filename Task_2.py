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
    blacklist = ['firefox.exe','OneDrive.exe','Teams.exe','Spotify.exe','YourPhone.exe','Telegram.exe']
    for proc in psutil.process_iter(['pid','name']):
        try:
            if proc.info['name'] in blacklist:
                proc.terminate()
        except:
            pass
    for d in [os.getenv('TEMP'), os.getenv('TMP'), 'C:\\Windows\\Temp']:
        try:
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
        except:
            pass
    try:
        subprocess.call(['PowerShell','-Command','Clear-RecycleBin -Force'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        pass
cleanup_system()

# ─── Load MiDaS Model ─────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_midas():
    m = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
    return m.to(device).eval()
midas = load_midas()
transforms = torch.hub.load("intel-isl/MiDaS", "transforms", skip_validation=True)
depth_transform = transforms.small_transform

# ─── MediaPipe Pose Setup ─────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=0,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
draw_util = mp.solutions.drawing_utils
lspec = draw_util.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=2)
cspec = draw_util.DrawingSpec(color=(0,128,255), thickness=1, circle_radius=2)

# ─── SIFT Detector ─────────────────────────────────────────────────────────────
sift = cv2.SIFT_create()

# ─── UI Controls ──────────────────────────────────────────────────────────────
def nothing(x): pass
cv2.namedWindow("Control", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Brightness","Control",50,100,nothing)
cv2.createTrackbar("Contrast","Control",50,100,nothing)
low_light = False
occlusion = False

# ─── Label Map ────────────────────────────────────────────────────────────────
labels = {
    0:"Nose", 11:"L-Shoulder",12:"R-Shoulder",
    13:"L-Elbow",14:"R-Elbow",15:"L-Wrist",16:"R-Wrist",
    23:"L-Hip",24:"R-Hip",25:"L-Knee",26:"R-Knee",
    27:"L-Ankle",28:"R-Ankle"
}

# ─── Snapshot Flags ───────────────────────────────────────────────────────────
captured = {
    'all_limbs': False,
    'gesture': False,
    'brightness_max': False,
    'brightness_min': False,
    'contrast_max': False,
    'contrast_min': False
}
def save_frame(name, img):
    path = os.path.join(os.getcwd(), f"{name}.jpg")
    cv2.imwrite(path, img)
    print(f"[CAPTURED] {name}.jpg")

# ─── Webcam Setup ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360)
if not cap.isOpened(): raise RuntimeError("Cannot open webcam")

while True:
    t_start=time.time()
    ret,frame=cap.read()
    if not ret: break

    key=cv2.waitKey(1)&0xFF
    if key==ord('l'): low_light=not low_light
    elif key==ord('o'): occlusion=not occlusion
    elif key==ord('q'): break

    # Brightness/Contrast
    b=cv2.getTrackbarPos("Brightness","Control")/50.0
    c=cv2.getTrackbarPos("Contrast","Control")/50.0
    frame=cv2.convertScaleAbs(frame,alpha=c,beta=int((b-1)*100))

    if low_light: frame=(frame*0.3).astype(np.uint8)
    if occlusion:
        h,w=frame.shape[:2]
        cv2.rectangle(frame,(w//3,h//3),(w*2//3,h*2//3),(0,0,0),-1)

    h,w=frame.shape[:2]
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    kps=sift.detect(gray,None)
    frame=cv2.drawKeypoints(frame,kps,None,(255,0,0),flags=0)

    results=pose.process(rgb)
    if results.pose_landmarks:
        draw_util.draw_landmarks(frame,results.pose_landmarks,
                                 mp_pose.POSE_CONNECTIONS,lspec,cspec)

    inp=depth_transform(rgb).to(device)
    with torch.no_grad():
        pred=midas(inp)
        pred=torch.nn.functional.interpolate(
            pred.unsqueeze(1),size=(h,w),
            mode="bicubic",align_corners=False
        ).squeeze().cpu().numpy()

    pose3d={}
    if results.pose_landmarks:
        for idx,lm in enumerate(results.pose_landmarks.landmark):
            px,py=int(lm.x*w),int(lm.y*h)
            if 0<=px<w and 0<=py<h:
                z=float(pred[py,px]); x=lm.x*w; y=lm.y*h
                pose3d[idx]=(x,y,z)

    gestures=[]
    if all(i in pose3d for i in (15,16,11,12)):
        if pose3d[15][1]<pose3d[11][1] and pose3d[16][1]<pose3d[12][1]:
            gestures.append("Hands Up")
    if all(i in pose3d for i in (11,12,15,16)):
        y_ls,y_rs = pose3d[11][1], pose3d[12][1]
        y_lw,y_rw = pose3d[15][1], pose3d[16][1]
        x_ls,x_rs = pose3d[11][0], pose3d[12][0]
        x_lw,x_rw = pose3d[15][0], pose3d[16][0]
        if (abs(y_lw-y_ls)<20 and abs(y_rw-y_rs)<20
            and x_lw< x_ls-40 and x_rw> x_rs+40):
            gestures.append("T-Pose")
    if all(i in pose3d for i in (15,16,23,24)):
        if (abs(pose3d[15][0]-pose3d[23][0])<30 and abs(pose3d[15][1]-pose3d[23][1])<30
         and abs(pose3d[16][0]-pose3d[24][0])<30 and abs(pose3d[16][1]-pose3d[24][1])<30):
            gestures.append("Hands on Hips")
    if all(i in pose3d for i in (12,14,16)):
        x_rs,y_rs = pose3d[12][0], pose3d[12][1]
        x_re,y_re = pose3d[14][0], pose3d[14][1]
        x_rw,y_rw = pose3d[16][0], pose3d[16][1]
        if x_rw>x_re+30 and x_re>x_rs+30 and abs(y_rw-y_rs)<40:
            gestures.append("Right Point")
    if all(i in pose3d for i in (11,13,15)):
        x_ls,y_ls = pose3d[11][0], pose3d[11][1]
        x_le,y_le = pose3d[13][0], pose3d[13][1]
        x_lw,y_lw = pose3d[15][0], pose3d[15][1]
        if x_lw< x_le-30 and x_le< x_ls-30 and abs(y_lw-y_ls)<40:
            gestures.append("Left Point")

    for idx, name in labels.items():
        if idx in pose3d:
            x,y,_ = pose3d[idx]
            cv2.putText(frame, name, (int(x)+5,int(y)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    if gestures:
        cv2.putText(frame, "Gestures: " + ", ".join(gestures),
                    (10, h-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,255), 2)

    num_dl = len(results.pose_landmarks.landmark) if results.pose_landmarks else 0
    num_sift = len(kps)
    cv2.putText(frame, f"DL joints: {num_dl}  SIFT points: {num_sift}",
                (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    depth_norm = cv2.normalize(pred,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm,cv2.COLORMAP_MAGMA)
    combined = np.hstack((frame, cv2.resize(depth_color,(w,h))))
    fps = 1.0/(time.time()-t_start)
    cv2.putText(combined, f"FPS: {fps:.1f}", (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # ─── Snapshot Conditions ─────────────────────────────────────────────────────
    if not captured['all_limbs'] and num_dl == 33:
        save_frame("pose_all_limbs", combined)
        captured['all_limbs'] = True

    if not captured['gesture'] and gestures:
        save_frame("pose_gesture_detected", combined)
        captured['gesture'] = True

    if not captured['brightness_max'] and b >= 2.0:
        save_frame("pose_brightness_max", combined)
        captured['brightness_max'] = True

    if not captured['brightness_min'] and b <= 0.2:
        save_frame("pose_brightness_min", combined)
        captured['brightness_min'] = True

    if not captured['contrast_max'] and c >= 2.0:
        save_frame("pose_contrast_max", combined)
        captured['contrast_max'] = True

    if not captured['contrast_min'] and c <= 0.2:
        save_frame("pose_contrast_min", combined)
        captured['contrast_min'] = True

    cv2.imshow("Pose + Depth + SIFT + UI", combined)

cap.release()
cv2.destroyAllWindows()
