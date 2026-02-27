import cv2
import time
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------- Initialization ----------------
model_path = "face_landmarker.task"
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=10 # Set to 10 for your group of friends
)
landmarker = FaceLandmarker.create_from_options(options)

# ---------------- Source Selection ----------------
print("0: Webcam\n1: video 1 (01.mp4)\n2: video 2 (02.mp4)\n3: Select file...")
choice = input("Enter choice: ")

is_webcam = False
if choice == "0":
    cap = cv2.VideoCapture(0)
    is_webcam = True
elif choice == "1":
    cap = cv2.VideoCapture('01.mp4')
elif choice == "2":
    cap = cv2.VideoCapture('02.mp4')
else:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename()
    cap = cv2.VideoCapture(path)
    root.destroy()

# ---------------- Main Loop ----------------
start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Logic: Webcams need elapsed time; videos need frame-based time
    if is_webcam:
        timestamp_ms = int((time.time() - start_time) * 1000)
    else:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
        timestamp_ms = int((frame_idx * 1000) / fps)

    # Process
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    # Visualize and Count
    face_count = 0
    if result.face_landmarks:
        face_count = len(result.face_landmarks)
        for face_landmarks in result.face_landmarks:
            for lm in face_landmarks:
                h, w, _ = frame.shape
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)

    # Display Counter
    cv2.putText(frame, f'Faces: {face_count}', (30, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.imshow("Face Counter", frame)

    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()