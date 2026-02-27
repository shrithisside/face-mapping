import cv2
import os
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tkinter as tk
from tkinter import filedialog

# 1. Path Safety Check
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "face_landmarker.task")

# 2. Input Selection with Finder
print("--- Face Counting System ---")
print("0: Webcam")
print("1: Choose from Finder")
choice = input("Select Source (0/1): ")

is_webcam = False
source_path = ""

if choice == "0":
    source_path = 0
    is_webcam = True
    is_image = False
else:
    # Safe Tkinter Initialization for macOS
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True) # Force Finder to the front
    
    source_path = filedialog.askopenfilename(
        title="Select Photo or Video",
        filetypes=[("Media Files", "*.jpg *.jpeg *.png *.mp4 *.mov *.avi")]
    )
    root.destroy()
    
    if not source_path:
        print("No file selected. Exiting."); exit()
        
    is_image = source_path.lower().endswith(('.jpg', '.jpeg', '.png'))

# 3. Optimized Recognition Settings
# Higher num_faces and lower confidence increases detection "range"
running_mode = vision.RunningMode.IMAGE if is_image else vision.RunningMode.VIDEO

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=running_mode,
    num_faces=50,                  # Increased range for large groups
    min_face_detection_confidence=0.2, # Lowered to catch distant/blurry faces
    min_face_presence_confidence=0.2
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# 4. Processing Loop
if is_image:
    frame = cv2.imread(source_path)
    # MediaPipe requires RGB
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp_image)
    
    if result.face_landmarks:
        for face in result.face_landmarks:
            for lm in face:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        count = len(result.face_landmarks)
        cv2.putText(frame, f"Faces: {count}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
        # Resize results to fit MacBook screen
        cv2.imshow("Result", cv2.resize(frame, (1280, 960)))
        cv2.waitKey(0)
else:
    cap = cv2.VideoCapture(source_path)
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Calculate timestamp for video
        timestamp_ms = int((time.time() - start_time) * 1000)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        face_count = len(result.face_landmarks) if result.face_landmarks else 0
        if result.face_landmarks:
            for face in result.face_landmarks:
                for lm in face:
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        cv2.putText(frame, f"Faces: {face_count}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
    
    cap.release()

landmarker.close()
cv2.destroyAllWindows()