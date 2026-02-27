import cv2
import time
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Configuration
model_path = "face_landmarker.task"
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

# 2. Input Selection
print("0: Webcam\n1: Select File (Photo or Video)")
choice = input("Enter choice: ")

source_path = 0
if choice == "1":
    # Skip the tkinter popup to avoid the termination crash
    print("Files in folder:", [f for f in os.listdir('.') if f.endswith(('.jpg', '.mp4'))])
    source_path = input("Enter the EXACT filename (e.g., group_photo.jpg): ")
    if not os.path.exists(source_path):
        print(f"File {source_path} not found!"); exit()

# Check if input is a static photo
is_image = str(source_path).lower().endswith(('.jpg', '.jpeg', '.png'))

# 3. Setup Landmarker based on Source Type
mode = VisionRunningMode.IMAGE if is_image else VisionRunningMode.VIDEO
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=mode,
    num_faces=10,
    min_face_detection_confidence=0.4 # Help detect faces in the background
)
landmarker = FaceLandmarker.create_from_options(options)

# 4. Execution
if is_image:
    # --- STATIC PHOTO LOGIC ---
    frame = cv2.imread(source_path)
    # Convert and detect using the IMAGE mode method
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp_image) # No timestamp for static images
    
    if result.face_landmarks:
        face_count = len(result.face_landmarks)
        for face in result.face_landmarks:
            for lm in face:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        cv2.putText(frame, f"Faces Found: {face_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Group Photo Detection", frame)
        cv2.waitKey(0)

else:
    # --- VIDEO / WEBCAM LOGIC ---
    cap = cv2.VideoCapture(source_path)
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Calculate timestamp required for VIDEO mode
        if choice == "0":
            ts = int((time.time() - start_time) * 1000)
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            ts = int((cap.get(cv2.CAP_PROP_POS_FRAMES) * 1000) / fps)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(mp_image, ts) # Timestamp is mandatory here

        face_count = len(result.face_landmarks) if result.face_landmarks else 0
        if result.face_landmarks:
            for face in result.face_landmarks:
                for lm in face:
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        cv2.putText(frame, f"Faces Found: {face_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Video Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
    
    cap.release()

landmarker.close()
cv2.destroyAllWindows()