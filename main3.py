import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. Fix Pathing ---
# This ensures the script looks in the same folder where the .py file is saved
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "face_landmarker.task")
image_path = os.path.join(current_dir, "group_photo.jpg") # Ensure this filename is correct!

if not os.path.exists(image_path):
    print(f"CRITICAL ERROR: Could not find {image_path}. Check your filename!")
    exit()

# --- 2. Configure for Group Photos ---
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE, # MUST be IMAGE for static photos
    num_faces=15, # Increased for your IET group size
    min_face_detection_confidence=0.3 # Lowered to catch people in the background
)
landmarker = FaceLandmarker.create_from_options(options)

# --- 3. Process ---
frame = cv2.imread(image_path)
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

print("Processing group photo... please wait.")
result = landmarker.detect(mp_image) # Use .detect for static images

# --- 4. Draw and Count ---
face_count = 0
if result.face_landmarks:
    face_count = len(result.face_landmarks)
    for face_landmarks in result.face_landmarks:
        for lm in face_landmarks:
            h, w, _ = frame.shape
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

# Display result
print(f"Faces detected: {face_count}")
cv2.putText(frame, f"Faces: {face_count}", (50, 150), 
            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

# Resize for display only so it fits on your MacBook Air screen
display_w = 1280
display_h = int(frame.shape[0] * (display_w / frame.shape[1]))
resized_view = cv2.resize(frame, (display_w, display_h))

cv2.imshow("IET Group Photo Results", resized_view)
cv2.waitKey(0)
cv2.destroyAllWindows()
landmarker.close()