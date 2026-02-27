import os

# This gets the exact path of where your script is saved
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "face_landmarker.task")

import cv2                           
import numpy as np                  
import mediapipe as mp               # MediaPipe main package
from mediapipe.tasks import python   # MediaPipe Tasks Python API
from mediapipe.tasks.python import vision  # Vision-related tasks (FaceLandmarker)

# ---------------- REQUIRED ALIASES (Missing Part) ----------------

# Get BaseOptions class from MediaPipe (used to load the model file)
BaseOptions = python.BaseOptions

# Get the FaceLandmarker class (this is the face landmark detector)
FaceLandmarker = vision.FaceLandmarker

# Get the options class used to configure FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions

# Get the running mode options (IMAGE, VIDEO, LIVE_STREAM)
VisionRunningMode = vision.RunningMode


# ---------------- Load Face Landmarker Model ----------------

# Path to the trained face landmark model file
model_path = "face_landmarker.task"

# Create configuration settings for the face landmark detector
options = FaceLandmarkerOptions(

    # Tell MediaPipe which model file to use
    base_options=BaseOptions(model_asset_path=model_path),

    # Tell MediaPipe that input will be video frames (needs timestamp)
    running_mode=VisionRunningMode.VIDEO,

    # Detect only one face in the frame
    num_faces=1
)

# Create the face landmark detector using the above settings
landmarker = FaceLandmarker.create_from_options(options)



# ---------------- Video Input ----------------

# Load video file (use 0 for webcam)
cap = cv2.VideoCapture('01.mp4', cv2.CAP_AVFOUNDATION) # Use this line to read from a video file (make sure the file is in the same directory as your script)
cap = cv2.VideoCapture(0) # Use 0 for webcam

# Get frames per second of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate delay between frames (milliseconds)
delay = int(1000 / fps)

# Keeps track of the time (in milliseconds) for each video frame
# Required in VIDEO mode so MediaPipe knows the order of frames
timestamp = 0


# ---------------- Main Loop ----------------
while cap.isOpened():

    # Read one frame from the video
    ret, frame = cap.read()

    # If video ends, exit loop
    if not ret:
        break

    # Resize frame for consistent processing
    frame = cv2.resize(frame, (960, 720))

    # Convert frame from BGR to RGB (MediaPipe requires RGB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert NumPy array to MediaPipe Image format
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Run face landmark detection using timestamp
    result = landmarker.detect_for_video(mp_image, timestamp)

    # Get frame height and width
    h, w = frame.shape[:2]

    # If face landmarks are detected
    if result.face_landmarks:

        # Loop through each detected face
        for face_landmarks in result.face_landmarks:

            # Loop through all landmarks of the face
            # print(len(face_landmarks))
            for lm in face_landmarks:

                # Convert normalized coordinates to pixel values
                x = int(lm.x * w)
                y = int(lm.y * h)

                # Draw landmark point
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Display the output frame
    cv2.imshow("Face Landmark Detection (MediaPipe Task)", frame)

    # Increase timestamp according to FPS
    timestamp += int(1000 / fps)

    # Exit when ESC key is pressed
    if cv2.waitKey(delay) & 0xFF == 27:
        break

# ---------------- Cleanup ----------------

# Release video capture object
cap.release()

# Close FaceLandmarker resources
landmarker.close()

# Close all OpenCV windows
cv2.destroyAllWindows()