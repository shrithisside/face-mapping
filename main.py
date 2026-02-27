import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Use 0 for the default MacBook webcam
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        # Draw detections and count faces
        face_count = 0
        if results.detections:
            face_count = len(results.detections)
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        # Display the count on the frame
        cv2.putText(image, f'Faces: {face_count}', (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        cv2.imshow('MediaPipe Face Counter', image)

        # Press 'Esc' to close
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()