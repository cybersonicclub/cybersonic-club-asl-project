import cv2 
import numpy as np 

# Extract functions from other files in src/
from src.hand_tracker import HandTracker
from src.landmarks import extract_features

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open the webcam")

tracker = HandTracker()

while True:
    # Read a frame from webcam
    ok, frame = cap.read()

    if not ok:
        print("ERROR: failed to read the frame")
        break

    # fix the inversion 
    frame = cv2.flip(frame, 1)

    # Run hand tracking on the frame 
    results = tracker.process(frame)

    if results:
        hand = results[0]

        landmarks = hand.landmarks

        features = extract_features(landmarks)

        # Display the webcam view 
        cv2.imshow("ASL Landmark Extraction", frame)

        # Quit when q is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
