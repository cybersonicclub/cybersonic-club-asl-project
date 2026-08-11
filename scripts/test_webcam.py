# Manual smoke test for HandTracker -> opens the webcam, overlays detected
# landmarks, prints them to console
#
# Press 'q' to quit.
#
# Run with: python scripts/test_webcam.py
 
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from src.hand_tracker import HandTracker

def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam. Check camera permissions.")
        return

    with HandTracker(max_num_hands=1) as tracker:
        print("Press 'q' to quit.")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: failed to read frame")
                break

            frame = cv2.flip(frame, 1)

            results = tracker.process(frame)

            if results:
                h, w, _ = frame.shape
                for hand in results:
                    for x, y, _z in hand.landmarks:
                        px, py = int(x * w), int(y * h)
                        cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)
                    cv2.putText(
                        frame,
                        f"{hand.handedness} ({hand.confidence:.2f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
            else:
                cv2.putText(
                    frame,
                    "No hand detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("HandTracker test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
