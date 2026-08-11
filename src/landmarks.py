import numpy as np

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    # Make hand landmarks invariant to postion and scale
    #
    # landmarks: shape (21, 3) raw coordinates from HandTracker, in MediaPipe's
    #     image-relative units (0-1 for x/y).
    #
    # Returns shape (21, 3) with:
    #   - the wrist (landmark 0) moved to the origin
    #   - all points scaled so the wrist-to-middle-fingertip distance is 1.0
    #
    # This is why two people signing the same letter at different distances
    # from the camera, or in different spots in the frame, end up producing
    # similar feature vectors, the classifier trains on hand shape,
    # not position.
    
    wrist = landmarks[0]
    centered = landmarks - wrist

    middle_fingertip = centered[12]  # landmark 12 -> tip of middle finger
    scale = np.linalg.norm(middle_fingertip)
    if scale < 1e-6:
        scale = 1e-6  # avoid divide-by-zero cause float work funny in base 2 

    return centered / scale

def flatten_landmarks(landmarks: np.ndarray) -> np.ndarray:
    # Flatten the 21x3 landmarks into a 1D array of 63 values
    return landmarks.flatten()

# I can be lazy with this 
def extract_features(landmarks: np.ndarray) -> np.ndarray:
    # Normalize the landmarks first
    normalized = normalize_landmarks(landmarks)

    # Flatten the normalized landmarks into 63 values
    return flatten_landmarks(normalized)
