import cv2
import numpy as np

class CameraStabilizer:
    def __init__(self, ref_frame):
        # Convert reference frame to grayscale
        self.ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
        # Define the motion model (Euclidean is good for slight rotation/translation)
        self.warp_mode = cv2.MOTION_EUCLIDEAN
        self.warp_matrix = np.eye(2, 3, dtype=np.float32)
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)

    def fix_frame(self, frame):
        target_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            # Find the shift between ref_gray and target_gray
            _, self.warp_matrix = cv2.findTransformECC(
                self.ref_gray, target_gray, self.warp_matrix, 
                self.warp_mode, self.criteria
            )
            # Warp the frame back to align with the original coordinates
            sz = frame.shape
            stable_frame = cv2.warpAffine(
                frame, self.warp_matrix, (sz[1], sz[0]), 
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
            return stable_frame
        except:
            # If the camera moves too much and ECC fails, return the original
            return frame