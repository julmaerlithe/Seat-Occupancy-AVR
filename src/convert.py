import cv2
import os

from stabilizer import CameraStabilizer


INPUT_PATH = '../data/raw_videos/AVR-vid-test-mp4.mp4'
OUTPUT_PATH = '../exports/rendered_videos/stable_video.avi'


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    cap = cv2.VideoCapture(INPUT_PATH)
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        print("Could not read input video.")
        return

    height, width = first_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    stabilizer = CameraStabilizer(first_frame)

    try:
        out.write(first_frame)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            stabilized_frame = stabilizer.fix_frame(frame)
            out.write(stabilized_frame)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Stabilized {frame_count // 30} seconds of video...")
    finally:
        # --- THIS PART MUST BE OUTSIDE THE LOOP ---
        cap.release()
        out.release() # This is the most important line!
        cv2.destroyAllWindows()
        print("Video successfully closed and saved.")


if __name__ == "__main__":
    main()
