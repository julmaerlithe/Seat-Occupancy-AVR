import cv2
import numpy as np
import json
import os

# --- SETTINGS ---
VIDEO_PATH = '../data/raw_videos/AVR-vid-test-mp4.mp4'
SEATS_PATH = '../config/seats.json'
OUTPUT_DIR = '../data/dataset'
FRAME_SKIP = 60  # Capture seat images every 60 frames (approx every 2 seconds)

# Create folders
for label in ['empty', 'occupied']:
    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)

with open(SEATS_PATH, 'r') as f:
    seats = json.load(f)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0
saved_count = 0

print("Starting data collection. Press 'q' to stop early.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Only capture every X frames so we get different people/lighting
    if frame_count % FRAME_SKIP == 0:
        for seat_id, coords in seats.items():
            # Get the square bounding box for the seat
            pts = np.array(coords, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            
            # Crop the seat from the frame
            # We add a tiny buffer so it's not too tight
            crop = frame[y:y+h, x:x+w]
            
            if crop.size > 0:
                # For now, we save everything to 'empty'. 
                # You will manually move the 'occupied' ones later!
                img_name = f"{seat_id}_frame_{frame_count}.jpg"
                save_path = os.path.join(OUTPUT_DIR, 'empty', img_name)
                cv2.imwrite(save_path, crop)
                saved_count += 1
        
        print(f"Processed frame {frame_count}... Saved {saved_count} images so far.")

    frame_count += 1
    cv2.imshow('Collecting Data...', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Done! Check the folder: {OUTPUT_DIR}/empty")