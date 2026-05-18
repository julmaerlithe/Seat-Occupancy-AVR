import cv2
import json
import os
import numpy as np

# Configuration
VIDEO_PATH = '../data/raw_videos/AVR-vid-test.MOV'  # Update with your file name
OUTPUT_JSON = '../config/seats.json'
MAX_DISPLAY_WIDTH = 1600
MAX_DISPLAY_HEIGHT = 900

points = []
all_seats = {}
current_seat_id = 1
original_img = None
display_img = None
display_scale = 1.0

def rescale_point(point):
    return (int(point[0] * display_scale), int(point[1] * display_scale))


def refresh_display():
    global display_img
    display_img = original_img.copy()
    if display_scale != 1.0:
        display_img = cv2.resize(display_img, (int(original_img.shape[1] * display_scale), int(original_img.shape[0] * display_scale)), interpolation=cv2.INTER_AREA)

    for seat_key, polygon in all_seats.items():
        scaled_poly = np.array([rescale_point(pt) for pt in polygon], dtype=np.int32)
        cv2.polylines(display_img, [scaled_poly], True, (0, 255, 0), 2)
        cv2.putText(display_img, seat_key, tuple(scaled_poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for pt in points:
        cv2.circle(display_img, rescale_point(pt), 3, (0, 0, 255), -1)


def delete_last_label():
    global current_seat_id
    if not all_seats:
        print("No saved seat label to delete.")
        return

    last_key = sorted(all_seats.keys())[-1]
    del all_seats[last_key]
    current_seat_id = int(last_key.split('_')[1]) if last_key.startswith('seat_') else current_seat_id
    print(f"Deleted {last_key}.")
    refresh_display()


def click_event(event, x, y, flags, params):
    global points, current_seat_id
    orig_x = int(x / display_scale)
    orig_y = int(y / display_scale)

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((orig_x, orig_y))
        refresh_display()
        cv2.imshow('Seat Mapper', display_img)
        
        if len(points) == 4:
            seat_key = f"seat_{current_seat_id:03d}"
            all_seats[seat_key] = list(points)
            
            print(f"Saved {seat_key}. Move to next seat.")
            points.clear()
            current_seat_id += 1
            refresh_display()
            cv2.imshow('Seat Mapper', display_img)

# 1. Load Video Frame
cap = cv2.VideoCapture(VIDEO_PATH)
success, original_img = cap.read()
if not success:
    print("Failed to load video.")
    exit()

# Resize the displayed frame if it is too large for a typical screen
scale_w = min(1.0, MAX_DISPLAY_WIDTH / original_img.shape[1])
scale_h = min(1.0, MAX_DISPLAY_HEIGHT / original_img.shape[0])
display_scale = min(scale_w, scale_h)

# 2. Setup Window
cv2.namedWindow('Seat Mapper', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Seat Mapper', click_event)
refresh_display()

print("INSTRUCTIONS:")
print("1. Click the 4 corners of a seat in order (Top-Left, Top-Right, Bottom-Right, Bottom-Left).")
print("2. After 4 clicks, the seat is saved automatically.")
print("3. Press 'd' to delete the last saved seat label.")
print("4. Press 's' to Save all to JSON and Exit.")
print("5. Press 'q' to Quit without saving.")

while True:
    cv2.imshow('Seat Mapper', display_img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):
        delete_last_label()
    elif key == ord('s'):
        # Create config directory if it doesn't exist
        os.makedirs('../config', exist_ok=True)
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(all_seats, f, indent=4)
        print(f"Successfully saved {len(all_seats)} seats to {OUTPUT_JSON}")
        break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()