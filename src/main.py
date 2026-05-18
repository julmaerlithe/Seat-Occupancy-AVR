import cv2
import json
import os
import pickle
import numpy as np
from dashboard import create_dashboard

# Paths
VIDEO_PATH = '../exports/rendered_videos/stable_video.avi'
SEATS_PATH = '../config/seats.json'

def extract_features(img):
    img = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # HOG captures the "shape" of a person (head/shoulders) 
    # instead of just the color of the seat
    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    
    hist = hog.compute(gray)
    return hist.flatten()

def main():
    # 1. Load the 125 seats you just labeled
    with open(SEATS_PATH, 'r') as f:
        seats = json.load(f)

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, first_frame = cap.read()
    if not ret: return

    # Create named windows that can be resized
    cv2.namedWindow('Auditorium Feed (AI Optimized)', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Generic Occupancy Dashboard', cv2.WINDOW_NORMAL)

    # Optional: Force them to specific sizes
    cv2.resizeWindow('Auditorium Feed (AI Optimized)', 1280, 720)
    cv2.resizeWindow('Generic Occupancy Dashboard', 800, 600)

    with open('../config/model.pkl', 'rb') as f:
        model = pickle.load(f)

    def get_ai_prediction(img, model):
        features = extract_features(img).reshape(1, -1)
        probabilities = model.predict_proba(features)[0]
        return probabilities[1]  # Return confidence for occupied

    seat_ids = list(seats.keys())
    current_index = 0
    # We use a 'buffer' to store recent results for every seat
    # 0 = Definitely Empty, 20 = Definitely Occupied
    seat_memory = {s_id: 0 for s_id in seats.keys()}
    seat_results = {s_id: False for s_id in seat_ids}

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- SPEED FIX: Only check a small batch (10 seats) per frame ---
        batch_size = 10
        for i in range(batch_size):

            target_id = seat_ids[current_index]
            coords = seats[target_id]
            
            # Crop and Predict
            pts = np.array(coords, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            seat_crop = frame[y:y+h, x:x+w]
            
            if seat_crop.size > 0:
                # Get the probability instead of just True/False
                # This asks the AI: "How sure are you (0.0 to 1.0)?"
                occupied_confidence = get_ai_prediction(seat_crop, model)

                # Add this inside the seat loop temporarily
                if occupied_confidence > 0.1: # Only print if there's a tiny bit of hope
                    print(f"Seat {target_id} Confidence: {occupied_confidence:.2f}")

                # --- STRENGHTENED LOGIC ---
                if occupied_confidence > 0.85:  # Higher bar to turn red
                    seat_memory[target_id] = min(25, seat_memory[target_id] + 5)
                
                elif occupied_confidence < 0.20: # Fast reset for definitely empty seats
                    seat_memory[target_id] = max(0, seat_memory[target_id] - 8)

                # Only show red if memory is very high
                seat_results[target_id] = seat_memory[target_id] >= 15

            # Cycle through the 125 seats
            current_index = (current_index + 1) % len(seat_ids)

        # --- DISPLAY OPTIMIZATION ---
        display_frame = frame.copy()
        for s_id, occupied in seat_results.items():
            color = (0, 0, 255) if occupied else (0, 255, 0)
            # Use a thinner line (1) to speed up drawing
            cv2.polylines(display_frame, [np.array(seats[s_id], np.int32)], True, color, 1)

        # Update Dashboard and Show
        dash_img = create_dashboard(seat_results)

        cv2.imshow('Auditorium Feed (AI Optimized)', display_frame)
        cv2.imshow('Generic Occupancy Dashboard', dash_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()