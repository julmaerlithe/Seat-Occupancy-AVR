import cv2
import numpy as np

def create_dashboard(seat_results, total_seats=125):
    # 1. Increase height slightly to make room for the counter at the top
    width, height = 1000, 480 # Added 80px for the header
    dash_img = np.zeros((height, width, 3), dtype=np.uint8)
    dash_img[:] = (30, 30, 30)  # Professional Dark Grey

    # --- PART A: CALCULATE STATS ---
    occupied_count = sum(1 for status in seat_results.values() if status == True)
    empty_count = total_seats - occupied_count

    # --- PART B: DRAW SUMMARY HEADER ---
    # Draw Occupied Stat
    cv2.putText(dash_img, f"OCCUPIED: {occupied_count}", (40, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (71, 71, 230), 2)
    
    # Draw Empty Stat
    cv2.putText(dash_img, f"EMPTY: {empty_count}", (300, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 179, 113), 2)
    
    # Optional: Add a separator line
    cv2.line(dash_img, (20, 75), (width - 20, 75), (100, 100, 100), 1)

    # --- PART C: DRAW SEAT GRID (Offset by 80px for header) ---
    cols = 25
    rows = 5
    padding = 4
    start_y = 90 # Grid starts below the stats
    cell_w = (width // cols) - padding
    cell_h = ((height - start_y) // rows) - padding

    for i in range(total_seats):
        seat_id = f"seat_{i+1:03}"
        is_occupied = seat_results.get(seat_id, False)

        r = i // cols
        c = i % cols
        x = c * (cell_w + padding) + padding // 2
        y = start_y + r * (cell_h + padding) + padding // 2

        color = (60, 179, 113) if not is_occupied else (71, 71, 230)
        
        # Draw Seat Box
        cv2.rectangle(dash_img, (x, y), (x + cell_w, y + cell_h), color, -1)
        cv2.rectangle(dash_img, (x, y), (x + cell_w, y + cell_h), (255, 255, 255), 1)

        # Center Text
        text = str(i + 1)
        t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.putText(dash_img, text, (x + (cell_w - t_size[0]) // 2, y + (cell_h + t_size[1]) // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return dash_img
