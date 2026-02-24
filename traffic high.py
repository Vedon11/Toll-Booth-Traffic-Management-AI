from ultralytics import YOLO
import cv2
import cvzone
import math

# --- 1. SETUP ---
# Use a video file for consistent testing (replace 'cars.mp4' with your video file name)
# If using a webcam, change 'cars.mp4' to 0 (cap = cv2.VideoCapture(0))
cap = cv2.VideoCapture('/Users/apurvaghare/My Files/Code/python/object detection/Main/traffic management/car11.mp4') 
# cap = cv2.VideoCapture(0) # For webcam

# Set resolution (may not work on all systems/videos)
# cap.set(3, 1270)
# cap.set(4, 720)

model = YOLO('yolov8n.pt')

# Filtered Class Names for Vehicles
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase","ID",
              "teddy bear", "PEN", "toothbrush"
              ]
vehicle_classes = ['car', 'truck', 'bus', 'motorbike'] # Target classes for counting

# --- 2. TRAFFIC MANAGEMENT PARAMETERS ---
# Define the coordinates for the 'Yellow Line'.
# Previously this was a horizontal line at a fixed y; change to a vertical line
# by specifying an x-coordinate (yellow_line_x). The line will be drawn from
# near the top to near the bottom of the frame at that x position.
yellow_line_x = 1500
line_color = (0, 255, 255) # BGR for Yellow

# Traffic threshold (number of vehicles detected crossing the line to trigger release)
TRAFFIC_THRESHOLD = 34

# Simple vehicle count past the line (needs proper tracking for accurate counting)
current_traffic_count = 0 

# To avoid counting the same vehicle in consecutive frames, you'd need a tracker here.
# For this example, we'll use a simple reset mechanism (not suitable for real-time traffic flow)

# --- 3. MAIN LOOP ---
while True:
    success, img = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Restart video if end reached
        current_traffic_count = 0 # Reset count on video loop
        continue

    # Get image dimensions to set line coordinates dynamically if needed
    h, w, _ = img.shape
    # Vertical line: x fixed, y varies from top (50) to bottom (h-50)
    line_start = (yellow_line_x, 50)
    line_end = (yellow_line_x, h - 50)

    # Draw the yellow line
    cv2.line(img, line_start, line_end, line_color, 3)

    results = model(img, stream=True)
    
    # Reset count for the current frame to simulate an instant density check (needs tracker for accumulation)
    frame_vehicle_count = 0 

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            # Class Name and ID
            cls = int(box.cls[0])
            class_name = classNames[cls]

            # Only process if confidence is high enough and it's a vehicle class
            if conf > 0.5 and class_name in vehicle_classes:
                # Bounding Box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w_box, h_box = x2 - x1, y2 - y1
                
                # Calculate the center point of the bounding box
                cx, cy = x1 + w_box // 2, y1 + h_box // 2

                # Draw the bounding box and label
                cvzone.cornerRect(img, (x1, y1, w_box, h_box), l=9, rt=2)
                cvzone.putTextRect(img, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=1.5, thickness=2, offset=10)
                
                # Draw center point
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                # --- 4. LINE CROSSING LOGIC (Simple check without tracking) ---
                # Check if the center point is past the vertical yellow line
                # (vehicle considered past the line when its center x coordinate
                # is to the left of the line)
                if cx < yellow_line_x:
                    frame_vehicle_count += 1
                    # Change box color to indicate it's past the line (e.g., Red)
                    cvzone.cornerRect(img, (x1, y1, w_box, h_box), colorR=(0, 0, 255), l=9, rt=2) 
    
    # Update the overall count (needs proper tracking for accumulation/true counting)
    current_traffic_count = frame_vehicle_count

    # --- 5. THRESHOLD LOGIC ---
    if current_traffic_count >= TRAFFIC_THRESHOLD:
        release_status = "RELEASE TOLL BOOTH (HIGH TRAFFIC)"
        status_color = (0, 0, 255) # Red for high traffic
    else:
        release_status = "Toll Booth Open"
        status_color = (0, 255, 0) # Green for low traffic

    # Display the traffic count and status
    cvzone.putTextRect(img, f'Count Past Line: {current_traffic_count}', (50, 80), scale=2, thickness=2, offset=10)
    # Increased vertical gap between the count indicator and the release status
    cvzone.putTextRect(img, release_status, (50, 220), scale=3, thickness=3, offset=15, colorR=status_color, font=cv2.FONT_HERSHEY_DUPLEX)
    cv2.imshow("Traffic Management System", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()