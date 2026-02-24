from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import os

# source can be a video file, camera index (0,1,...) or an image file
source = '/Users/apurvaghare/My Files/Code/python/object detection/Main/traffic management/car11.mp4'  # Example video file path
cap = cv2.VideoCapture(source)
cap.set(3, 1270)
cap.set(4, 720)

model = YOLO('../yolo-weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# prev_frame_time = 0
# new_frame_time = 0

while True:
    success, img = cap.read()
    # if no frame was read (end of video or bad path), stop the loop
    if not success or img is None:
        print("No frame received from source - exiting loop.")
        break

    results = model(img, stream=True)
    # Count cars in this frame
    car_count = 0
    for r in results:
        boxes = r.boxes
        for box in boxes:

            #Bouncing Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,255), 3)
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            
            #confidence
            conf = math.ceil((box.conf[0]*100))/100
            # print(conf)
            # cvzone.putTextRect(img, f'{conf}', (max(0, x1), max(35, y1)))
            
            #classname
            cls = int(box.cls[0])
            class_name = classNames[cls]
            # increment car counter when detected
            if class_name.lower() == 'car':
                car_count += 1

            cvzone.putTextRect(img, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=2, thickness=2)
    

    # Draw car counter on top-left
    cvzone.putTextRect(img, f'Cars: {car_count}', (20, 40), scale=2, thickness=2, offset=10)

    cv2.imshow("Image", img)

    # If the source is a static image file, keep the window open until a key is pressed.
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    is_image = os.path.isfile(source) and os.path.splitext(source)[1].lower() in image_exts

    if is_image:
        cv2.waitKey(0)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
