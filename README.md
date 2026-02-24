🚦 Toll-Booth Traffic Management AI
An intelligent Computer Vision system built to modernize toll booth operations through real-time traffic monitoring and automated congestion control. This project utilizes YOLOv8 (Nano) and OpenCV to detect vehicles, analyze queue density, and trigger automated toll release decisions when congestion exceeds a defined safety limit.
Unlike traditional toll systems that rely on manual supervision or ground sensors, this solution performs visual queue assessment using spatial analysis. A virtual boundary (“Yellow Line”) is defined within the video frame, and vehicle centroids are tracked to determine real-time occupancy in the congestion zone.
When the number of vehicles beyond this boundary crosses a configurable threshold, the system activates a “Release Toll Booth” state—simulating a smart signal to reduce traffic buildup and improve throughput efficiency.

🔍 Core Features
Real-time vehicle detection (cars, buses, trucks, motorcycles)
Virtual congestion boundary using coordinate-based logic
Dynamic queue density monitoring
Automated toll release trigger mechanism
Live visual feedback with bounding boxes and status HUD
Lightweight YOLOv8n model for high FPS performance

🛠️ Technology Stack
Python 3.x
YOLOv8 (Ultralytics) – Object Detection Engine
OpenCV – Frame processing and visualization
cvzone – Enhanced bounding box rendering
COCO Pre-trained Dataset – Vehicle class detection

⚙️ System Workflow
Vehicle Detection – YOLOv8n detects objects in each video frame.
Class Filtering – Only vehicle categories are processed.
Spatial Evaluation – Vehicle centroids are compared against a predefined Y-axis boundary (yellow_line_y).
Queue Calculation – Vehicles beyond the boundary are counted as part of the congestion zone.
Decision Logic –
Below threshold → Normal operation
At/Above threshold → “RELEASE TOLL BOOTH” triggered

📈 Future Enhancements
DeepSORT-based multi-object tracking
ANPR (Automatic Number Plate Recognition) integration
Cloud-based analytics dashboard for traffic insights
Hardware-level toll gate signal integration
