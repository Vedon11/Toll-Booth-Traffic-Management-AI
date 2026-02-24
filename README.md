🚦 AI-Powered Smart Toll Management System
An innovative Computer Vision solution designed to automate toll booth operations. By leveraging YOLOv8 and OpenCV, this system monitors traffic density in real-time and triggers automated "Release" signals when vehicle queues exceed a predefined safety threshold (the "Yellow Line").

<p align="center">
  <kbd>
    <img src="working/demo.gif" width="600">
  </kbd>
  <br>
  <em>Real-time vehicle detection and toll-release logic in action</em>
</p>

🌟 The Innovation
Traditional toll booths often rely on manual oversight or simple pressure sensors, which don't account for visual queue length or vehicle types.

This project introduces:

Virtual Boundary Detection: A coordinate-based "Yellow Line" that acts as a digital trigger.

Real-time Density Analysis: Instead of just counting total cars, it calculates how many vehicles are currently occupying the "congested zone."

Automated Decision Logic: An algorithmic trigger that instructs the toll system to "Release" traffic when the threshold is hit, preventing gridlock.

🛠️ Tech Stack
AI Model: YOLOv8 (You Only Look Once) - Nano version for high FPS.

Language: Python 3.x

Libraries: * ultralytics: For the object detection engine.

OpenCV: For video frame processing and UI overlays.

cvzone: For optimized bounding boxes and text rendering.

math: For precision confidence calculation.

🚀 How It Works
1. Object Detection & Filtering

The system initializes the YOLOv8n model, pre-trained on the COCO dataset. To ensure accuracy, the script specifically filters for vehicle classes (cars, trucks, buses, motorbikes), ignoring irrelevant objects like pedestrians or birds.

2. Spatial Mapping (The "Yellow Line")

We define a mathematical threshold on the Y-axis of the video frame:

Python
yellow_line_y = 450 # The digital 'Toll Gate' boundary
The system monitors the Centroid (cx, cy) of every detected vehicle.

3. Logic & Trigger Mechanism

Detection: If cy > yellow_line_y, the vehicle is marked as "In Queue."

Counting: The script tallies all vehicles currently past the line in the current frame.

Action: * If Count < Threshold: Status is Normal.

If Count >= Threshold: Status changes to "RELEASE TOLL BOOTH", simulating a signal sent to the gate hardware.

📸 Implementation Preview
The UI provides immediate visual feedback:

Blue Boxes: Vehicles detected.

Red Boxes: Vehicles that have crossed the congestion line.

HUD (Heads-Up Display): Real-time count and release status displayed at the top-left.

🏗️ Installation & Usage
Clone the repository:

Bash
git clone https://github.com/yourusername/smart-toll-yolo.git
Install dependencies:

Bash
pip install ultralytics opencv-python cvzone
Run the application:

Bash
python main.py
🔮 Future Roadmap
DeepSORT Integration: Adding unique ID tracking to count total daily traffic flow.

ANPR Integration: Automatically reading license plates of vehicles crossing the line.

Cloud Logging: Sending congestion data to a dashboard for city planners.
