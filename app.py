import cv2
import json
import webbrowser
import keyboard
import os
import signal
from threading import Timer
from flask import Flask, render_template, jsonify
from ultralytics import YOLO

# Load configuration
with open("config.json", "r") as config_file:
    config = json.load(config_file)

# Load YOLO model
model = YOLO('yolo_model/yolov8s.pt')  # Use a YOLOv8 model pre-trained or fine-tuned for vehicles
class_names = model.names  # Get class names from the model

app = Flask(__name__)

class TrafficSignalController:
    def __init__(self, images, min_green, max_green):
        self.images = images  # List of image paths
        self.min_green = min_green
        self.max_green = max_green
        self.vehicle_counts = []
        self.time_allocations = []
        self.vehicle_types = []

    def detect_vehicles(self):
        """Detect vehicles in each lane using the YOLO model."""
        self.vehicle_counts = []
        self.vehicle_types = []  # To track vehicle types (e.g., cars, trucks, buses)

        for image_path in self.images:
            img = cv2.imread(image_path)

            if img is None:
                print(f"Error loading image: {image_path}")
                self.vehicle_counts.append(0)
                self.vehicle_types.append({})
                continue

            # Preprocess the image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = model.predict(source=img, conf=0.5)  # Adjust confidence threshold if needed

            # Extract detection data
            detected_objects = results[0].boxes.data if results[0].boxes is not None else []

            detected_objects = [
                obj for obj in detected_objects if int(obj[-1]) in class_names and class_names[int(obj[-1])] in ['car', 'truck', 'bus', 'motorcycle']
            ]

            # Count vehicles per type
            vehicle_count = len(detected_objects)
            self.vehicle_counts.append(vehicle_count)

            # Track types of vehicles
            vehicle_type_counts = {}
            for obj in detected_objects:
                class_name = class_names[int(obj[-1])]
                vehicle_type_counts[class_name] = vehicle_type_counts.get(class_name, 0) + 1
            self.vehicle_types.append(vehicle_type_counts)

    def calculate_time_allocations(self):
        """Allocate traffic signal time based on vehicle count per lane."""
        self.time_allocations = []

        for count in self.vehicle_counts:
            if count <= 5:
                allocated_time = 5
            elif 5 < count <= 10:
                allocated_time = 8
            elif 10 < count <= 20:
                allocated_time = 15
            elif 20 < count <= 30:
                allocated_time = 25
            elif 30 < count <= 40:
                allocated_time = 35
            elif 40 < count < 50:
                allocated_time = 45
            elif count >= 50:
                allocated_time = 60
            else:
                allocated_time = self.min_green  # Default fallback to minimum green time

            self.time_allocations.append(allocated_time)

    def get_signal_data(self):
        """Return the vehicle counts, vehicle types, and time allocations for each lane."""
        return {
            "vehicle_counts": self.vehicle_counts,
            "vehicle_types": self.vehicle_types,
            "time_allocations": self.time_allocations,
        }

# Initialize the controller with images and settings
controller = TrafficSignalController(
    images=config["images"],  # Paths to lane images
    min_green=config["min_green"],
    max_green=config["max_green"],
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/signal_data")
def signal_data():
    controller.detect_vehicles()
    controller.calculate_time_allocations()
    return jsonify(controller.get_signal_data())

def open_browser():
    """Open the web browser to the Flask app."""
    webbrowser.open_new("http://127.0.0.1:5000/")  # Open the Flask app in the browser

def shutdown_server():
    """Shutdown the Flask development server."""
    os.kill(os.getpid(), signal.SIGINT)  # Send SIGINT to stop Flask server

if __name__ == "__main__":
    # Use a timer to delay the browser opening until the Flask app starts
    Timer(1, open_browser).start()  # Delay by 1 second to ensure the server is running

    # Run the Flask app in a separate thread or process
    try:
        # Start the Flask app
        app.run(debug=True, use_reloader=False)  # use_reloader=False to prevent multiple runs
    except KeyboardInterrupt:
        print("Flask server stopped.")

    # Check for a key press to stop the server
    print("Press 'q' to stop the server.")
    while True:
        if keyboard.is_pressed('q'):  # Press 'q' to quit
            print("Shutting down...")
            shutdown_server()
            break
