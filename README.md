# Object-detection
# Object Detection with YOLO and OpenCV

This project demonstrates object detection using **YOLO (You Only Look Once)** and **OpenCV** for image processing. The provided scripts leverage pre-trained YOLO models to detect objects in images and live video streams.

## Files

### 1. **YOLO Static Image Detection**

This script is used for object detection on a **single image**. It loads the pre-trained YOLOv4 model and processes the input image to detect objects. The detected objects are highlighted with bounding boxes and labeled with the object class name.

- **Key Features:**
  - Loads and processes static images using YOLOv4.
  - Displays the image with bounding boxes and object labels.
  - Randomly assigns colors to different object classes for visual distinction.

### 2. **YOLO with ROS Integration **

This script performs object detection on a **real-time video stream** by integrating YOLO with **ROS (Robot Operating System)**. It subscribes to an image feed from a ROS topic (e.g., a webcam) and performs object detection on each frame.

- **Key Features:**
  - Subscribes to a ROS topic for continuous image stream processing.
  - Uses `CvBridge` to convert ROS images to OpenCV format.
  - Displays real-time object detection results with bounding boxes and labels.

## Note

The rest of the project code, including specific configurations and further details, cannot be provided as it belongs to **DJS Antariksh**. For any inquiries or access to the complete codebase, Please reach out.
