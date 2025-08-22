#!/usr/bin/env python3
"""
Raspberry Pi Client - YOLOv8 Human Detection & Bounding Box Sender
Sends bounding box data to laptop server via WebSocket
"""

import cv2
import numpy as np
import json
import time
from datetime import datetime
from collections import deque
import socketio
from ultralytics import YOLO
import threading
import logging
import base64
import os
from dotenv import load_dotenv

# Load environment variables from local .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PiClient:
    def __init__(self, server_url=None, sequence_mode=True, sequence_length=61):
        """
        Initialize Raspberry Pi Client
        
        Args:
            server_url: URL of the laptop server
            sequence_mode: If True, send sequences of bounding boxes. If False, send per frame
            sequence_length: Number of frames in a sequence
        """
        self.server_url = server_url or os.getenv('SERVER_URL', 'http://192.168.100.235:5001')
        self.sequence_mode = sequence_mode
        self.sequence_length = sequence_length
        
        # Initialize camera
        self.cap = None
        self.frame_id = 0
        
        # Initialize YOLO model
        self.model = None
        
        # Initialize SocketIO client
        self.sio = socketio.Client()
        self.connected = False
        
        # Buffer for sequence mode
        self.bbox_buffer = deque(maxlen=sequence_length)
        self.timestamp_buffer = deque(maxlen=sequence_length)
        
        # Setup SocketIO events
        self.setup_socketio_events()
        
    def setup_socketio_events(self):
        """Setup SocketIO event handlers"""
        @self.sio.event
        def connect():
            logger.info("Connected to server")
            self.connected = True
            
        @self.sio.event
        def disconnect():
            logger.info("Disconnected from server")
            self.connected = False
            
        @self.sio.event
        def connect_error(data):
            logger.error(f"Connection failed: {data}")
            
    def initialize_camera(self):
        """Initialize camera (USB or PiCamera)"""
        try:
            # Try PiCamera first (for Raspberry Pi camera module)
            try:
                from picamera2 import Picamera2
                self.picam = Picamera2()
                config = self.picam.create_preview_configuration(
                    main={"size": (640, 480)}
                )
                self.picam.configure(config)
                self.picam.start()
                logger.info("PiCamera initialized successfully")
                self.camera_type = "picamera"
                return True
            except ImportError:
                logger.info("PiCamera2 not available, trying USB camera")
            
            # Fallback to USB camera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Cannot open USB camera")
                
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            
            logger.info("USB Camera initialized successfully")
            self.camera_type = "usb"
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return False
            
    def initialize_model(self):
        """Initialize YOLOv8 model"""
        try:
            # Load custom trained YOLOv8 model
            model_path = os.getenv('YOLO_MODEL_PATH', os.path.join(os.path.dirname(__file__), 'yolov8n.pt'))
            self.model = YOLO(model_path)
            logger.info(f"Custom YOLOv8 model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            # Fallback to default model if custom model fails
            try:
                fallback_path = os.path.join(os.path.dirname(__file__), 'yolov8n.pt')
                self.model = YOLO(fallback_path)
                logger.info(f"Fallback: YOLOv8 model loaded successfully from {fallback_path}")
                return True
            except Exception as fallback_e:
                logger.error(f"Failed to load fallback model: {fallback_e}")
                return False
            
    def connect_to_server(self):
        """Connect to the laptop server"""
        try:
            self.sio.connect(self.server_url)
            return True
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return False
            
    def detect_humans(self, frame):
        """Detect humans in frame and return bounding boxes"""
        try:
            # Run YOLOv8 inference
            results = self.model(frame, verbose=False)
            
            human_detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Check if detection is a person (class 0 in COCO dataset)
                        if int(box.cls[0]) == 0:  # person class
                            confidence = float(box.conf[0])
                            
                            # Only include detections with confidence >= 0.5
                            if confidence >= 0.5:
                                # Get bounding box coordinates
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                
                                detection = {
                                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                    "confidence": confidence
                                }
                                human_detections.append(detection)
                                
            return human_detections
            
        except Exception as e:
            logger.error(f"Error in human detection: {e}")
            return []
            
    def send_frame_data(self, detections, timestamp):
        """Send per-frame detection data"""
        if not self.connected or not detections:
            return
            
        for detection in detections:
            data = {
                "frame_id": self.frame_id,
                "bbox": detection["bbox"],
                "confidence": detection["confidence"],
                "timestamp": timestamp
            }
            
            try:
                self.sio.emit("bbox_data", data)
                logger.debug(f"Sent frame data: {data}")
            except Exception as e:
                logger.error(f"Failed to send frame data: {e}")
                
    def send_sequence_data(self):
        """Send sequence of bounding boxes"""
        if not self.connected or len(self.bbox_buffer) < self.sequence_length:
            return
            
        data = {
            "sequence": list(self.bbox_buffer),
            "timestamps": list(self.timestamp_buffer)
        }
        
        try:
            self.sio.emit("bbox_data", data)
            logger.debug(f"Sent sequence data with {len(self.bbox_buffer)} frames")
        except Exception as e:
            logger.error(f"Failed to send sequence data: {e}")
            
    def run(self):
        """Main processing loop"""
        logger.info("Starting Pi Client...")
        
        # Initialize components
        if not self.initialize_camera():
            return
            
        if not self.initialize_model():
            return
            
        if not self.connect_to_server():
            return
            
        logger.info("All components initialized. Starting detection loop...")
        
        try:
            while True:
                # Read frame based on camera type
                if hasattr(self, 'camera_type') and self.camera_type == "picamera":
                    try:
                        frame = self.picam.capture_array()
                        ret = True
                    except Exception as e:
                        logger.warning(f"Failed to read PiCamera frame: {e}")
                        ret = False
                        frame = None
                else:
                    ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    logger.warning("Failed to read frame")
                    continue
                    
                self.frame_id += 1
                timestamp = datetime.now().isoformat()
                
                # Detect humans in frame
                detections = self.detect_humans(frame)
                
                # Draw bounding boxes on frame for visualization
                for detection in detections:
                    bbox = detection["bbox"]
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # Encode frame as JPEG and convert to base64
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send video frame to server
                if self.connected:
                    try:
                        self.sio.emit('video_frame', {
                            'frame': frame_base64,
                            'timestamp': timestamp
                        })
                    except Exception as e:
                        logger.error(f"Failed to send video frame: {e}")
                
                # Send heartbeat every 30 frames (~1 second) to maintain connection status
                if self.frame_id % 30 == 0 and self.connected:
                    try:
                        self.sio.emit('heartbeat', {
                            'timestamp': timestamp,
                            'frame_id': self.frame_id,
                            'status': 'active',
                            'detections_count': len(detections)
                        })
                        logger.debug(f"Sent heartbeat at frame {self.frame_id}")
                    except Exception as e:
                        logger.error(f"Failed to send heartbeat: {e}")
                
                if detections:
                    if self.sequence_mode:
                        # Add to buffer for sequence mode
                        # Use the first detection's bbox for simplicity
                        self.bbox_buffer.append(detections[0]["bbox"])
                        self.timestamp_buffer.append(timestamp)
                        
                        # Send sequence when buffer is full
                        if len(self.bbox_buffer) == self.sequence_length:
                            self.send_sequence_data()
                    else:
                        # Send immediately for per-frame mode
                        self.send_frame_data(detections, timestamp)
                else:
                    # Send empty bbox data periodically to maintain connection
                    if not self.sequence_mode and self.frame_id % 60 == 0 and self.connected:
                        try:
                            empty_data = {
                                "frame_id": self.frame_id,
                                "bbox": [0, 0, 0, 0],
                                "confidence": 0.0,
                                "timestamp": timestamp,
                                "no_detection": True
                            }
                            self.sio.emit("bbox_data", empty_data)
                            logger.debug(f"Sent empty bbox data at frame {self.frame_id}")
                        except Exception as e:
                            logger.error(f"Failed to send empty bbox data: {e}")
                        
                # Optional: Display frame with detections (for debugging)
                # Uncomment the following lines if you want to see the video feed
                # for detection in detections:
                #     bbox = detection["bbox"]
                #     cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                # cv2.imshow('Pi Client', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                    
                # Small delay to control FPS
                time.sleep(0.033)  # ~30 FPS
                
        except KeyboardInterrupt:
            logger.info("Stopping Pi Client...")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources"""
        # Cleanup camera resources
        if hasattr(self, 'camera_type') and self.camera_type == "picamera":
            if hasattr(self, 'picam'):
                try:
                    self.picam.stop()
                    self.picam.close()
                except Exception as e:
                    logger.error(f"Error stopping PiCamera: {e}")
        elif hasattr(self, 'cap') and self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        if self.connected:
            self.sio.disconnect()
        logger.info("Cleanup completed")

def main():
    # Configuration from environment variables
    SERVER_URL = os.getenv('SERVER_URL', 'http://192.168.100.235:5001')
    SEQUENCE_MODE = os.getenv('SEQUENCE_MODE', 'True').lower() == 'true'
    SEQUENCE_LENGTH = int(os.getenv('SEQUENCE_LENGTH', '61'))
    
    # Create and run client
    client = PiClient(
        server_url=SERVER_URL,
        sequence_mode=SEQUENCE_MODE,
        sequence_length=SEQUENCE_LENGTH
    )
    

    
    client.run()

if __name__ == "__main__":
    main()