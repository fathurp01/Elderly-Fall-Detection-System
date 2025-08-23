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
from queue import Queue, Full, Empty
import multiprocessing
import socket
import struct
import hashlib
from functools import lru_cache
import subprocess
try:
    import psutil
except ImportError:
    psutil = None
    logger.warning("psutil not available, system monitoring disabled")

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
        self.server_url = server_url or os.getenv('SERVER_URL', 'http://10.243.149.68:5001/')
        self.sequence_mode = sequence_mode
        self.sequence_length = sequence_length
        
        # Initialize camera
        self.cap = None
        self.frame_id = 0
        
        # Initialize YOLO model
        self.model = None
        
        # Frame skipping & caching optimization for RPi 4
        self.inference_interval = 3  # Process every 3rd frame for RPi 4
        self.frame_counter = 0
        self.last_detections = []
        self.detection_cache_time = time.time()
        self.cache_duration = 0.15  # 150ms cache for RPi 4
        
        # Conditional encoding optimization
        self.last_frame_encoded = 0
        self.encoding_interval = 5  # Encode every 5th frame when no detection
        
        # Initialize SocketIO client
        self.sio = socketio.Client()
        self.connected = False
        
        # Buffer for sequence mode
        self.bbox_buffer = deque(maxlen=sequence_length)
        self.timestamp_buffer = deque(maxlen=sequence_length)
        
        # System detection and adaptive configuration
        self.cpu_count = multiprocessing.cpu_count()
        self.available_memory = self._get_available_memory()
        
        # Adaptive threading based on hardware
        if self.cpu_count == 1:
            self.use_threading = False
            self.queue_size = 5  # Minimal queue for single-core
            logger.warning("Single-core detected, using synchronous mode")
        else:
            self.use_threading = True
            # Dynamic queue sizing based on memory
            if self.available_memory < 512 * 1024 * 1024:  # < 512MB
                self.queue_size = 10
            elif self.available_memory < 1024 * 1024 * 1024:  # < 1GB
                self.queue_size = 20
            else:
                self.queue_size = 30
            logger.info(f"Multi-core detected ({self.cpu_count} cores), using threading with queue size {self.queue_size}")
        
        # Threading optimization - Non-blocking I/O
        self.send_queue = Queue(maxsize=self.queue_size)
        self.running = True
        self.sender_thread = None
        
        # Connection resilience
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5
        
        # Network monitoring for adaptive compression
        self.network_latency = 0
        self.packet_loss_rate = 0
        self.last_ping_time = 0
        self.ping_interval = 5  # seconds
        
        # UDP socket for video streaming (fallback)
        self.udp_socket = None
        self.use_udp_video = False
        self.udp_port = 5002
        
        # Hardware encoder detection
        self.hardware_encoder_available = self._detect_hardware_encoder()
        
        # Smart caching
        self.frame_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Adaptive compression parameters
        self.base_jpeg_quality = 65
        self.min_jpeg_quality = 30
        self.max_jpeg_quality = 85
        self.adaptive_resize_factor = 0.5
        
        # Performance monitoring
        self.frame_stats = {
            'total_frames': 0,
            'dropped_frames': 0,
            'queue_overflows': 0,
            'last_health_check': time.time()
        }
        
        # Setup SocketIO events
        self.setup_socketio_events()
        
    def setup_socketio_events(self):
        """Setup SocketIO event handlers with connection resilience"""
        @self.sio.event
        def connect():
            logger.info("Connected to server")
            self.connected = True
            self.reconnect_attempts = 0  # Reset reconnect counter
            
        @self.sio.event
        def disconnect():
            logger.info("Disconnected from server")
            self.connected = False
            # Auto-reconnect with exponential backoff
            if self.running and self.reconnect_attempts < self.max_reconnect_attempts:
                self.reconnect_attempts += 1
                delay = min(self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), 60)
                logger.info(f"Attempting reconnect #{self.reconnect_attempts} in {delay} seconds...")
                threading.Timer(delay, self._attempt_reconnect).start()
            
        @self.sio.event
        def connect_error(data):
            logger.error(f"Connection failed: {data}")
            self.connected = False
            
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
        """Initialize YOLOv8 model with RPi 4 optimizations"""
        try:
            # Load custom trained YOLOv8 model
            model_path = os.getenv('YOLO_MODEL_PATH', os.path.join(os.path.dirname(__file__), 'yolov8n.pt'))
            self.model = YOLO(model_path)
            
            # Model optimization for Raspberry Pi 4
            self.model.to('cpu')  # Ensure CPU usage
            
            # Optimize PyTorch for RPi 4 (4 cores)
            try:
                import torch
                # Use 2 threads for inference on RPi 4 (half of available cores)
                torch.set_num_threads(2)
                # Enable optimizations
                torch.set_num_interop_threads(1)
                logger.info(f"PyTorch optimized: {torch.get_num_threads()} inference threads, {torch.get_num_interop_threads()} interop threads")
            except ImportError:
                logger.warning("PyTorch optimization not available")
            
            logger.info(f"YOLOv8n model loaded and optimized for RPi 4 from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            # Fallback to default model if custom model fails
            try:
                fallback_path = os.path.join(os.path.dirname(__file__), 'yolov8n.pt')
                self.model = YOLO(fallback_path)
                self.model.to('cpu')
                
                # Apply same optimizations to fallback
                try:
                    import torch
                    torch.set_num_threads(2)
                    torch.set_num_interop_threads(1)
                except ImportError:
                    pass
                    
                logger.info(f"Fallback: YOLOv8n model loaded and optimized for RPi 4 from {fallback_path}")
                return True
            except Exception as fallback_e:
                logger.error(f"Failed to load fallback model: {fallback_e}")
                return False
            
    def connect_to_server(self):
        """Connect to the laptop server"""
        try:
            self.sio.connect(self.server_url)
            # Setup UDP socket for video streaming if needed
            if self._setup_udp_socket():
                self.use_udp_video = True
                logger.info("UDP video streaming enabled")
            # Start sender thread after successful connection
            self.start_sender_thread()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return False
            
    def start_sender_thread(self):
        """Start the data sender thread"""
        if self.sender_thread is None or not self.sender_thread.is_alive():
            self.sender_thread = threading.Thread(
                target=self.sender_worker,
                name="DataSender",
                daemon=True
            )
            self.sender_thread.start()
            logger.info("Data sender thread started")
            
    def sender_worker(self):
        """Worker thread untuk mengirim data secara asynchronous"""
        logger.info("[DataSender] Worker thread started")
        while self.running and self.connected:
            try:
                # Ambil data dari queue dengan timeout
                data = self.send_queue.get(timeout=1)
                
                if not self.connected:
                    break
                    
                # Kirim berdasarkan tipe data
                if data['type'] == 'video_frame':
                    self.sio.emit('video_frame', data['payload'])
                elif data['type'] == 'bbox_data':
                    self.sio.emit('bbox_data', data['payload'])
                elif data['type'] == 'heartbeat':
                    self.sio.emit('heartbeat', data['payload'])
                    
                self.send_queue.task_done()
                logger.debug(f"[DataSender] Sent {data['type']} data")
                
            except Empty:
                continue  # Normal timeout, lanjutkan loop
            except Exception as e:
                logger.error(f"[DataSender] Error sending data: {e}")
                
        logger.info("[DataSender] Worker thread stopped")
        
    def queue_data(self, data_type, payload):
        """Menambahkan data ke queue untuk dikirim dengan adaptive behavior"""
        if not self.connected or not self.running:
            return False
        
        # Fallback to synchronous mode for single-core systems
        if not self.use_threading:
            try:
                self.sio.emit(data_type, payload)
                return True
            except Exception as e:
                logger.error(f"Direct send failed: {e}")
                return False
            
        try:
            self.send_queue.put({
                'type': data_type,
                'payload': payload
            }, block=False)  # Non-blocking put
            return True
            
        except Full:
            self.frame_stats['queue_overflows'] += 1
            logger.warning(f"Send queue full, dropping {data_type} data (overflow #{self.frame_stats['queue_overflows']})")
            
            # Adaptive quality reduction for video frames
            if data_type == 'video_frame' and self.send_queue.qsize() > self.queue_size * 0.8:
                # Reduce JPEG quality when queue is congested
                if 'frame' in payload:
                    payload = self._reduce_frame_quality(payload)
            
            # Drop oldest data and try again
            try:
                dropped_data = self.send_queue.get_nowait()
                self.frame_stats['dropped_frames'] += 1
                logger.debug(f"Dropped {dropped_data['type']} data to make room")
                
                self.send_queue.put({
                    'type': data_type,
                    'payload': payload
                }, block=False)
                return True
            except:
                return False
                
    def get_queue_stats(self):
        """Get comprehensive queue and system statistics for monitoring"""
        stats = {
            'queue_size': self.send_queue.qsize(),
            'max_queue_size': self.send_queue.maxsize,
            'queue_utilization': (self.send_queue.qsize() / self.send_queue.maxsize) * 100,
            'thread_alive': self.sender_thread.is_alive() if self.sender_thread else False,
            'connected': self.connected,
            'use_threading': self.use_threading,
            'cpu_count': self.cpu_count,
            'total_frames': self.frame_stats['total_frames'],
            'dropped_frames': self.frame_stats['dropped_frames'],
            'queue_overflows': self.frame_stats['queue_overflows'],
            'drop_rate': (self.frame_stats['dropped_frames'] / max(self.frame_stats['total_frames'], 1)) * 100
        }
        
        # Add system monitoring if psutil is available
        if psutil:
            try:
                stats.update({
                    'cpu_usage': psutil.cpu_percent(interval=0.1),
                    'memory_usage': psutil.virtual_memory().percent,
                    'available_memory_mb': psutil.virtual_memory().available // (1024 * 1024)
                })
            except Exception as e:
                logger.debug(f"System monitoring error: {e}")
        
        return stats
            
    def detect_humans(self, frame):
        """Detect humans in frame with frame skipping & caching optimization for RPi 4"""
        try:
            self.frame_counter += 1
            current_time = time.time()
            
            # Frame skipping optimization - process every 3rd frame on RPi 4
            if (self.frame_counter % self.inference_interval != 0 and 
                current_time - self.detection_cache_time < self.cache_duration):
                # Return cached detections
                logger.debug(f"Using cached detections (frame {self.frame_counter})")
                return self.last_detections
            
            # Run YOLOv8 inference on selected frames
            logger.debug(f"Running inference on frame {self.frame_counter}")
            results = self.model(frame, verbose=False)
            
            human_detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Check if detection is a person (class 0 in COCO dataset)
                        if int(box.cls[0]) == 0:  # person class
                            confidence = float(box.conf[0])
                            
                            # Keep original confidence threshold (0.5)
                            if confidence >= 0.5:
                                # Get bounding box coordinates
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                
                                detection = {
                                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                    "confidence": confidence
                                }
                                human_detections.append(detection)
            
            # Update cache
            self.last_detections = human_detections
            self.detection_cache_time = current_time
            
            return human_detections
            
        except Exception as e:
            logger.error(f"Error in human detection: {e}")
            return self.last_detections  # Return cached detections on error
            
    def send_frame_data(self, detections, timestamp):
        """Send individual frame detection data to server
        
        IMPORTANT: This data is used for LSTM processing and MUST NOT be modified
        by any optimization. Only video frames can be optimized for transmission.
        """
        if not self.connected or not detections:
            return
            
        # LSTM data - DO NOT MODIFY - Critical for fall detection accuracy
        for detection in detections:
            data = {
                "frame_id": self.frame_id,
                "bbox": detection["bbox"],
                "confidence": detection["confidence"],
                "timestamp": timestamp
            }
            
            # Always use reliable WebSocket for LSTM data (never UDP)
            success = self.queue_data('bbox_data', data)
            if success:
                logger.debug(f"Queued frame data: {data}")
            else:
                logger.warning(f"Failed to queue frame data for frame {self.frame_id}")
                
    def send_sequence_data(self):
        """Send buffered sequence data to server for LSTM processing
        
        IMPORTANT: This data is used for LSTM processing and MUST NOT be modified
        by any optimization. Only video frames can be optimized for transmission.
        """
        if not self.connected or len(self.bbox_buffer) < self.sequence_length:
            return
            
        # LSTM sequence data - DO NOT MODIFY - Critical for fall detection accuracy
        data = {
            "sequence": list(self.bbox_buffer),
            "timestamps": list(self.timestamp_buffer)
        }
        
        # Always use reliable WebSocket for LSTM data (never UDP)
        success = self.queue_data('bbox_data', data)
        if success:
            logger.debug(f"Queued sequence data with {len(self.bbox_buffer)} frames")
        else:
            logger.warning(f"Failed to queue sequence data")
            
        # Clear buffers after sending
        self.bbox_buffer.clear()
        self.timestamp_buffer.clear()
            
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
                self.frame_stats['total_frames'] += 1
                timestamp = datetime.now().isoformat()
                
                # Periodic health check and system monitoring
                current_time = time.time()
                if current_time - self.frame_stats['last_health_check'] > 30:  # Every 30 seconds
                    self._perform_health_check()
                    self.frame_stats['last_health_check'] = current_time
                
                # Detect humans in frame
                detections = self.detect_humans(frame)
                
                # Draw bounding boxes on frame for visualization
                for detection in detections:
                    bbox = detection["bbox"]
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # Adaptive compression based on network conditions
                jpeg_quality, resize_factor = self._adaptive_compression_params()
                
                # Conditional frame encoding optimization for RPi 4
                should_encode = False
                
                if detections:
                    # Always encode when there are detections
                    should_encode = True
                    self.last_frame_encoded = self.frame_id
                elif self.frame_id - self.last_frame_encoded >= self.encoding_interval:
                    # Encode every 5th frame when no detections (maintain connection)
                    should_encode = True
                    self.last_frame_encoded = self.frame_id
                
                if should_encode:
                    # Adaptive frame resizing based on network conditions
                    new_width = int(640 * resize_factor)
                    new_height = int(480 * resize_factor)
                    transmission_frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Smart caching - check if similar frame already encoded
                    frame_hash = hashlib.md5(transmission_frame.tobytes()).hexdigest()
                    cached_frame = self._get_cached_frame_hash(frame_hash)
                    
                    if cached_frame:
                        frame_base64 = cached_frame
                        self.cache_hits += 1
                        logger.debug(f"Cache hit for frame {self.frame_id}")
                    else:
                        self.cache_misses += 1
                        frame_base64 = None  # Initialize variable
                        
                        # Try hardware encoding first
                        if self.hardware_encoder_available and jpeg_quality > 50:
                            frame_base64 = self._hardware_encode_frame(transmission_frame, jpeg_quality)
                        
                        # Fallback to software encoding
                        if not frame_base64:
                            _, buffer = cv2.imencode('.jpg', transmission_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                            frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        # Cache the encoded frame
                        self._cache_encoded_frame(frame_hash, frame_base64)
                    
                    # Choose transmission method based on network conditions
                    frame_payload = {
                        'frame': frame_base64,
                        'timestamp': timestamp,
                        'frame_id': self.frame_id,
                        'quality': jpeg_quality,
                        'resolution': f'{new_width}x{new_height}'
                    }
                    
                    # Use UDP for video if latency is high and UDP is available
                    if self.network_latency > 200 and self.use_udp_video and self.udp_socket:
                        success = self._send_udp_frame(frame_base64)
                        if not success:
                            # Fallback to WebSocket
                            success = self.queue_data('video_frame', frame_payload)
                    else:
                        # Use WebSocket for reliable delivery
                        success = self.queue_data('video_frame', frame_payload)
                    
                    if not success:
                        logger.debug("Failed to send video frame data")
                    else:
                        logger.debug(f"Frame sent (detections: {len(detections)}, quality: {jpeg_quality}, method: {'UDP' if self.use_udp_video and self.network_latency > 200 else 'WebSocket'})")
                else:
                    logger.debug(f"Frame encoding skipped (no detections, last encoded: {self.last_frame_encoded})")
                
                # Send enhanced heartbeat with optimization metrics every 30 frames (~1 second)
                if self.frame_id % 30 == 0 and self.connected:
                    total_cache_requests = self.cache_hits + self.cache_misses
                    cache_hit_rate = (self.cache_hits / max(total_cache_requests, 1)) * 100
                    
                    heartbeat_data = {
                        'timestamp': timestamp,
                        'frame_id': self.frame_id,
                        'status': 'active',
                        'detections_count': len(detections),
                        'queue_stats': self.get_queue_stats(),
                        'optimization_stats': {
                            'network_latency': self.network_latency,
                            'cache_hit_rate': cache_hit_rate,
                            'hardware_encoder': self.hardware_encoder_available,
                            'udp_enabled': self.use_udp_video,
                            'adaptive_quality': jpeg_quality,
                            'adaptive_resolution': f'{new_width}x{new_height}' if should_encode else 'N/A'
                        }
                    }
                    # Always use WebSocket for heartbeat (reliable delivery)
                    success = self.queue_data('heartbeat', heartbeat_data)
                    if success:
                        logger.debug(f"Queued enhanced heartbeat at frame {self.frame_id}")
                    else:
                        logger.warning(f"Failed to queue heartbeat at frame {self.frame_id}")
                
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
                        empty_data = {
                            "frame_id": self.frame_id,
                            "bbox": [0, 0, 0, 0],
                            "confidence": 0.0,
                            "timestamp": timestamp,
                            "no_detection": True
                        }
                        success = self.queue_data('bbox_data', empty_data)
                        if success:
                            logger.debug(f"Queued empty bbox data at frame {self.frame_id}")
                        else:
                            logger.warning(f"Failed to queue empty bbox data at frame {self.frame_id}")
                        
                # Optional: Display frame with detections (for debugging)
                # Uncomment the following lines if you want to see the video feed
                # for detection in detections:
                #     bbox = detection["bbox"]
                #     cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                # cv2.imshow('Pi Client', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                    
                # Adaptive FPS control for RPi 4 based on CPU usage
                if psutil:
                    try:
                        cpu_usage = psutil.cpu_percent(interval=0)
                        if cpu_usage > 85:  # High CPU usage on RPi 4
                            time.sleep(0.1)   # 10 FPS
                            logger.debug(f"High CPU ({cpu_usage}%), reduced to 10 FPS")
                        elif cpu_usage > 70:  # Medium CPU usage
                            time.sleep(0.066) # 15 FPS
                            logger.debug(f"Medium CPU ({cpu_usage}%), reduced to 15 FPS")
                        elif cpu_usage > 50:  # Normal CPU usage
                            time.sleep(0.05)  # 20 FPS
                        else:  # Low CPU usage
                            time.sleep(0.033) # 30 FPS
                    except Exception as e:
                        logger.debug(f"CPU monitoring error: {e}")
                        time.sleep(0.05)  # Default 20 FPS for RPi 4
                else:
                    # Default adaptive FPS for RPi 4 without psutil
                    if detections:
                        time.sleep(0.05)  # 20 FPS when detections found
                    else:
                        time.sleep(0.066) # 15 FPS when no detections
                
        except KeyboardInterrupt:
            logger.info("Stopping Pi Client...")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources with proper thread termination"""
        logger.info("Starting cleanup process...")
        
        # Stop the sender thread
        self.running = False
        
        # Wait for sender thread to finish
        if self.sender_thread and self.sender_thread.is_alive():
            logger.info("Waiting for sender thread to finish...")
            self.sender_thread.join(timeout=3)
            if self.sender_thread.is_alive():
                logger.warning("Sender thread did not terminate gracefully")
        
        # Clear the queue
        while not self.send_queue.empty():
            try:
                self.send_queue.get_nowait()
            except:
                break
        
        # Close UDP socket
        if self.udp_socket:
            try:
                self.udp_socket.close()
                logger.info("UDP socket closed")
            except Exception as e:
                logger.error(f"Error closing UDP socket: {e}")
        
        # Cleanup camera resources
        if hasattr(self, 'camera_type') and self.camera_type == "picamera":
            if hasattr(self, 'picam'):
                try:
                    self.picam.stop()
                    self.picam.close()
                    logger.info("PiCamera stopped successfully")
                except Exception as e:
                    logger.error(f"Error stopping PiCamera: {e}")
        elif hasattr(self, 'cap') and self.cap:
            self.cap.release()
            logger.info("USB Camera released successfully")
        
        cv2.destroyAllWindows()
        
        # Disconnect from server
        if self.connected:
            try:
                self.sio.disconnect()
                logger.info("Disconnected from server")
            except Exception as e:
                logger.error(f"Error disconnecting from server: {e}")
        
        # Log cache statistics
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests > 0:
            cache_hit_rate = (self.cache_hits / total_cache_requests) * 100
            logger.info(f"Cache statistics - Hit rate: {cache_hit_rate:.1f}% ({self.cache_hits}/{total_cache_requests})")
        
        logger.info("Cleanup completed successfully")
    
    def _get_available_memory(self):
        """Get available system memory in bytes"""
        if psutil:
            try:
                return psutil.virtual_memory().available
            except Exception as e:
                logger.debug(f"Memory detection error: {e}")
        # Fallback estimate for Raspberry Pi 4
        return 1024 * 1024 * 1024  # 1GB default
    
    def _detect_hardware_encoder(self):
        """Detect if hardware encoder is available on Raspberry Pi"""
        try:
            # Check for Raspberry Pi hardware encoder
            result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
            if result.returncode == 0:
                # Check for h264_omx encoder
                result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True)
                if 'h264_omx' in result.stdout or 'h264_v4l2m2m' in result.stdout:
                    logger.info("Hardware encoder detected: h264_omx/h264_v4l2m2m")
                    return True
        except Exception as e:
            logger.debug(f"Hardware encoder detection failed: {e}")
        return False
    
    def _setup_udp_socket(self):
        """Setup UDP socket for video streaming"""
        try:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
            server_ip = self.server_url.split('://')[1].split(':')[0]
            self.udp_target = (server_ip, self.udp_port)
            logger.info(f"UDP socket setup for video streaming to {self.udp_target}")
            return True
        except Exception as e:
            logger.error(f"Failed to setup UDP socket: {e}")
            return False
    
    def _measure_network_latency(self):
        """Measure network latency to server"""
        try:
            start_time = time.time()
            # Simple ping via socket connection
            server_ip = self.server_url.split('://')[1].split(':')[0]
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((server_ip, 5001))
            sock.close()
            
            if result == 0:
                self.network_latency = (time.time() - start_time) * 1000  # ms
                logger.debug(f"Network latency: {self.network_latency:.2f}ms")
            else:
                self.network_latency = 1000  # High latency fallback
        except Exception as e:
            logger.debug(f"Latency measurement failed: {e}")
            self.network_latency = 500  # Default fallback
    
    def _adaptive_compression_params(self):
        """Calculate adaptive compression parameters based on network conditions"""
        # Measure latency periodically
        current_time = time.time()
        if current_time - self.last_ping_time > self.ping_interval:
            self._measure_network_latency()
            self.last_ping_time = current_time
        
        # Adjust parameters based on latency and queue utilization
        queue_utilization = (self.send_queue.qsize() / self.send_queue.maxsize) * 100
        
        # Adaptive JPEG quality
        if self.network_latency > 300 or queue_utilization > 80:
            jpeg_quality = self.min_jpeg_quality
            resize_factor = 0.3
        elif self.network_latency > 150 or queue_utilization > 60:
            jpeg_quality = self.base_jpeg_quality - 20
            resize_factor = 0.4
        elif self.network_latency > 50 or queue_utilization > 40:
            jpeg_quality = self.base_jpeg_quality - 10
            resize_factor = 0.5
        else:
            jpeg_quality = self.base_jpeg_quality
            resize_factor = 0.6
        
        return max(jpeg_quality, self.min_jpeg_quality), resize_factor
    
    @lru_cache(maxsize=50)
    def _get_cached_frame_hash(self, frame_data_hash):
        """Get cached encoded frame by hash"""
        return self.frame_cache.get(frame_data_hash)
    
    def _cache_encoded_frame(self, frame_hash, encoded_data):
        """Cache encoded frame data"""
        if len(self.frame_cache) > 100:  # Limit cache size
            # Remove oldest entries
            oldest_keys = list(self.frame_cache.keys())[:20]
            for key in oldest_keys:
                del self.frame_cache[key]
        
        self.frame_cache[frame_hash] = encoded_data
    
    def _hardware_encode_frame(self, frame, quality=65):
        """Encode frame using hardware encoder if available"""
        if not self.hardware_encoder_available:
            return None
        
        try:
            height, width = frame.shape[:2]
            # Use ffmpeg with hardware encoder
            cmd = [
                'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                '-s', f'{width}x{height}', '-r', '30', '-i', '-',
                '-c:v', 'h264_omx', '-b:v', f'{quality*10}k',
                '-preset', 'ultrafast', '-f', 'mp4', '-movflags', 'frag_keyframe+empty_moov', '-'
            ]
            
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(input=frame.tobytes())
            
            if process.returncode == 0 and stdout:
                return base64.b64encode(stdout).decode('utf-8')
        except Exception as e:
            logger.debug(f"Hardware encoding failed: {e}")
        
        return None
    
    def _send_udp_frame(self, frame_data):
        """Send frame via UDP for low-latency video streaming"""
        if not self.udp_socket:
            return False
        
        try:
            # Split large frames into chunks
            chunk_size = 60000  # 60KB chunks
            frame_bytes = frame_data.encode('utf-8')
            total_chunks = (len(frame_bytes) + chunk_size - 1) // chunk_size
            
            for i in range(total_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, len(frame_bytes))
                chunk = frame_bytes[start:end]
                
                # Add header: frame_id(4) + chunk_id(4) + total_chunks(4) + data
                header = struct.pack('!III', self.frame_id, i, total_chunks)
                packet = header + chunk
                
                self.udp_socket.sendto(packet, self.udp_target)
            
            return True
        except Exception as e:
            logger.debug(f"UDP send failed: {e}")
            return False
    
    def _attempt_reconnect(self):
        """Attempt to reconnect to server with exponential backoff"""
        if not self.running:
            return
            
        try:
            logger.info(f"Reconnecting to server (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})...")
            self.sio.connect(self.server_url)
            logger.info("Reconnection successful")
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            if self.reconnect_attempts < self.max_reconnect_attempts:
                # Schedule next attempt
                delay = min(self.reconnect_delay * (2 ** self.reconnect_attempts), 60)
                threading.Timer(delay, self._attempt_reconnect).start()
            else:
                logger.error("Max reconnection attempts reached. Manual restart required.")
    
    def _reduce_frame_quality(self, payload):
        """Reduce frame quality for congestion control"""
        if 'frame' in payload:
            try:
                # Decode base64 frame
                frame_data = base64.b64decode(payload['frame'])
                frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                
                # Re-encode with lower quality
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                payload['frame'] = base64.b64encode(buffer).decode('utf-8')
                
                logger.debug("Reduced frame quality due to congestion")
            except Exception as e:
                logger.debug(f"Frame quality reduction failed: {e}")
        
        return payload
    
    def _perform_health_check(self):
        """Perform comprehensive system health check"""
        stats = self.get_queue_stats()
        
        # Cache statistics
        total_cache_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / max(total_cache_requests, 1)) * 100
        
        # Log health status with enhanced metrics
        logger.info(f"Health Check - Frames: {stats['total_frames']}, "
                   f"Dropped: {stats['dropped_frames']} ({stats['drop_rate']:.1f}%), "
                   f"Queue: {stats['queue_size']}/{stats['max_queue_size']} ({stats['queue_utilization']:.1f}%), "
                   f"Cache Hit Rate: {cache_hit_rate:.1f}%, "
                   f"Network Latency: {self.network_latency:.1f}ms")
        
        if psutil:
            logger.info(f"System - CPU: {stats.get('cpu_usage', 'N/A')}%, "
                       f"Memory: {stats.get('memory_usage', 'N/A')}%, "
                       f"Available: {stats.get('available_memory_mb', 'N/A')}MB, "
                       f"HW Encoder: {'Yes' if self.hardware_encoder_available else 'No'}")
        
        # Performance warnings
        if stats['drop_rate'] > 10:
            logger.warning(f"High drop rate detected: {stats['drop_rate']:.1f}%")
        
        if stats['queue_utilization'] > 90:
            logger.warning(f"Queue nearly full: {stats['queue_utilization']:.1f}%")
        
        if psutil and stats.get('memory_usage', 0) > 85:
            logger.warning(f"High memory usage: {stats['memory_usage']}%")
        
        if self.network_latency > 300:
            logger.warning(f"High network latency: {self.network_latency:.1f}ms - Using aggressive compression")
        
        # Auto-optimization suggestions
        if stats['queue_overflows'] > 50:
            logger.info("Consider reducing frame rate or increasing queue size")
        
        if cache_hit_rate < 20 and total_cache_requests > 50:
            logger.info("Low cache hit rate - Frames are highly variable")
        
        if not stats['thread_alive'] and self.use_threading:
            logger.error("Sender thread died! Attempting restart...")
            self.start_sender_thread()

def main():
    # Configuration from environment variables
    SERVER_URL = os.getenv('SERVER_URL', 'http://10.243.149.68:5001/')
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