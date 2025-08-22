#!/usr/bin/env python3
"""
Laptop Server - Flask-SocketIO + LSTM Classification + Real-time Dashboard
Receives bounding box data from Raspberry Pi and performs fall detection
"""

import os
import json
import csv
from datetime import datetime
from collections import deque
import logging
import threading
import time

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
import pandas as pd
import base64
from dotenv import load_dotenv

# Load environment variables from local .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Import LSTM model handler
from lstm_model import LSTMModel
from firebase_config import FirebaseHandler

# Import caching and rate limiting
try:
    from cache_config import initialize_cache_and_rate_limiting, get_cache_manager, get_rate_limit_manager
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    logging.warning("Optimization features not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FallDetectionServer:
    def __init__(self, sequence_length=61, confidence_threshold=0.7):
        """
        Initialize Fall Detection Server
        
        Args:
            sequence_length: Number of frames in a sequence for LSTM
            confidence_threshold: Minimum confidence for fall detection
        """
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'fall_detection_secret_key')
        allowed_origins = os.getenv('ALLOWED_ORIGINS', '*')
        self.socketio = SocketIO(self.app, cors_allowed_origins=allowed_origins)
        
        # Initialize caching and rate limiting
        if OPTIMIZATION_AVAILABLE:
            try:
                redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
                self.cache_manager, self.rate_limit_manager = initialize_cache_and_rate_limiting(self.app, redis_url)
                logger.info("Caching and rate limiting initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize optimization features: {e}")
                self.cache_manager = None
                self.rate_limit_manager = None
        else:
            self.cache_manager = None
            self.rate_limit_manager = None
        
        # Data buffers
        self.bbox_buffer = deque(maxlen=sequence_length)
        self.timestamp_buffer = deque(maxlen=sequence_length)
        
        # LSTM model
        self.lstm_model = LSTMModel()
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'fall_events': 0,
            'last_detection': None,
            'server_start_time': datetime.now().isoformat()
        }
        
        # Recent events for dashboard
        self.recent_events = deque(maxlen=100)
        
        # Generate session ID
        import uuid
        self.session_id = str(uuid.uuid4())[:8]
        
        # Initialize Firebase handler
        try:
            service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH', 'fall-detection-system-3defc-firebase-adminsdk-fbsvc-a29f222b61.json')
            project_id = os.getenv('FIREBASE_PROJECT_ID', 'fall-detection-system-3defc')
            self.firebase_handler = FirebaseHandler(
                service_account_path=service_account_path,
                project_id=project_id
            )
            logger.info(f"Firebase handler initialized successfully (Session: {self.session_id})")
        except Exception as e:
            logger.error(f"Failed to initialize Firebase handler: {e}")
            self.firebase_handler = None

        self.system_start_time = datetime.now()
        
        # Setup routes and socket events
        self.setup_routes()
        self.setup_socket_events()
        
        # Initialize log file
        self.init_log_file()
    
    def _process_timeline_data(self, logs):
        """Process logs for timeline chart (last 7 days)"""
        from collections import defaultdict
        from datetime import datetime, timedelta
        
        # Get last 7 days
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=6)
        
        # Initialize data structure
        timeline_data = []
        daily_counts = defaultdict(lambda: {'falls': 0, 'normal': 0})
        
        # Process logs
        for log in logs:
            try:
                log_date = datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00')).date()
                if start_date <= log_date <= end_date:
                    is_fall = log.get('type') == 'fall' or log.get('is_fall') == True
                    if is_fall:
                        daily_counts[log_date]['falls'] += 1
                    else:
                        daily_counts[log_date]['normal'] += 1
            except:
                continue
        
        # Create timeline data for last 7 days
        current_date = start_date
        while current_date <= end_date:
            timeline_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'day': current_date.strftime('%a'),
                'falls': daily_counts[current_date]['falls'],
                'normal': daily_counts[current_date]['normal']
            })
            current_date += timedelta(days=1)
        
        return timeline_data
    
    def _process_hourly_pattern(self, logs):
        """Process logs for hourly pattern analysis"""
        from collections import defaultdict
        
        hourly_counts = defaultdict(lambda: {'falls': 0, 'normal': 0})
        
        for log in logs:
            try:
                log_datetime = datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00'))
                hour = log_datetime.hour
                is_fall = log.get('type') == 'fall' or log.get('is_fall') == True
                
                if is_fall:
                    hourly_counts[hour]['falls'] += 1
                else:
                    hourly_counts[hour]['normal'] += 1
            except:
                continue
        
        # Create hourly pattern data (0-23 hours)
        hourly_data = []
        for hour in range(24):
            hourly_data.append({
                'hour': hour,
                'time': f'{hour:02d}:00',
                'falls': hourly_counts[hour]['falls'],
                'normal': hourly_counts[hour]['normal']
            })
        
        return hourly_data
    
    def _process_weekly_trend(self, logs):
        """Process logs for weekly trend analysis (last 4 weeks)"""
        from collections import defaultdict
        from datetime import datetime, timedelta
        import calendar
        
        # Get last 4 weeks
        end_date = datetime.now().date()
        start_date = end_date - timedelta(weeks=4)
        
        weekly_counts = defaultdict(lambda: {'falls': 0, 'normal': 0})
        
        for log in logs:
            try:
                log_date = datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00')).date()
                if start_date <= log_date <= end_date:
                    # Get week number
                    year, week, _ = log_date.isocalendar()
                    week_key = f'{year}-W{week:02d}'
                    
                    is_fall = log.get('type') == 'fall' or log.get('is_fall') == True
                    if is_fall:
                        weekly_counts[week_key]['falls'] += 1
                    else:
                        weekly_counts[week_key]['normal'] += 1
            except:
                continue
        
        # Create weekly trend data
        weekly_data = []
        current_date = start_date
        while current_date <= end_date:
            year, week, _ = current_date.isocalendar()
            week_key = f'{year}-W{week:02d}'
            
            # Get week start date for display
            week_start = current_date - timedelta(days=current_date.weekday())
            
            if week_key not in [w['week'] for w in weekly_data]:  # Avoid duplicates
                weekly_data.append({
                    'week': week_key,
                    'week_start': week_start.strftime('%m/%d'),
                    'falls': weekly_counts[week_key]['falls'],
                    'normal': weekly_counts[week_key]['normal']
                })
            
            current_date += timedelta(days=7)
        
        return weekly_data[-4:]  # Return last 4 weeks only
        
    def init_log_file(self):
        """Initialize CSV log file"""
        log_dir = os.path.dirname(os.getenv('LOG_FILE', 'logs/fall_detection.log'))
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = 'logs.csv'
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'label', 'confidence', 'bbox_data'])
                
    def setup_routes(self):
        """Setup Flask routes"""
        @self.app.route('/')
        def dashboard():
            # Get Firebase config from environment variables
            firebase_config = {
                'apiKey': os.getenv('FIREBASE_API_KEY', ''),
                'authDomain': os.getenv('FIREBASE_AUTH_DOMAIN', ''),
                'projectId': os.getenv('FIREBASE_PROJECT_ID', ''),
                'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET', ''),
                'messagingSenderId': os.getenv('FIREBASE_MESSAGING_SENDER_ID', ''),
                'appId': os.getenv('FIREBASE_APP_ID', '')
            }
            return render_template('dashboard.html', firebase_config=firebase_config)
        
        @self.app.route('/system-check')
        def system_check():
            return render_template('system_check.html')
        
        @self.app.route('/logs')
        def logs():
            return render_template('logs.html')
        

            
        @self.app.route('/api/firebase-test')
        def test_firebase():
            # Apply rate limiting if available
            if self.rate_limit_manager and self.rate_limit_manager.initialized:
                @self.rate_limit_manager.limit("10 per minute")
                def _test_firebase():
                    return _execute_firebase_test()
                return _test_firebase()
            else:
                return _execute_firebase_test()
        
        def _execute_firebase_test():
            if self.firebase_handler.initialized:
                try:
                    # Use optimized method if available
                    if hasattr(self.firebase_handler, 'get_recent_detections_optimized'):
                        recent_detections = self.firebase_handler.get_recent_detections_optimized(limit=1, hours=1)
                    else:
                        recent_detections = self.firebase_handler.get_recent_detections(1)
                    
                    return jsonify({
                        'status': 'success',
                        'message': 'Firebase connection successful',
                        'detections_count': len(recent_detections),
                        'cache_enabled': self.cache_manager.initialized if self.cache_manager else False
                    })
                except Exception as e:
                    return jsonify({
                        'status': 'error',
                        'message': f'Firebase connection failed: {str(e)}'
                    }), 500
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Firebase not initialized'
                }), 500
        
        @self.app.route('/api/stats')
        def get_stats():
            # Calculate today's falls
            today = datetime.now().date()
            today_falls = 0
            last_fall_time = None
            
            # Count today's falls from recent events
            for event in self.recent_events:
                try:
                    event_date = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00')).date()
                    if event_date == today and event['label'] == 'Jatuh':
                        today_falls += 1
                        if last_fall_time is None or event['timestamp'] > last_fall_time:
                            last_fall_time = event['timestamp']
                except:
                    continue
            
            return jsonify({
                'totalFalls': self.stats['fall_events'],
                'todayFalls': today_falls,
                'lastFallTime': last_fall_time,
                'uptime': str(datetime.now() - self.system_start_time)
            })
             
        @self.app.route('/api/recent_events')
        def get_recent_events():
            return jsonify(list(self.recent_events))
             
        @self.app.route('/api/system-health')
        def get_system_health():
            # Check if we've received data recently (within last 10 seconds)
            camera_online = len(self.bbox_buffer) > 0 and self.stats.get('last_detection')
            if camera_online and self.stats.get('last_detection'):
                try:
                    last_detection_time = datetime.fromisoformat(self.stats['last_detection'].replace('Z', '+00:00'))
                    camera_online = (datetime.now() - last_detection_time).total_seconds() < 10
                except:
                    camera_online = False
            
            return jsonify({
                'cameraStatus': camera_online,
                'lstmModel': self.lstm_model is not None,
                'database': self.firebase_handler.initialized if self.firebase_handler else False,
                'piClient': camera_online  # Assume Pi client is online if camera is sending data
            })
        
        @self.app.route('/api/firebase-logs')
        def get_firebase_logs():
            # Apply rate limiting if available
            if self.rate_limit_manager and self.rate_limit_manager.initialized:
                @self.rate_limit_manager.limit("30 per minute")
                def _get_firebase_logs():
                    return _execute_get_firebase_logs()
                return _get_firebase_logs()
            else:
                return _execute_get_firebase_logs()
        
        def _execute_get_firebase_logs():
            try:
                # Use optimized method if available
                if hasattr(self.firebase_handler, 'get_all_logs_optimized'):
                    logs = self.firebase_handler.get_all_logs_optimized(limit=500, days=30)
                else:
                    logs = self.firebase_handler.get_all_logs(limit=500)
                
                return jsonify({
                    'logs': logs,
                    'count': len(logs),
                    'optimized': hasattr(self.firebase_handler, 'get_all_logs_optimized'),
                    'cached': self.cache_manager.initialized if self.cache_manager else False
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/timeline-stats')
        def get_timeline_stats():
            """Get timeline statistics for charts"""
            # Apply rate limiting if available
            if self.rate_limit_manager and self.rate_limit_manager.initialized:
                @self.rate_limit_manager.limit("20 per minute")
                def _get_timeline_stats():
                    return _execute_get_timeline_stats()
                return _get_timeline_stats()
            else:
                return _execute_get_timeline_stats()
        
        def _execute_get_timeline_stats():
            try:
                # Use optimized method if available
                if hasattr(self.firebase_handler, 'get_all_logs_optimized'):
                    logs = self.firebase_handler.get_all_logs_optimized(limit=1000, days=30)
                else:
                    logs = self.firebase_handler.get_all_logs(limit=1000)
                
                # Process data for timeline chart
                timeline_data = self._process_timeline_data(logs)
                hourly_pattern = self._process_hourly_pattern(logs)
                weekly_trend = self._process_weekly_trend(logs)
                
                return jsonify({
                    'timeline': timeline_data,
                    'hourly_pattern': hourly_pattern,
                    'weekly_trend': weekly_trend,
                    'total_logs': len(logs),
                    'fall_count': len([log for log in logs if log.get('type') == 'fall' or log.get('is_fall') == True]),
                    'optimized': hasattr(self.firebase_handler, 'get_all_logs_optimized'),
                    'cached': self.cache_manager.initialized if self.cache_manager else False
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/optimization-status')
        def get_optimization_status():
            """Get status of optimization features (caching and rate limiting)"""
            # Apply basic rate limiting
            if self.rate_limit_manager and self.rate_limit_manager.initialized:
                @self.rate_limit_manager.limit("60 per minute")
                def _get_optimization_status():
                    return _execute_get_optimization_status()
                return _get_optimization_status()
            else:
                return _execute_get_optimization_status()
        
        def _execute_get_optimization_status():
            try:
                cache_stats = {
                    'enabled': self.cache_manager.initialized if self.cache_manager else False,
                    'type': 'Redis' if (self.cache_manager and self.cache_manager.redis_client) else 'SimpleCache' if self.cache_manager else 'None'
                }
                
                rate_limit_stats = {
                    'enabled': self.rate_limit_manager.initialized if self.rate_limit_manager else False,
                    'storage': 'Redis' if os.getenv('REDIS_URL') else 'Memory'
                }
                
                return jsonify({
                    'optimization_available': OPTIMIZATION_AVAILABLE,
                    'cache': cache_stats,
                    'rate_limiting': rate_limit_stats,
                    'firebase_optimized': hasattr(self.firebase_handler, 'get_all_logs_optimized'),
                    'redis_url': os.getenv('REDIS_URL', 'Not configured')
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/cache-clear')
        def clear_cache():
            """Clear application cache (admin function)"""
            # Apply strict rate limiting for admin functions
            if self.rate_limit_manager and self.rate_limit_manager.initialized:
                @self.rate_limit_manager.limit("5 per minute")
                def _clear_cache():
                    return _execute_clear_cache()
                return _clear_cache()
            else:
                return _execute_clear_cache()
        
        def _execute_clear_cache():
            try:
                if self.cache_manager and self.cache_manager.initialized:
                    success = self.cache_manager.clear()
                    return jsonify({
                        'status': 'success' if success else 'failed',
                        'message': 'Cache cleared successfully' if success else 'Failed to clear cache'
                    })
                else:
                    return jsonify({
                        'status': 'info',
                        'message': 'Cache not available or not initialized'
                    })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
            
    def setup_socket_events(self):
        """Setup SocketIO events"""
        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"Client connected: {request.sid if 'request' in globals() else 'unknown'}")
            emit('status', {'message': 'Connected to Fall Detection Server'})
            
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info(f"Client disconnected: {request.sid if 'request' in globals() else 'unknown'}")
            
        @self.socketio.on('bbox_data')
        def handle_bbox_data(data):
            """Handle incoming bounding box data from Raspberry Pi"""
            try:
                self.process_bbox_data(data)
            except Exception as e:
                logger.error(f"Error processing bbox data: {e}")
                
        @self.socketio.on('heartbeat')
        def handle_heartbeat(data):
            """Handle heartbeat from Pi Client to maintain connection status"""
            try:
                # Update last detection time to keep Pi Client status as online
                self.stats['last_detection'] = data.get('timestamp', datetime.now().isoformat())
                logger.debug(f"Received heartbeat from Pi Client at frame {data.get('frame_id', 'unknown')}")
            except Exception as e:
                logger.error(f"Error processing heartbeat: {e}")
                
        @self.socketio.on('video_frame')
        def handle_video_frame(data):
            try:
                # Broadcast video frame to all connected clients
                emit('video_frame', {
                    'frame': data['frame'],
                    'timestamp': datetime.now().isoformat()
                }, broadcast=True)
                
            except Exception as e:
                logger.error(f"Error handling video frame: {e}")
                
    def process_bbox_data(self, data):
        """Process incoming bounding box data"""
        self.stats['total_frames'] += 1
        
        if 'sequence' in data:
            # Handle sequence data
            self.process_sequence_data(data)
        else:
            # Handle per-frame data
            self.process_frame_data(data)
            
    def process_frame_data(self, data):
        """Process per-frame bounding box data"""
        bbox = data.get('bbox')
        confidence = data.get('confidence', 0)
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        if bbox and confidence >= 0.5:
            self.stats['total_detections'] += 1
            self.stats['last_detection'] = timestamp
            
            # Add to buffer
            self.bbox_buffer.append(bbox)
            self.timestamp_buffer.append(timestamp)
            
            # Process when buffer is full
            if len(self.bbox_buffer) == self.sequence_length:
                self.classify_sequence()
                
    def process_sequence_data(self, data):
        """Process sequence of bounding boxes"""
        sequence = data.get('sequence', [])
        timestamps = data.get('timestamps', [])
        
        if len(sequence) >= self.sequence_length:
            self.stats['total_detections'] += len(sequence)
            self.stats['last_detection'] = timestamps[-1] if timestamps else datetime.now().isoformat()
            
            # Use the sequence directly for classification
            self.classify_sequence(sequence, timestamps)
            
    def classify_sequence(self, sequence=None, timestamps=None):
        """Classify sequence using LSTM model"""
        try:
            # Use provided sequence or buffer
            if sequence is None:
                sequence = list(self.bbox_buffer)
                timestamps = list(self.timestamp_buffer)
                
            if len(sequence) < self.sequence_length:
                return
                
            # Prepare data for LSTM
            sequence_array = np.array(sequence[-self.sequence_length:])
            
            # Get prediction from LSTM model
            prediction = self.lstm_model.predict(sequence_array)
            
            label = prediction['label']
            confidence = prediction['confidence']
            
            # Create event data
            event_data = {
                'timestamp': timestamps[-1] if timestamps else datetime.now().isoformat(),
                'label': label,
                'confidence': confidence,
                'bbox_sequence': sequence[-self.sequence_length:]
            }
            
            # Log all events (both fall and normal activity)
            self.log_event(event_data)
            
            # Save to Firebase (save all events for comprehensive logging)
            self.save_to_firebase(event_data)
            
            # Add to recent events
            self.recent_events.append(event_data)
            
            # Broadcast to dashboard
            self.socketio.emit('detection_result', event_data)
            
            # Emit system status
            self.socketio.emit('system_status', {
                'camera_online': True,
                'model_loaded': True,
                'database_connected': self.firebase_handler.initialized if self.firebase_handler else False
            })
            
            # Handle fall detection
            if label == "Jatuh" and confidence >= self.confidence_threshold:
                self.handle_fall_detection(event_data)
                logger.info(f"Classification: {label} (confidence: {confidence:.2f}) - FALL ALERT!")
            elif label == "Jatuh":
                # Log fall detection even if below threshold
                logger.info(f"Classification: {label} (confidence: {confidence:.2f}) - Below threshold")
            else:
                # Log normal activity as well for comprehensive monitoring
                logger.info(f"Classification: {label} (confidence: {confidence:.2f})")
            
        except Exception as e:
            logger.error(f"Error in sequence classification: {e}")
            
    def handle_fall_detection(self, event_data):
        """Handle detected fall event"""
        self.stats['fall_events'] += 1
        
        # Create fall alert
        alert_data = {
            'type': 'fall_alert',
            'timestamp': event_data['timestamp'],
            'confidence': event_data['confidence'],
            'message': f"FALL DETECTED! Confidence: {event_data['confidence']:.2f}"
        }
        
        # Save to Firebase
        try:
            # Flatten bbox_sequence to avoid nested arrays
            bbox_sequence_flat = []
            if 'bbox_sequence' in event_data and event_data['bbox_sequence']:
                for bbox in event_data['bbox_sequence']:
                    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                        bbox_sequence_flat.append({
                            'x': float(bbox[0]),
                            'y': float(bbox[1]),
                            'width': float(bbox[2]),
                            'height': float(bbox[3])
                        })
            
            detection_data = {
                'confidence': float(event_data['confidence']),
                'timestamp': event_data['timestamp'],
                'bounding_boxes': bbox_sequence_flat,
                'label': 'Jatuh',
                'type': 'fall',
                'is_fall': True
            }
            self.firebase_handler.save_detection(detection_data)
            logger.info("Fall detection saved to Firebase")
        except Exception as firebase_error:
            logger.error(f"Failed to save to Firebase: {firebase_error}")
        
        # Broadcast fall alert to all connected clients
        self.socketio.emit('fall_alert', alert_data)
        
        logger.warning(f"FALL DETECTED! Confidence: {event_data['confidence']:.2f}")
        
        # Optional: Send notification (implement as needed)
        # self.send_notification(alert_data)
        
    def save_to_firebase(self, event_data):
        """Save detection event to Firebase (all events for comprehensive logging)"""
        try:
            if self.firebase_handler:
                # Save all events for comprehensive logging
                logger.debug(f"Saving event to Firebase: {event_data['label']} (confidence: {event_data['confidence']:.2f})")
                
                # Flatten bbox_sequence to avoid nested arrays
                bbox_sequence_flat = []
                if 'bbox_sequence' in event_data and event_data['bbox_sequence']:
                    for bbox in event_data['bbox_sequence']:
                        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                            bbox_sequence_flat.append({
                                'x': float(bbox[0]),
                                'y': float(bbox[1]),
                                'width': float(bbox[2]),
                                'height': float(bbox[3])
                            })
                
                # Add additional metadata
                is_fall = event_data['label'] == "Jatuh"
                firebase_data = {
                    'timestamp': event_data['timestamp'],
                    'label': event_data['label'],
                    'confidence': float(event_data['confidence']),
                    'type': 'fall' if is_fall else 'normal',  # Add type field for frontend compatibility
                    'is_fall': is_fall,
                    'bbox_sequence': bbox_sequence_flat,
                    'device_id': 'laptop_server',
                    'session_id': getattr(self, 'session_id', 'default_session')
                }
                
                self.firebase_handler.save_detection(firebase_data)
                logger.info(f"Saved detection to Firebase: {event_data['label']} ({event_data['confidence']:.2f})")
            else:
                logger.warning("Firebase handler not available")
        except Exception as e:
            logger.error(f"Error saving to Firebase: {e}")
            
    def log_event(self, event_data):
        """Log event to CSV file"""
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    event_data['timestamp'],
                    event_data['label'],
                    event_data['confidence'],
                    json.dumps(event_data['bbox_sequence'])
                ])
        except Exception as e:
            logger.error(f"Error logging event: {e}")
            
    def send_notification(self, alert_data):
        """Send notification (placeholder for future implementation)"""
        # TODO: Implement Telegram bot or email notification
        pass
        
    def run(self, host='0.0.0.0', port=5001, debug=False):
        """Run the server"""
        logger.info(f"Starting Fall Detection Server on {host}:{port}")
        logger.info(f"Sequence length: {self.sequence_length}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        
        # Start background tasks
        self.start_background_tasks()
        
        # Run the server
        self.socketio.run(self.app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
        
    def start_background_tasks(self):
        """Start background tasks"""
        # Start stats update thread
        stats_thread = threading.Thread(target=self.update_stats_periodically)
        stats_thread.daemon = True
        stats_thread.start()
        
    def update_stats_periodically(self):
        """Update statistics periodically"""
        while True:
            try:
                # Broadcast updated stats to dashboard
                self.socketio.emit('stats_update', self.stats)
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                logger.error(f"Error updating stats: {e}")
                time.sleep(5)

def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fall Detection Server')
    parser.add_argument('--port', type=int, default=5001, help='Port to run server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind server to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    # Configuration from environment variables with command line overrides
    SEQUENCE_LENGTH = 61  # Match LSTM training configuration
    CONFIDENCE_THRESHOLD = 0.7
    HOST = args.host if args.host != '0.0.0.0' else os.getenv('SERVER_HOST', '0.0.0.0')
    PORT = args.port if args.port != 5001 else int(os.getenv('SERVER_PORT', '5001'))
    DEBUG = args.debug or os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Create and run server
    server = FallDetectionServer(
        sequence_length=SEQUENCE_LENGTH,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    
    try:
        server.run(host=HOST, port=PORT, debug=DEBUG)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == '__main__':
    main()