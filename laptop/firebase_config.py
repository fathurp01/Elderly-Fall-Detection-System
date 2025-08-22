import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import caching utilities
try:
    from cache_config import cached_firebase_query, optimized_firebase_query, get_cache_manager
    CACHE_INTEGRATION = True
except ImportError:
    CACHE_INTEGRATION = False
    logging.warning("Cache integration not available")

try:
    import firebase_admin
    from firebase_admin import credentials, firestore, storage
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    logging.warning("Firebase Admin SDK not installed. Install with: pip install firebase-admin")

class FirebaseHandler:
    """Firebase integration for Fall Detection System"""
    
    def __init__(self, service_account_path: str = None, project_id: str = None):
        self.db = None
        self.bucket = None
        self.initialized = False
        
        if not FIREBASE_AVAILABLE:
            logging.error("Firebase Admin SDK not available")
            return
            
        try:
            # Check if service account file exists
            if service_account_path:
                if not os.path.exists(service_account_path):
                    logging.error(f"Service account file not found: {service_account_path}")
                    self.initialized = False
                    return
                else:
                    logging.info(f"Service account file found: {service_account_path}")
            
            # Initialize Firebase Admin SDK
            if not firebase_admin._apps:
                if service_account_path and os.path.exists(service_account_path):
                    logging.info("Initializing Firebase with service account")
                    cred = credentials.Certificate(service_account_path)
                    database_url = os.getenv('FIREBASE_DATABASE_URL')
                    config = {}
                    if database_url:
                        config['databaseURL'] = database_url
                    if project_id:
                        config['storageBucket'] = f'{project_id}.appspot.com'
                    firebase_admin.initialize_app(cred, config)
                else:
                    logging.info("Initializing Firebase with default credentials")
                    firebase_admin.initialize_app()
            else:
                logging.info("Firebase app already initialized")
            
            self.db = firestore.client()
            self.bucket = storage.bucket() if project_id else None
            self.initialized = True
            logging.info(f"Firebase initialized successfully with project: {project_id}")
            
        except Exception as e:
            logging.error(f"Failed to initialize Firebase: {e}")
            import traceback
            logging.error(f"Firebase initialization traceback: {traceback.format_exc()}")
            self.initialized = False
    
    def save_detection(self, detection_data: Dict) -> Optional[str]:
        """Save fall detection event to Firestore"""
        if not self.initialized:
            logging.warning("Firebase not initialized, skipping save")
            return None
            
        try:
            # Add timestamp if not present
            if 'timestamp' not in detection_data:
                detection_data['timestamp'] = datetime.now().isoformat()
            
            # Save to Firestore
            doc_ref = self.db.collection('fall_detections').add(detection_data)
            detection_id = doc_ref[1].id
            
            logging.info(f"Detection saved to Firebase with ID: {detection_id}")
            return detection_id
            
        except Exception as e:
            logging.error(f"Failed to save detection to Firebase: {e}")
            return None
    
    def get_recent_detections(self, limit: int = 10) -> List[Dict]:
        """Get recent fall detections from Firestore"""
        if not self.initialized:
            return []
            
        try:
            if not self.db:
                return []
            docs = (self.db.collection('fall_detections')
                   .order_by('timestamp', direction=firestore.Query.DESCENDING)
                   .limit(limit)
                   .stream())
            
            detections = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                detections.append(data)
            
            return detections
            
        except Exception as e:
            logging.error(f"Failed to get recent detections: {e}")
            return []
    
    @cached_firebase_query(get_cache_manager() if CACHE_INTEGRATION and get_cache_manager() else None, timeout=300, key_prefix="recent_detections")
    def get_recent_detections_optimized(self, limit: int = 10, hours: int = 24) -> List[Dict]:
        """Get recent fall detections with time filtering and caching"""
        if not self.initialized:
            return []
            
        try:
            if not self.db:
                return []
            
            # Calculate time threshold
            time_threshold = datetime.now() - timedelta(hours=hours)
            time_threshold_str = time_threshold.isoformat()
            
            # Optimized query with time filtering
            docs = (self.db.collection('fall_detections')
                   .where('timestamp', '>=', time_threshold_str)
                   .order_by('timestamp', direction=firestore.Query.DESCENDING)
                   .limit(limit)
                   .stream())
            
            detections = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                detections.append(data)
            
            logging.info(f"Retrieved {len(detections)} recent detections (last {hours}h, limit {limit})")
            return detections
            
        except Exception as e:
            logging.error(f"Failed to get recent detections optimized: {e}")
            return []
    
    def get_all_logs(self, limit: int = 1000) -> List[Dict]:
        """Get all detection logs for the logs page"""
        try:
            if not self.db:
                # Return mock data if Firebase is not configured
                return self._get_mock_logs()
            
            docs = self.db.collection('fall_detections').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit).stream()
            logs = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                logs.append(data)
            return logs
        except Exception as e:
            logging.error(f"Error getting logs: {e}")
            # Return mock data on error
            return self._get_mock_logs()
    
    @cached_firebase_query(get_cache_manager() if CACHE_INTEGRATION and get_cache_manager() else None, timeout=600, key_prefix="all_logs")
    def get_all_logs_optimized(self, limit: int = 500, days: int = 30) -> List[Dict]:
        """Get detection logs with time filtering and optimized query"""
        try:
            if not self.db:
                return self._get_mock_logs()
            
            # Calculate time threshold for better performance
            time_threshold = datetime.now() - timedelta(days=days)
            time_threshold_str = time_threshold.isoformat()
            
            # Optimized query with time filtering
            docs = (self.db.collection('fall_detections')
                   .where('timestamp', '>=', time_threshold_str)
                   .order_by('timestamp', direction=firestore.Query.DESCENDING)
                   .limit(limit)
                   .stream())
            
            logs = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                logs.append(data)
            
            logging.info(f"Retrieved {len(logs)} logs (last {days} days, limit {limit})")
            return logs
            
        except Exception as e:
            logging.error(f"Error getting optimized logs: {e}")
            return self._get_mock_logs()
    
    def _get_mock_logs(self) -> List[Dict]:
        """Generate mock logs for demonstration when Firebase is not available"""
        import random
        from datetime import timedelta
        
        mock_logs = []
        base_time = datetime.now()
        
        for i in range(50):
            # Generate random timestamp within last 30 days
            random_days = random.randint(0, 30)
            random_hours = random.randint(0, 23)
            random_minutes = random.randint(0, 59)
            
            timestamp = base_time - timedelta(days=random_days, hours=random_hours, minutes=random_minutes)
            
            # Generate random confidence and type
            confidence = random.uniform(0.1, 0.95)
            is_fall = confidence > 0.7 or random.random() < 0.3  # 30% chance of fall for better demo
            
            mock_logs.append({
                'id': f'mock_{i}',
                'timestamp': timestamp.isoformat(),
                'confidence': confidence,
                'type': 'fall' if is_fall else 'normal',
                'label': 'Jatuh' if is_fall else 'Tidak Jatuh',
                'is_fall': is_fall,
                'details': f'Mock detection #{i+1} - {"Fall detected" if is_fall else "Normal activity"}',
                'screenshot_url': None
            })
        
        return mock_logs
    
    def get_statistics(self) -> Dict:
        """Get detection statistics from Firestore"""
        if not self.initialized:
            return {
                'total_detections': 0,
                'today_detections': 0,
                'this_week_detections': 0,
                'this_month_detections': 0
            }
            
        try:
            # Get total detections
            total_docs = self.db.collection('fall_detections').stream()
            total_count = sum(1 for _ in total_docs)
            
            # Get today's detections
            today = datetime.now().date().isoformat()
            today_docs = (self.db.collection('fall_detections')
                         .where('timestamp', '>=', f'{today}T00:00:00')
                         .where('timestamp', '<', f'{today}T23:59:59')
                         .stream())
            today_count = sum(1 for _ in today_docs)
            
            return {
                'total_detections': total_count,
                'today_detections': today_count,
                'this_week_detections': 0,  # TODO: Implement week calculation
                'this_month_detections': 0   # TODO: Implement month calculation
            }
            
        except Exception as e:
            logging.error(f"Failed to get statistics: {e}")
            return {
                'total_detections': 0,
                'today_detections': 0,
                'this_week_detections': 0,
                'this_month_detections': 0
            }
    
    def save_screenshot(self, image_data: bytes, filename: str) -> Optional[str]:
        """Save screenshot to Firebase Storage"""
        if not self.initialized or not self.bucket:
            logging.warning("Firebase Storage not available")
            return None
            
        try:
            blob = self.bucket.blob(f'screenshots/{filename}')
            blob.upload_from_string(image_data, content_type='image/jpeg')
            
            # Make the blob publicly accessible (optional)
            blob.make_public()
            
            return blob.public_url
            
        except Exception as e:
            logging.error(f"Failed to save screenshot: {e}")
            return None
    
    def update_system_health(self, health_data: Dict) -> bool:
        """Update system health status"""
        if not self.initialized:
            return False
            
        try:
            health_data['last_updated'] = datetime.now().isoformat()
            
            self.db.collection('system_health').document('current').set(
                health_data, merge=True
            )
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to update system health: {e}")
            return False
    
    def get_system_health(self) -> Dict:
        """Get current system health status"""
        if not self.initialized:
            return {'status': 'unknown', 'last_updated': None}
            
        try:
            doc = self.db.collection('system_health').document('current').get()
            if doc.exists:
                return doc.to_dict()
            else:
                return {'status': 'unknown', 'last_updated': None}
                
        except Exception as e:
            logging.error(f"Failed to get system health: {e}")
            return {'status': 'error', 'last_updated': None}

# Global Firebase handler instance
firebase_handler = None

def initialize_firebase(service_account_path: str = None, project_id: str = None):
    """Initialize global Firebase handler"""
    global firebase_handler
    firebase_handler = FirebaseHandler(service_account_path, project_id)
    return firebase_handler

def get_firebase_handler() -> Optional[FirebaseHandler]:
    """Get global Firebase handler instance"""
    return firebase_handler