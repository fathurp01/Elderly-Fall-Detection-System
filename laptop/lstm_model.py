#!/usr/bin/env python3
"""
LSTM Model Handler for Fall Detection
Placeholder implementation - replace with actual trained model
"""

import numpy as np
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Using TensorFlow/Keras for the trained model
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import backend as K
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available, using placeholder model")

logger = logging.getLogger(__name__)

# ====== FOCAL LOSS IMPLEMENTATION (sesuai dengan training) ======
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss untuk mengatasi class imbalance dengan fokus pada hard examples.
    gamma: focusing parameter (higher = lebih fokus pada hard examples)
    alpha: balancing parameter untuk class imbalance
    """
    def focal_loss_fixed(y_true, y_pred):
        if not TENSORFLOW_AVAILABLE:
            return 0.0
            
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # Calculate focal loss
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)

        focal_loss_value = -alpha_t * K.pow((1 - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss_value)

    return focal_loss_fixed

class LSTMModel:
    def __init__(self, model_path=None, sequence_length=41, input_features=13):
        """
        Initialize LSTM Model
        
        Args:
            model_path: Path to the trained model file
            sequence_length: Length of input sequences (41 frames as per training)
            input_features: Number of input features (13 for enhanced features)
        """
        self.model_path = model_path or os.getenv('LSTM_MODEL_PATH', 'best3.keras')  # Use the actual trained model
        self.sequence_length = sequence_length
        self.input_features = input_features
        self.model = None
        self.is_loaded = False
        
        # Training parameters (sesuai dengan untitled15 (2).py)
        self.TARGET_FPS = 30
        self.TARGET_SEQUENCE_LENGTH = 61  # Sesuai dengan training
        self.BATCH_SIZE = 32
        self.NUM_FEATURES = 13  # Sesuai dengan training: 13 features
        self.LOCF_MAX_FRAMES = 5
        self.EMA_ALPHA = 0.3
        self.PADDING_VALUE = -1.0
        
        # Initialize scalers
        self.scalers = None
        
        # Load model if it exists
        self.load_model()
        
    def load_model(self):
        """
        Load pre-trained LSTM model and scaler with optimization
        """
        try:
            if os.path.exists(self.model_path) and TENSORFLOW_AVAILABLE:
                logger.info(f"Loading LSTM model from {self.model_path}...")
                
                # Load the trained model without compilation for faster loading
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
                
                # Compile once with optimized settings
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Warm up the model with a dummy prediction for faster subsequent predictions
                dummy_input = np.zeros((1, self.sequence_length, self.input_features), dtype=np.float32)
                _ = self.model.predict(dummy_input, verbose=0)
                
                logger.info(f"LSTM model loaded and warmed up successfully from {self.model_path}")
                self.is_loaded = True
                
                # Load scaler (only once during initialization)
                self._load_scaler()
                
            else:
                if not os.path.exists(self.model_path):
                    logger.warning(f"Model file not found: {self.model_path}")
                if not TENSORFLOW_AVAILABLE:
                    logger.warning("TensorFlow not available")
                logger.info("Using placeholder model for demonstration")
                self.is_loaded = False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Using placeholder prediction method")
            self.is_loaded = False
            
    def _load_scaler(self):
        """Load scaler from file with proper error handling"""
        try:
            import joblib
            # Try multiple scaler file locations
            scaler_paths = [
                os.path.join(os.path.dirname(self.model_path), 'scaler2.pkl'),
                os.path.join(os.path.dirname(self.model_path), 'fase_2c_feature_scalers.pkl'),
                'scaler2.pkl',
                'fase_2c_feature_scalers.pkl'
            ]
            
            for scaler_path in scaler_paths:
                if os.path.exists(scaler_path):
                    self.scalers = joblib.load(scaler_path)
                    logger.info(f"Scaler loaded successfully from {scaler_path}")
                    return
                    
            logger.warning("No scaler file found in expected locations")
            self.scalers = None
            
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")
            self.scalers = None
            
    def extract_features(self, bbox_sequence, frame_width=640, frame_height=480):
        """
        Extract 13 features from bounding box sequence (sesuai dengan training untitled15 (2).py)
        
        Args:
            bbox_sequence: List of bounding boxes [[x1,y1,x2,y2], ...]
            frame_width: Frame width for normalization (default 640)
            frame_height: Frame height for normalization (default 480)
            
        Returns:
            List of 13-dimensional feature vectors
        """
        features_sequence = []
        
        # Initialize tracking variables (sesuai dengan training)
        prev_features = None
        prev_area = None
        prev_aspect = None
        prev_center_y = None
        prev_vy = None
        
        for i, bbox in enumerate(bbox_sequence):
            # Convert bbox to numpy array for safe comparison
            bbox_array = np.array(bbox) if not isinstance(bbox, np.ndarray) else bbox
            # Check if bbox is valid (not all zeros and has 4 elements)
            if len(bbox) == 4 and not np.array_equal(bbox_array, [0, 0, 0, 0]):  # Valid detection
                x1, y1, x2, y2 = bbox
                
                # Normalized bbox coordinates (sesuai training)
                nx = x1 / frame_width
                ny = y1 / frame_height
                nw = (x2 - x1) / frame_width
                nh = (y2 - y1) / frame_height
                
                # Center coordinates
                center_x = nx + nw / 2
                center_y = ny + nh / 2
                
                # Aspect ratio
                aspect_ratio = nh / (nw + 1e-6)
                
                # Area
                area = nw * nh
                
                # Presence feature (1 for valid detection)
                presence = 1.0
                
                # Temporal derivatives (sesuai training)
                if i == 0:
                    vx = vy = 0.0
                    dArea = dAspect = dCenterY = 0.0
                    vy_pos = 0.0
                else:
                    # Velocity
                    vx = center_x - prev_features['center_x'] if prev_features else 0.0
                    vy = center_y - prev_features['center_y'] if prev_features else 0.0
                    
                    # Derivatives
                    dArea = (area - prev_area) if prev_area is not None else 0.0
                    dAspect = (aspect_ratio - prev_aspect) if prev_aspect is not None else 0.0
                    dCenterY = (center_y - prev_center_y) if prev_center_y is not None else 0.0
                    
                    # Advanced velocity features
                    vy_pos = max(0.0, vy)  # Only positive (downward) vertical velocity
                
                # ====== 13-FEATURE VECTOR (sesuai training) ======
                # [nx, ny, nw, nh, vx, vy, aspect_ratio, presence, area, dArea, dAspect, dCenterY, vy_pos]
                features = [
                    nx, ny, nw, nh,           # 0-3: Basic bbox (normalized)
                    vx, vy,                   # 4-5: Velocity
                    aspect_ratio,             # 6: Aspect ratio
                    presence,                 # 7: Detection presence (0/1)
                    area,                     # 8: Bbox area
                    dArea,                    # 9: Area change rate
                    dAspect,                  # 10: Aspect ratio change rate
                    dCenterY,                 # 11: Vertical position change
                    vy_pos                    # 12: Positive vertical velocity
                ]
                
                features_sequence.append(features)
                
                # ====== UPDATE PREVIOUS VALUES ======
                prev_features = {
                    'nx': nx, 'ny': ny, 'nw': nw, 'nh': nh,
                    'center_x': center_x, 'center_y': center_y
                }
                prev_area = area
                prev_aspect = aspect_ratio
                prev_center_y = center_y
                prev_vy = vy
                
            else:
                # No detection - padding with -1.0 for Masking layer compatibility
                features_sequence.append([-1.0] * 13)
        
        # Apply scaling if scaler is available
        if hasattr(self, 'scalers') and self.scalers is not None:
            try:
                # Scale each feature dimension separately
                scaled_features = []
                for features in features_sequence:
                    scaled_feature_row = []
                    for feature_idx, feature_val in enumerate(features):
                        if feature_idx in self.scalers and feature_val != self.PADDING_VALUE:
                            scaler = self.scalers[feature_idx]
                            scaled_val = scaler.transform([[feature_val]])[0][0]
                            scaled_feature_row.append(scaled_val)
                        else:
                            scaled_feature_row.append(feature_val)
                    scaled_features.append(scaled_feature_row)
                features_sequence = scaled_features
                logger.debug("Features scaled successfully")
            except Exception as scale_error:
                logger.warning(f"Error applying scaler: {scale_error}. Using unscaled features.")
        
        return features_sequence
    
    def preprocess_sequence(self, bbox_sequence):
        """
        Preprocess bounding box sequence for LSTM input with 13 features
        
        Args:
            bbox_sequence: List of bounding boxes [[x1,y1,x2,y2], ...]
            
        Returns:
            Preprocessed numpy array ready for model input
        """
        try:
            # Extract 13 features from bbox sequence
            features_sequence = self.extract_features(bbox_sequence)
            
            # Convert to numpy array
            sequence = np.array(features_sequence, dtype=np.float32)
            
            # Ensure we have the right sequence length
            if len(sequence) > self.sequence_length:
                sequence = sequence[-self.sequence_length:]
            elif len(sequence) < self.sequence_length:
                # Pad with -1.0 for Masking layer compatibility
                padding = np.full((self.sequence_length - len(sequence), self.input_features), -1.0)
                sequence = np.vstack([padding, sequence])
            
            # Apply scaler if available (use the one loaded in constructor)
            if hasattr(self, 'scalers') and self.scalers is not None:
                try:
                    # Apply scaling sesuai dengan training (skip presence feature index 7 dan padding values)
                    for feature_idx in range(self.input_features):
                        if feature_idx == 7:  # Skip presence feature (binary)
                            continue
                            
                        if feature_idx in self.scalers:
                            # Create mask for non-padding values
                            mask = sequence[:, feature_idx] != self.PADDING_VALUE
                            if np.any(mask):
                                # Only scale non-padding values
                                non_padding_values = sequence[:, feature_idx][mask]
                                scaled_values = self.scalers[feature_idx].transform(non_padding_values.reshape(-1, 1)).flatten()
                                sequence[:, feature_idx][mask] = scaled_values
                except Exception as e:
                    logger.warning(f"Could not apply scaling: {e}")
            else:
                logger.warning("No scalers available for preprocessing")
                
            # Add batch dimension for model input
            sequence = sequence.reshape(1, self.sequence_length, self.input_features)
            
            return sequence
            
        except Exception as e:
            logger.error(f"Error preprocessing sequence: {e}")
            return None
            
    def extract_features_from_bbox(self, bbox_data):
        """
        Extract 13 features from bounding box data (sesuai dengan training)
        
        Features: [nx, ny, nw, nh, vx, vy, aspect_ratio, presence, area, dArea, dAspect, dCenterY, vy_pos]
        
        Args:
            bbox_data: List of bbox dictionaries with keys: x, y, w, h, confidence, timestamp
            
        Returns:
            numpy array of shape (sequence_length, 13)
        """
        if not bbox_data:
            # Return padding sequence if no data
            return np.full((self.sequence_length, self.input_features), self.PADDING_VALUE, dtype=np.float32)
        
        sequence = []
        
        # Variables for temporal derivatives
        prev_area = None
        prev_aspect = None
        prev_center_y = None
        prev_vy = None
        
        # Process each frame's bbox data
        for i, bbox in enumerate(bbox_data):
            if bbox is None:
                # No detection in this frame - use padding
                features = [self.PADDING_VALUE] * self.input_features
            else:
                # Extract basic features
                x = bbox.get('x', 0)
                y = bbox.get('y', 0) 
                w = bbox.get('w', 0)
                h = bbox.get('h', 0)
                
                # Normalize coordinates (assuming frame size 640x480 as default)
                frame_width = bbox.get('frame_width', 640)
                frame_height = bbox.get('frame_height', 480)
                
                nx = x / frame_width
                ny = y / frame_height
                nw = w / frame_width
                nh = h / frame_height
                
                # Calculate derived features
                center_x = nx + nw / 2
                center_y = ny + nh / 2
                area = nw * nh
                aspect_ratio = nh / (nw + 1e-6)
                
                # Calculate velocity (if previous frame exists)
                if i > 0 and len(sequence) > 0 and sequence[-1][7] > 0:  # presence check
                    prev_center_x = sequence[-1][0] + sequence[-1][2] / 2  # prev nx + prev nw/2
                    prev_center_y = sequence[-1][1] + sequence[-1][3] / 2  # prev ny + prev nh/2
                    vx = center_x - prev_center_x
                    vy = center_y - prev_center_y
                else:
                    vx = vy = 0.0
                
                # Temporal derivatives
                if i == 0:
                    dArea = dAspect = dCenterY = 0.0
                    vy_pos = 0.0
                else:
                    # Area change rate
                    dArea = (area - prev_area) if prev_area is not None else 0.0
                    # Aspect ratio change rate
                    dAspect = (aspect_ratio - prev_aspect) if prev_aspect is not None else 0.0
                    # Vertical position change
                    dCenterY = (center_y - prev_center_y) if prev_center_y is not None else 0.0
                    # Positive vertical velocity (downward movement)
                    vy_pos = max(0.0, vy)
                
                # 13-Feature vector sesuai training:
                # [nx, ny, nw, nh, vx, vy, aspect_ratio, presence, area, dArea, dAspect, dCenterY, vy_pos]
                features = [
                    nx, ny, nw, nh,           # 0-3: Basic bbox (normalized)
                    vx, vy,                   # 4-5: Velocity
                    aspect_ratio,             # 6: Aspect ratio
                    1.0,                      # 7: Detection presence (1.0)
                    area,                     # 8: Bbox area
                    dArea,                    # 9: Area change rate
                    dAspect,                  # 10: Aspect ratio change rate
                    dCenterY,                 # 11: Vertical position change
                    vy_pos                    # 12: Positive vertical velocity
                ]
                
                # Update previous values for next iteration
                prev_area = area
                prev_aspect = aspect_ratio
                prev_center_y = center_y
                prev_vy = vy
            
            sequence.append(features)
        
        # Pad or truncate to sequence_length
        if len(sequence) < self.sequence_length:
            # Pad with PADDING_VALUE
            padding_needed = self.sequence_length - len(sequence)
            padding_features = [self.PADDING_VALUE] * self.input_features
            sequence.extend([padding_features] * padding_needed)
        elif len(sequence) > self.sequence_length:
            sequence = sequence[:self.sequence_length]
        
        return np.array(sequence, dtype=np.float32)
            
    def predict(self, bbox_sequence):
        """
        Predict fall/no-fall from bounding box sequence
        
        Args:
            bbox_sequence: List of bounding boxes [[x1,y1,x2,y2], ...]
            
        Returns:
            Dictionary with 'label' and 'confidence'
        """
        try:
            # Preprocess the sequence
            processed_sequence = self.preprocess_sequence(bbox_sequence)
            
            if processed_sequence is None:
                return self._get_default_prediction()
                
            if self.is_loaded and self.model is not None and TENSORFLOW_AVAILABLE:
                # Use the actual trained TensorFlow model
                prediction = self.model.predict(processed_sequence, verbose=0)
                confidence = float(prediction[0][0])  # Binary classification output
                
                # Apply confidence threshold (sesuai dengan training)
                # Model output is probability of "Jatuh" (fall)
                if confidence > 0.5:
                    label = "Jatuh"
                    confidence_score = confidence
                else:
                    label = "Tidak Jatuh"
                    confidence_score = 1 - confidence  # Confidence for "Tidak Jatuh"
                
                logger.debug(f"LSTM prediction: {label} with confidence {confidence_score:.3f}")
                
                return {
                    'label': label,
                    'confidence': confidence_score
                }
            else:
                # Use placeholder prediction when model is not loaded
                return self._get_placeholder_prediction(bbox_sequence)
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return self._get_default_prediction()
            
    def _get_placeholder_prediction(self, bbox_sequence):
        """
        Placeholder prediction logic for demonstration
        Replace this with actual model prediction
        """
        try:
            # Simple heuristic for demonstration:
            # Analyze bounding box movement patterns
            
            if len(bbox_sequence) < 2:
                return self._get_default_prediction()
                
            # Calculate movement metrics
            movements = []
            height_changes = []
            
            for i in range(1, len(bbox_sequence)):
                prev_bbox = bbox_sequence[i-1]
                curr_bbox = bbox_sequence[i]
                
                # Calculate center movement
                prev_center_x = (prev_bbox[0] + prev_bbox[2]) / 2
                prev_center_y = (prev_bbox[1] + prev_bbox[3]) / 2
                curr_center_x = (curr_bbox[0] + curr_bbox[2]) / 2
                curr_center_y = (curr_bbox[1] + curr_bbox[3]) / 2
                
                movement = np.sqrt((curr_center_x - prev_center_x)**2 + (curr_center_y - prev_center_y)**2)
                movements.append(movement)
                
                # Calculate height change (bbox height)
                prev_height = prev_bbox[3] - prev_bbox[1]
                curr_height = curr_bbox[3] - curr_bbox[1]
                height_change = abs(curr_height - prev_height)
                height_changes.append(height_change)
                
            # Simple fall detection heuristic
            avg_movement = np.mean(movements) if movements else 0
            avg_height_change = np.mean(height_changes) if height_changes else 0
            
            # Calculate fall probability based on movement patterns
            # High movement + significant height change might indicate a fall
            fall_score = (avg_movement * 0.6 + avg_height_change * 0.4) / 100
            fall_score = max(0, min(1, fall_score))  # Clamp to [0,1]
            
            # Use deterministic thresholds for more consistent predictions
            # Most activities should be classified as normal unless clear fall patterns
            if avg_movement > 50 and avg_height_change > 30:
                fall_score = min(0.8, fall_score + 0.3)  # Likely fall
            elif avg_movement < 20 and avg_height_change < 10:
                fall_score = max(0.1, fall_score - 0.2)  # Likely normal
            # else: keep calculated fall_score
            
            # Determine label based on threshold
            if fall_score > 0.5:
                label = "Jatuh"
                confidence = fall_score
            else:
                label = "Tidak Jatuh"
                confidence = 1 - fall_score
                
            return {
                'label': label,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error in placeholder prediction: {e}")
            return self._get_default_prediction()
            
    def _get_default_prediction(self):
        """
        Default prediction when errors occur
        """
        return {
            'label': "Tidak Jatuh",
            'confidence': 0.5
        }
        
    def save_model(self, model, save_path=None):
        """
        Save trained model
        
        Args:
            model: Trained model object
            save_path: Path to save the model
        """
        save_path = save_path or self.model_path
        
        try:
            # TODO: Implement model saving
            # For PyTorch:
            # torch.save(model, save_path)
            
            # For TensorFlow:
            # model.save(save_path)
            
            logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            
    def get_model_info(self):
        """
        Get information about the loaded model
        """
        return {
            'model_path': self.model_path,
            'is_loaded': self.is_loaded,
            'sequence_length': self.sequence_length,
            'input_features': self.input_features,
            'model_type': 'LSTM for Fall Detection'
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the LSTM model
    model = LSTMModel()
    
    # Create sample bounding box sequence
    sample_sequence = [
        [100, 100, 200, 300],  # Standing position
        [105, 105, 205, 305],  # Slight movement
        [110, 120, 210, 320],  # More movement
        [120, 150, 220, 350],  # Falling motion
        [150, 200, 250, 400],  # On ground
    ] * 4  # Repeat to get 20 frames, will be trimmed to 16
    
    # Get prediction
    result = model.predict(sample_sequence)
    print(f"Prediction: {result['label']} (confidence: {result['confidence']:.2f})")
    
    # Print model info
    info = model.get_model_info()
    print(f"Model info: {info}")