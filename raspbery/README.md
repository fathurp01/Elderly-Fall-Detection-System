# Raspberry Pi Client - Fall Detection System

This is the Raspberry Pi client component that captures video, detects humans using YOLOv8, and sends bounding box data to the laptop server.

## Requirements

- Raspberry Pi 4 (recommended) or Raspberry Pi 3B+
- USB Camera or PiCamera
- Python 3.11
- Minimum 4GB RAM
- MicroSD card (32GB recommended)

## Installation

### 1. Update System
```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Install System Dependencies
```bash
# Install Python and pip
sudo apt install python3.11 python3.11-pip python3.11-venv -y

# Install OpenCV dependencies
sudo apt install libopencv-dev python3-opencv -y

# Install other system dependencies
sudo apt install libatlas-base-dev libhdf5-dev libhdf5-serial-dev libatlas-base-dev libjasper-dev libqtgui4 libqt4-test -y
```

### 3. Create Virtual Environment
```bash
cd /home/pi
mkdir fall_detection
cd fall_detection

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate
```

### 4. Install Python Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch for ARM architecture
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install -r requirements.txt
```

### 5. Download YOLOv8 Model
```bash
# The model will be automatically downloaded on first run
# Or manually download:
python -c "from ultralytics import YOLO; YOLO('best.pt')"
```

## Configuration

### 1. Network Setup
Edit the `SERVER_URL` in `pi_client.py` to match your laptop's IP address:
```python
SERVER_URL = "http://192.168.1.100:5000"  # Change to your laptop's IP
```

### 2. Camera Setup
The script automatically detects USB cameras. For PiCamera:
1. Enable camera interface: `sudo raspi-config` → Interface Options → Camera → Enable
2. Uncomment PiCamera2 in requirements.txt and install it
3. Modify the camera initialization code if needed

### 3. Performance Tuning
For better performance on Raspberry Pi:
```python
# In pi_client.py, you can adjust:
SEQUENCE_LENGTH = 16  # Reduce for faster processing
FPS = 15  # Adjust camera FPS
CONFIDENCE_THRESHOLD = 0.5  # Adjust detection confidence
```

## Usage

### 1. Start the Script
```bash
source venv/bin/activate
python pi_client.py
```

### 2. Monitor Logs
The script will output logs showing:
- Camera initialization status
- YOLOv8 model loading
- Server connection status
- Detection and transmission status

### 3. Stop the Script
Press `Ctrl+C` to stop the script gracefully.

## Modes

### Per-Frame Mode
Sends bounding box data for each frame immediately:
```python
SEQUENCE_MODE = False
```

### Sequence Mode (Recommended)
Buffers frames and sends sequences for LSTM processing:
```python
SEQUENCE_MODE = True
SEQUENCE_LENGTH = 16
```

## Troubleshooting

### Camera Issues
```bash
# Check if camera is detected
lsusb  # For USB cameras
vcgencmd get_camera  # For PiCamera

# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"
```

### Memory Issues
```bash
# Increase GPU memory split
sudo raspi-config → Advanced Options → Memory Split → 128

# Monitor memory usage
free -h
htop
```

### Network Issues
```bash
# Check network connectivity
ping 192.168.1.100  # Replace with your laptop IP

# Check if server is running
telnet 192.168.1.100 5000
```

### Performance Issues
- Reduce camera resolution: `640x480` → `320x240`
- Lower FPS: `15` → `10`
- Use YOLOv8 nano model (already configured)
- Reduce sequence length: `16` → `8`

## Auto-Start on Boot (Optional)

### 1. Create Service File
```bash
sudo nano /etc/systemd/system/fall-detection.service
```

### 2. Service Configuration
```ini
[Unit]
Description=Fall Detection Pi Client
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/fall_detection
Environment=PATH=/home/pi/fall_detection/venv/bin
ExecStart=/home/pi/fall_detection/venv/bin/python pi_client.py
Restart=always

[Install]
WantedBy=multi-user.target
```

### 3. Enable Service
```bash
sudo systemctl enable fall-detection.service
sudo systemctl start fall-detection.service

# Check status
sudo systemctl status fall-detection.service
```

## Hardware Optimization

### For Raspberry Pi 4
- Use USB 3.0 port for camera
- Ensure adequate cooling
- Use high-speed MicroSD card (Class 10 or better)

### For Raspberry Pi 3B+
- Consider reducing processing load
- Use lower resolution/FPS
- Monitor temperature: `vcgencmd measure_temp`

## Data Format

The client sends data in these formats:

### Per-Frame Data
```json
{
  "frame_id": 123,
  "bbox": [x_min, y_min, x_max, y_max],
  "confidence": 0.85,
  "timestamp": "2025-01-12T10:30:15.123456"
}
```

### Sequence Data
```json
{
  "sequence": [
    [x1, y1, x2, y2],
    [x1, y1, x2, y2],
    ...
  ],
  "timestamps": [
    "2025-01-12T10:30:15.123456",
    "2025-01-12T10:30:15.156789",
    ...
  ]
}
```