# Laptop Server - Fall Detection System

This is the laptop server component that receives bounding box data from Raspberry Pi, performs LSTM-based fall classification, and provides a real-time web dashboard.

## Features

- **Real-time Communication**: Flask-SocketIO server for receiving data from Raspberry Pi
- **LSTM Classification**: Machine learning model for fall detection
- **Web Dashboard**: Real-time monitoring interface
- **Data Logging**: CSV logging of all events
- **Alert System**: Real-time fall alerts and notifications

## Requirements

- Python 3.11
- Windows/Linux/macOS
- Minimum 4GB RAM
- Network connection to Raspberry Pi

## Installation

### 1. Create Virtual Environment
```bash
# Navigate to laptop folder
cd laptop

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### 3. Choose ML Framework
The system supports both PyTorch and TensorFlow. Choose one:

#### Option A: PyTorch (Recommended)
```bash
# Already included in requirements.txt
# No additional installation needed
```

#### Option B: TensorFlow
```bash
# Uncomment TensorFlow lines in requirements.txt
# Then install:
pip install tensorflow==2.13.0 keras==2.13.1
```

## Configuration

### 1. Network Setup
Ensure your laptop and Raspberry Pi are on the same network. Find your laptop's IP address:

```bash
# Windows
ipconfig

# Linux/macOS
ifconfig
```

Update the Raspberry Pi client to use your laptop's IP address.

### 2. Model Setup
The system includes a placeholder LSTM model. To use your trained model:

1. **For PyTorch**: Place your model file as `lstm_model.pth`
2. **For TensorFlow**: Place your model file as `lstm_model.h5`
3. Update the model loading code in `lstm_model.py`

### 3. Server Configuration
Edit configuration in `server.py`:
```python
# Server settings
SEQUENCE_LENGTH = 61        # Number of frames for LSTM
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for fall detection
HOST = '0.0.0.0'           # Listen on all interfaces
PORT = 5000                # Server port
```

## Usage

### 1. Start the Server
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# Start the server
python server.py
```

You should see:
```
INFO:__main__:Starting Fall Detection Server on 0.0.0.0:5000
INFO:__main__:Sequence length: 16
INFO:__main__:Confidence threshold: 0.7
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.1.100:5000
```

### 2. Access Dashboard
Open your web browser and navigate to:
- Local access: `http://localhost:5000`
- Network access: `http://YOUR_LAPTOP_IP:5000`

### 3. Connect Raspberry Pi
Start the Raspberry Pi client (see raspbery/README.md for instructions).

## Dashboard Features

### Main Status Display
- **Normal State**: Green card showing "System Normal"
- **Fall Detected**: Red pulsing card showing "FALL DETECTED!"
- **Confidence Score**: Real-time confidence percentage

### Statistics Panel
- **Total Frames**: Number of frames processed
- **Detections**: Number of human detections
- **Fall Events**: Number of fall events detected
- **Last Detection**: Timestamp of last detection

### Event Log
- Real-time log of all classification results
- Color-coded entries (green for normal, red for falls)
- Automatic scrolling and entry limiting

### Alerts
- Pop-up notifications for fall events
- Auto-dismissing alerts
- Sound notifications (browser dependent)

## Data Formats

### Input from Raspberry Pi

#### Per-Frame Data
```json
{
  "frame_id": 123,
  "bbox": [x_min, y_min, x_max, y_max],
  "confidence": 0.85,
  "timestamp": "2025-01-12T10:30:15.123456"
}
```

#### Sequence Data
```json
{
  "sequence": [
    [x1, y1, x2, y2],
    [x1, y1, x2, y2],
    ...
  ],
  "timestamps": [
    "2025-01-12T10:30:15.123456",
    "2025-01-12T10:30:15.189012",
    ...
  ]
}
```

### Output Logs
CSV format in `logs.csv`:
```csv
timestamp,label,confidence,bbox_data
2025-01-12T10:30:15.123456,Tidak Jatuh,0.85,"[[100,100,200,300],...]"
2025-01-12T10:30:16.123456,Jatuh,0.92,"[[150,200,250,400],...]"
```

## API Endpoints

### REST API
- `GET /`: Dashboard interface
- `GET /api/stats`: Current statistics
- `GET /api/recent_events`: Recent classification events

### WebSocket Events
- `bbox_data`: Receive bounding box data from Pi
- `prediction_result`: Broadcast classification results
- `fall_alert`: Broadcast fall alerts
- `stats_update`: Broadcast statistics updates

## Customization

### 1. LSTM Model Integration
Replace the placeholder in `lstm_model.py`:

```python
def load_model(self):
    # For PyTorch
    self.model = torch.load(self.model_path)
    self.model.eval()
    
    # For TensorFlow
    # self.model = keras.models.load_model(self.model_path)

def predict(self, bbox_sequence):
    processed_sequence = self.preprocess_sequence(bbox_sequence)
    
    # PyTorch prediction
    with torch.no_grad():
        output = self.model(torch.FloatTensor(processed_sequence))
        probability = torch.softmax(output, dim=1)
        confidence = float(torch.max(probability))
        predicted_class = int(torch.argmax(probability))
    
    label = "Jatuh" if predicted_class == 1 else "Tidak Jatuh"
    return {'label': label, 'confidence': confidence}
```

### 2. Notification Integration

#### Telegram Bot
```python
# Add to requirements.txt: python-telegram-bot==20.5

import telegram

def send_telegram_notification(self, alert_data):
    bot = telegram.Bot(token='YOUR_BOT_TOKEN')
    message = f"🚨 FALL DETECTED!\nTime: {alert_data['timestamp']}\nConfidence: {alert_data['confidence']:.2f}"
    bot.send_message(chat_id='YOUR_CHAT_ID', text=message)
```

#### Email Notification
```python
import smtplib
from email.mime.text import MIMEText

def send_email_notification(self, alert_data):
    msg = MIMEText(f"Fall detected at {alert_data['timestamp']}")
    msg['Subject'] = 'Fall Detection Alert'
    msg['From'] = 'your_email@gmail.com'
    msg['To'] = 'recipient@gmail.com'
    
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('your_email@gmail.com', 'your_password')
    server.send_message(msg)
    server.quit()
```

### 3. Dashboard Customization
Modify `templates/dashboard.html`:
- Change colors and styling
- Add new statistics
- Customize alert behavior
- Add charts and graphs

## Troubleshooting

### Server Won't Start
```bash
# Check if port is in use
netstat -an | findstr :5000  # Windows
lsof -i :5000  # Linux/macOS

# Use different port
python server.py --port 5001
```

### Raspberry Pi Can't Connect
```bash
# Check firewall settings
# Windows: Allow Python through Windows Firewall
# Linux: sudo ufw allow 5000

# Test connectivity
telnet YOUR_LAPTOP_IP 5000
```

### Model Loading Issues
```bash
# Check model file exists
ls lstm_model.pth  # or lstm_model.h5

# Check PyTorch/TensorFlow installation
python -c "import torch; print(torch.__version__)"
python -c "import tensorflow; print(tensorflow.__version__)"
```

### Performance Issues
- Reduce sequence length: `SEQUENCE_LENGTH = 30`
- Increase confidence threshold: `CONFIDENCE_THRESHOLD = 0.8`
- Monitor CPU/memory usage
- Use GPU acceleration if available

## Development

### Running in Debug Mode
```bash
python server.py --debug
```

### Testing with Sample Data
```python
# Test LSTM model
python lstm_model.py

# Send test data via curl
curl -X POST http://localhost:5000/test_data
```

### Log Levels
```python
# In server.py
logging.basicConfig(level=logging.DEBUG)  # More verbose
logging.basicConfig(level=logging.WARNING)  # Less verbose
```

## Production Deployment

### Using Gunicorn (Linux/macOS)
```bash
pip install gunicorn
gunicorn --worker-class eventlet -w 1 server:app --bind 0.0.0.0:5000
```

### Using Waitress (Windows)
```bash
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 server:app
```

### Systemd Service (Linux)
```ini
# /etc/systemd/system/fall-detection.service
[Unit]
Description=Fall Detection Server
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/laptop
Environment=PATH=/path/to/laptop/venv/bin
ExecStart=/path/to/laptop/venv/bin/python server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

## File Structure

```
laptop/
├── server.py              # Main server application
├── lstm_model.py          # LSTM model handler
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   └── dashboard.html    # Web dashboard
├── static/              # Static files (optional)
├── logs.csv             # Event logs (generated)
└── lstm_model.pth       # Trained model (add your own)
```