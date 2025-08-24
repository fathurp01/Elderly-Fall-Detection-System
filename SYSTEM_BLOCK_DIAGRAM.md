# Diagram Blok Sistem Fall Detection Website

## Arsitektur Sistem dengan Struktur Input-Proses-Output

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                              INPUT                                                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐                │
│  │     Camera       │    │   Environment    │    │   User Config    │    │   Model Files    │                │
│  │  (USB/PiCam)     │    │   (Lighting,     │    │  (Thresholds,    │    │  (best3.keras,   │                │
│  │   640x480        │    │    Movement)     │    │   Settings)      │    │   scaler2.pkl)   │                │
│  │   Real-time      │    │                  │    │                  │    │                  │                │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘    └──────────────────┘                │
│           │                        │                        │                        │                        │
└───────────┼────────────────────────┼────────────────────────┼────────────────────────┼────────────────────────┘
            │                        │                        │                        │
            ▼                        ▼                        ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                             PROSES                                                             │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                                    RASPBERRY PI CLIENT                                                   │   │
│  │                                                                                                           │   │
│  │  ┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐                │   │
│  │  │ Frame Capture│───▶│   YOLOv8 Human  │───▶│  Bounding Box    │───▶│   Sequence      │                │   │
│  │  │ & Preprocessing│   │   Detection     │    │   Extraction     │    │   Buffer        │                │   │
│  │  │ (640x480)    │    │ (Person Class)  │    │  [x1,y1,x2,y2]   │    │  (61 frames)    │                │   │
│  │  └──────────────┘    └─────────────────┘    └──────────────────┘    └─────────────────┘                │   │
│  │                              │                                                │                          │   │
│  │                     ┌────────▼────────┐                                       │                          │   │
│  │                     │  Frame Skipping │                                       │                          │   │
│  │                     │ & Optimization  │                                       │                          │   │
│  │                     │ (Every 3rd)     │                                       │                          │   │
│  │                     └─────────────────┘                                       │                          │   │
│  │                                                                               │                          │   │
│  │  ┌─────────────────────────────────────────────────────────────────────────────┼──────────────────────┐   │   │
│  │  │                              SOCKETIO CLIENT                               │                      │   │   │
│  │  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │                      │   │   │
│  │  │  │  Connection     │    │   Data Queue    │    │   Auto-Reconnect│        │                      │   │   │
│  │  │  │  Management     │    │   (Threading)   │    │   & Resilience  │        │                      │   │   │
│  │  │  └─────────────────┘    └─────────────────┘    └─────────────────┘        │                      │   │   │
│  │  └─────────────────────────────────────────────────────────────────────────────┼──────────────────────┘   │   │
│  │                                                                               │                          │   │
│  │                                                      ┌─────────────────────────▼──────────────────────┐   │   │
│  │                                                      │         JSON Payload Transmission             │   │   │
│  │                                                      │    {frame_id, bbox, confidence, timestamp}    │   │   │
│  │                                                      └────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                      │                                         │
│                                                            ┌─────────▼─────────┐                               │
│                                                            │      NETWORK      │                               │
│                                                            │  (WiFi/Ethernet)  │                               │
│                                                            │  WebSocket/HTTP   │                               │
│                                                            └─────────┬─────────┘                               │
│                                                                      │                                         │
│  ┌─────────────────────────────────────────────────────────────────┼─────────────────────────────────────────┐ │
│  │                                    LAPTOP SERVER                 │                                         │ │
│  │                                                                  │                                         │ │
│  │                                                        ┌────────▼────────┐                                │ │
│  │                                                        │  Flask-SocketIO │                                │ │
│  │                                                        │     Server      │                                │ │
│  │                                                        │   (Port 5000)   │                                │ │
│  │                                                        └────────┬────────┘                                │ │
│  │                                                                 │                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────┼─────────────────────────────────────────┐ │ │
│  │  │                            DATA PROCESSING PIPELINE          │                                         │ │ │
│  │  │                                                             │                                         │ │ │
│  │  │  ┌─────────────────┐    ┌─────────────────┐    ┌──────────▼──────────┐    ┌─────────────────┐        │ │ │
│  │  │  │   Sequence      │    │   Feature       │    │     LSTM Model      │    │   Prediction    │        │ │ │
│  │  │  │  Validation     │───▶│  Extraction     │───▶│   (TensorFlow)      │───▶│   Processing    │        │ │ │
│  │  │  │  (61 frames)    │    │  (13 features)  │    │   best3.keras       │    │  (Jatuh/Tidak)  │        │ │ │
│  │  │  └─────────────────┘    └─────────────────┘    └─────────────────────┘    └─────────────────┘        │ │ │
│  │  │                                │                                                   │                  │ │ │
│  │  │                         ┌──────▼──────┐                                    ┌──────▼──────┐           │ │ │
│  │  │                         │   Scaler    │                                    │ Confidence  │           │ │ │
│  │  │                         │ (scaler2.pkl)│                                   │ Threshold   │           │ │ │
│  │  │                         └─────────────┘                                    │   (>0.5)    │           │ │ │
│  │  │                                                                            └─────────────┘           │ │ │
│  │  └─────────────────────────────────────────────────────────────────────────────────┼─────────────────────┘ │ │
│  │                                                                                      │                   │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────┼─────────────────────┐ │ │
│  │  │                            SMART FILTERING SYSTEM                                │                   │ │ │
│  │  │                                                                                   │                   │ │ │
│  │  │                                                    ┌──────────────────────────────▼──────────────────┐ │ │ │
│  │  │                                                    │        Fall Detection Decision                  │ │ │ │
│  │  │                                                    └──────────────────────────────┬──────────────────┘ │ │ │
│  │  │                                                                                   │                   │ │ │
│  │  │  ┌─────────────────┐                                                    ┌────────▼────────┐          │ │ │
│  │  │  │      JATUH      │                                                    │   TIDAK JATUH   │          │ │ │
│  │  │  │   (Fall Event)  │                                                    │ (Normal Activity)│          │ │ │
│  │  │  └─────────┬───────┘                                                    └────────┬────────┘          │ │ │
│  │  │            │                                                                     │                   │ │ │
│  │  │    ┌───────▼───────┐                                                   ┌────────▼────────┐          │ │ │
│  │  │    │  Save to DB   │                                                   │ Smart Filtering │          │ │ │
│  │  │    │   (100%)      │                                                   │  - Sampling     │          │ │ │
│  │  │    └───────────────┘                                                   │  - Time Filter  │          │ │ │
│  │  │                                                                        │  - Confidence   │          │ │ │
│  │  │                                                                        └────────┬────────┘          │ │ │
│  │  │                                                                                 │                   │ │ │
│  │  │                                                                        ┌────────▼────────┐          │ │ │
│  │  │                                                                        │  Save to DB     │          │ │ │
│  │  │                                                                        │    (~10%)       │          │ │ │
│  │  │                                                                        └─────────────────┘          │ │ │
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                              │
└─────────────────────────────────────────────────────────────────┼────────────────────────────────────────────┘
                                                                  │
                                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                             OUTPUT                                                             │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                                    DATABASE & STORAGE                                                   │   │
│  │                                                                                                           │   │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐              │   │
│  │  │   Firebase      │    │   Local CSV     │    │   Cache System  │    │   Log Files     │              │   │
│  │  │   Firestore     │    │   (logs.csv)    │    │   (Redis)       │    │   (Debugging)   │              │   │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                                    WEB INTERFACE                                                        │   │
│  │                                                                                                           │   │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐              │   │
│  │  │   Dashboard     │    │   Real-time     │    │   Event Logs    │    │   System Stats  │              │   │
│  │  │  (dashboard.html)│    │   Monitoring    │    │  (logs.html)    │    │   & Health      │              │   │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘              │   │
│  │                                │                                                                         │   │
│  │                         ┌──────▼──────┐                                                                 │   │
│  │                         │  WebSocket  │                                                                 │   │
│  │                         │ Real-time   │                                                                 │   │
│  │                         │  Updates    │                                                                 │   │
│  │                         └─────────────┘                                                                 │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                                    NOTIFICATION SYSTEM                                                  │   │
│  │                                                                                                           │   │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐              │   │
│  │  │   Email Alert   │    │   SMS/WhatsApp  │    │   Web Push      │    │   System Alert  │              │   │
│  │  │   (Optional)    │    │   (Optional)    │    │  Notification   │    │   (Console)     │              │   │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                                    MOBILE INTERFACE                                                     │   │
│  │                                                                                                           │   │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐              │   │
│  │  │   Mobile App    │    │   Push Notif    │    │   Emergency     │    │   Remote        │              │   │
│  │  │   (Optional)    │    │   (Fall Alert)  │    │   Contact       │    │   Monitoring    │              │   │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Komponen Utama Sistem

### 1. Raspberry Pi Client (`raspbery/pi_client.py`)
- **Camera Input**: USB Camera atau PiCamera (640x480)
- **YOLOv8 Detection**: Human detection dengan optimasi RPi 4
- **Frame Processing**: Skip setiap 3 frame untuk optimasi
- **Sequence Buffer**: Buffer 61 frame untuk LSTM
- **SocketIO Client**: Komunikasi real-time dengan server
- **Auto-Reconnect**: Resiliensi koneksi dengan exponential backoff

### 2. Laptop Server (`laptop/server.py`)
- **Flask-SocketIO Server**: Web server pada port 5000
- **Data Processing Pipeline**: Validasi dan preprocessing data
- **LSTM Model Handler**: Model TensorFlow untuk klasifikasi
- **Smart Filtering**: Filter untuk aktivitas normal
- **Database Integration**: Firebase Firestore dan CSV lokal
- **Web Dashboard**: Interface monitoring real-time

### 3. LSTM Model (`laptop/lstm_model.py`)
- **Model**: TensorFlow/Keras (best3.keras)
- **Features**: 13 fitur dari bounding box sequence
- **Scaler**: Normalisasi data (scaler2.pkl)
- **Sequence Length**: 61 frame input
- **Output**: Binary classification (Jatuh/Tidak Jatuh)

## Alur Data Sistem

### 1. Data Acquisition (Raspberry Pi)
```
Camera → Frame Capture → YOLOv8 → Bounding Box → Sequence Buffer → JSON Payload
```

### 2. Data Transmission
```
Raspberry Pi → WebSocket/HTTP → Laptop Server
```

### 3. Data Processing (Laptop Server)
```
JSON Input → Sequence Validation → Feature Extraction → LSTM Prediction → Smart Filtering → Database Storage
```

### 4. User Interface
```
Database → Web Dashboard → Real-time Updates → User Monitoring
```

## Format Data

### Input dari Raspberry Pi (Per Frame)
```json
{
  "frame_id": 123,
  "bbox": [x_min, y_min, x_max, y_max],
  "confidence": 0.85,
  "timestamp": "2025-01-12T10:30:15.123456"
}
```

### Input Sequence Mode
```json
{
  "sequence_id": "seq_001",
  "bbox_sequence": [
    [x1, y1, x2, y2],
    [x1, y1, x2, y2],
    ...
  ],
  "timestamps": [...],
  "sequence_length": 61
}
```

### Output Prediction
```json
{
  "label": "Jatuh",
  "confidence": 0.87,
  "timestamp": "2025-01-12T10:30:15.123456",
  "sequence_id": "seq_001"
}
```

## Fitur Sistem

### 1. Optimasi Raspberry Pi
- Frame skipping (setiap 3 frame)
- Threading untuk I/O non-blocking
- Adaptive compression berdasarkan network
- Hardware encoder detection
- Memory management

### 2. Smart Filtering
- Sampling rate untuk aktivitas normal
- Time-based filtering
- Confidence threshold filtering
- Mengurangi false positive

### 3. Real-time Monitoring
- WebSocket untuk update real-time
- Dashboard dengan statistik sistem
- Log viewer untuk debugging
- System health monitoring

### 4. Resiliensi Sistem
- Auto-reconnect dengan exponential backoff
- Error handling dan logging
- Fallback mechanisms
- Connection monitoring

## Teknologi yang Digunakan

### Raspberry Pi
- Python 3.x
- OpenCV untuk computer vision
- YOLOv8 (Ultralytics) untuk deteksi manusia
- SocketIO untuk komunikasi
- Threading untuk optimasi

### Laptop Server
- Flask + Flask-SocketIO
- TensorFlow/Keras untuk LSTM
- Firebase Admin SDK
- NumPy, Pandas untuk data processing
- HTML/CSS/JavaScript untuk dashboard

### Database
- Firebase Firestore (cloud)
- CSV files (local backup)
- Redis (caching, optional)

## Konfigurasi Sistem

### Environment Variables
```bash
# Raspberry Pi
SERVER_URL=http://10.42.117.102:5001
YOLO_MODEL_PATH=yolov8n.pt

# Laptop Server
LSTM_MODEL_PATH=best3.keras
FIREBASE_CREDENTIALS=path/to/credentials.json
SEQUENCE_LENGTH=61
CONFIDENCE_THRESHOLD=0.7
```

### Port Configuration
- **Web Server**: 5000 (HTTP/WebSocket)
- **Alternative**: 5001 (backup)
- **UDP Video**: 5002 (fallback streaming)

Sistem ini dirancang untuk deteksi jatuh lansia secara real-time dengan akurasi tinggi dan optimasi untuk hardware terbatas (Raspberry Pi 4).