# Fall Detection System for Elderly Care

A real-time fall detection system using LSTM neural networks and computer vision to monitor elderly individuals and provide immediate alerts when falls are detected.

## 🚀 Features

- **Real-time Fall Detection**: Uses LSTM model trained on movement patterns
- **Web Dashboard**: Live monitoring interface with real-time alerts
- **Firebase Integration**: Cloud storage for detection events and statistics
- **Multi-device Support**: Laptop server + Raspberry Pi client architecture
- **Socket.IO Communication**: Real-time bidirectional communication
- **Configurable Thresholds**: Adjustable confidence levels for detection
- **Comprehensive Logging**: Event logging with CSV export

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Raspberry Pi   │───▶│  Laptop Server  │───▶│    Firebase     │
│   (Camera +     │    │   (LSTM Model   │    │   (Cloud DB)    │
│  Detection)     │    │   + Dashboard)  │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  Web Dashboard  │
                       │ (Real-time UI)  │
                       └─────────────────┘
```

## 📋 Prerequisites

- Python 3.11+
- Conda or virtualenv
- Firebase account and project
- Webcam or camera module (for Raspberry Pi)

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd klasifikasi-lansia-jatuh
```

### 2. Environment Setup
```bash
# Create conda environment
conda create -n comvis python=3.11
conda activate comvis

# Install dependencies
cd laptop
pip install -r requirements.txt
```

### 3. Firebase Configuration
1. Create a Firebase project at [Firebase Console](https://console.firebase.google.com/)
2. Generate service account key (JSON file)
3. Place the JSON file in the `laptop/` directory
4. Update the Firebase configuration in `.env` file

### 4. Environment Variables
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env
```

Update the following variables in `.env`:
- `FIREBASE_SERVICE_ACCOUNT_PATH`: Path to your Firebase service account JSON file
- `FIREBASE_PROJECT_ID`: Your Firebase project ID
- `FIREBASE_DATABASE_URL`: Your Firebase Realtime Database URL
- `SECRET_KEY`: Change to a secure secret key for production

## 🚀 Usage

### Laptop Server (Main Processing Unit)
```bash
cd laptop
conda activate comvis
python server.py --port 5000
```

### Raspberry Pi Client (Camera Module)
```bash
cd raspbery
python pi_client.py
```

### Web Dashboard
Open your browser and navigate to:
```
http://localhost:5000
```

## 📊 Dashboard Features

- **Live Video Feed**: Real-time camera stream
- **Detection Status**: Current system status and alerts
- **Statistics**: Fall detection counts and confidence levels
- **Event Logs**: Historical detection events
- **System Health**: Server and model status monitoring

## 🔧 Configuration

### Model Parameters
- **Sequence Length**: 60 frames (adjustable)
- **Confidence Threshold**: 0.7 (adjustable)
- **Input Features**: 4 (x, y, width, height of bounding boxes)

### Server Settings
- **Host**: 0.0.0.0 (configurable)
- **Port**: 5000 (configurable)
- **Debug Mode**: Disabled in production

## 📁 Project Structure

```
klasifikasi-lansia-jatuh/
├── laptop/                 # Main server application
│   ├── server.py          # Flask server with Socket.IO
│   ├── lstm_model.py      # LSTM model handler
│   ├── firebase_config.py # Firebase integration
│   ├── lstm_fall.keras    # Trained model file
│   ├── requirements.txt   # Python dependencies
│   └── templates/         # Web dashboard templates
│       ├── dashboard.html
│       ├── logs.html
│       └── system_check.html
├── raspbery/              # Raspberry Pi client
│   ├── pi_client.py      # Camera client application
│   ├── best.pt           # YOLO model for person detection
│   └── requirements.txt  # Pi-specific dependencies
├── Reference/             # Development references
│   ├── LSTM_fall.ipynb   # Model training notebook
│   └── peopleDetect.py   # Detection utilities
├── .env.example          # Environment configuration template
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## 🔒 Security Considerations

- Firebase service account keys are gitignored
- Environment variables for sensitive configuration
- HTTPS recommended for production deployment
- Regular security updates for dependencies

## 📈 Performance Optimization

- **Model Optimization**: TensorFlow Lite for edge deployment
- **Caching**: Redis integration for session management
- **Load Balancing**: Multiple server instances support
- **Database Indexing**: Optimized Firebase queries

## 🐛 Troubleshooting

### Common Issues

1. **TensorFlow Not Available**
   ```bash
   conda activate comvis
   pip install tensorflow==2.13.0
   ```

2. **Firebase Connection Error**
   - Check service account JSON path
   - Verify Firebase project configuration
   - Ensure internet connectivity

3. **Model File Not Found**
   - Ensure `lstm_fall.keras` is in laptop directory
   - Check file permissions

4. **Camera Access Issues**
   - Verify camera permissions
   - Check camera device index
   - Ensure camera is not used by other applications

## 📝 Logging

- **Console Logs**: Real-time system status
- **CSV Logs**: Detection events with timestamps
- **Firebase Logs**: Cloud-stored event history
- **Error Logs**: Detailed error tracking

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Development Team** - Initial work and maintenance

## 🙏 Acknowledgments

- TensorFlow team for machine learning framework
- Firebase team for cloud infrastructure
- OpenCV community for computer vision tools
- Flask and Socket.IO communities for web framework

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the documentation

---

**Note**: This system is designed for monitoring and alert purposes. It should not replace professional medical care or supervision.