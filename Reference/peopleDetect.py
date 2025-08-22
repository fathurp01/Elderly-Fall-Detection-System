import cv2
import numpy as np
from ultralytics import YOLO
import time

class HumanDetector:
    def __init__(self, model_path="best.pt"):
        """
        Inisialisasi detector dengan model YOLOv8
        
        Args:
            model_path (str): Path ke file model .pt
        """
        try:
            # Load model YOLOv8
            self.model = YOLO(model_path)
            print(f"Model berhasil dimuat dari: {model_path}")
            
            # Pengaturan warna untuk bounding box (BGR format)
            self.colors = {
                'person': (0, 255, 0),  # Hijau untuk manusia
                'box': (255, 0, 0),     # Biru untuk box
                'text': (255, 255, 255) # Putih untuk text
            }
            
            # Confidence threshold
            self.conf_threshold = 0.5
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def detect_humans(self, frame):
        """
        Deteksi manusia dalam frame
        
        Args:
            frame: Frame dari webcam
            
        Returns:
            frame: Frame dengan bounding box
            detections: List deteksi
        """
        # Prediksi menggunakan model
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detections = []
        
        # Proses hasil deteksi
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Dapatkan koordinat bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Dapatkan nama kelas
                    class_name = self.model.names[class_id]
                    
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(confidence),
                        'class': class_name
                    })
        
        return frame, detections
    
    def draw_detections(self, frame, detections):
        """
        Gambar bounding box dan label pada frame
        
        Args:
            frame: Frame original
            detections: List deteksi
            
        Returns:
            frame: Frame dengan annotation
        """
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            
            # Gambar bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['person'], 2)
            
            # Buat label
            label = f"{class_name}: {confidence:.2f}"
            
            # Dapatkan ukuran text
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Gambar background untuk text
            cv2.rectangle(
                frame, 
                (x1, y1 - text_height - 10), 
                (x1 + text_width, y1), 
                self.colors['person'], 
                -1
            )
            
            # Gambar text
            cv2.putText(
                frame, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                self.colors['text'], 
                2
            )
        
        return frame
    
    def run_webcam_detection(self, camera_index=0):
        """
        Jalankan deteksi real-time menggunakan webcam
        
        Args:
            camera_index (int): Index kamera (0 untuk kamera default)
        """
        # Inisialisasi webcam
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Tidak dapat membuka webcam")
            return
        
        # Set resolusi webcam (opsional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Deteksi manusia dimulai. Tekan 'q' untuk keluar, 's' untuk screenshot")
        print(f"Confidence threshold: {self.conf_threshold}")
        
        # Variabel untuk FPS
        fps_counter = 0
        start_time = time.time()
        
        while True:
            # Baca frame dari webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Tidak dapat membaca frame dari webcam")
                break
            
            # Flip frame secara horizontal (mirror effect)
            frame = cv2.flip(frame, 1)
            
            # Deteksi manusia
            frame, detections = self.detect_humans(frame)
            
            # Gambar hasil deteksi
            frame = self.draw_detections(frame, detections)
            
            # Hitung dan tampilkan FPS
            fps_counter += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = fps_counter / elapsed_time
                fps_counter = 0
                start_time = time.time()
            else:
                fps = fps_counter / elapsed_time if elapsed_time > 0 else 0
            
            # Tampilkan informasi pada frame
            info_text = f"Deteksi: {len(detections)} | FPS: {fps:.1f}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Tampilkan frame
            cv2.imshow('YOLOv8 Human Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Keluar dari program...")
                break
            elif key == ord('s'):
                # Screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"detection_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot disimpan: {filename}")
            elif key == ord('c'):
                # Toggle confidence threshold
                if self.conf_threshold == 0.5:
                    self.conf_threshold = 0.3
                else:
                    self.conf_threshold = 0.5
                print(f"Confidence threshold diubah ke: {self.conf_threshold}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

def main():
    """
    Fungsi main untuk menjalankan program
    """
    try:
        # Inisialisasi detector
        detector = HumanDetector("best.pt")
        
        # Jalankan deteksi webcam
        detector.run_webcam_detection(camera_index=0)
        
    except FileNotFoundError:
        print("Error: File model 'best.pt' tidak ditemukan!")
        print("Pastikan file best.pt berada di direktori yang sama dengan script ini.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()