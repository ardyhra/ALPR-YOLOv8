from ultralytics import YOLO
import cv2
import torch

WEIGHTS = "runs/detect/lp_yolov8n2/weights/best.pt"  # ganti sesuai lokasi
DEVICE = 0 if torch.cuda.is_available() else "cpu"  # gunakan GPU jika ada
CONF = 0.25
IMGSZ = 640

def main():
    model = YOLO(WEIGHTS)
    cap = cv2.VideoCapture(0)  # webcam

    if not cap.isOpened():
        print("Webcam tidak terbuka.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # prediksi (stream=True untuk iterasi cepat)
        results = model.predict(source=frame, imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False)
        # draw hasil ke frame
        annotated = results[0].plot()  # Ultralytics sudah sediakan visualisasi

        cv2.imshow("LP Detection (YOLOv8)", annotated)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC buat keluar
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
