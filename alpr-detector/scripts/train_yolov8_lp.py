# scripts/train_yolov8_lp.py
from ultralytics import YOLO
import torch
from pathlib import Path

# === EDIT jika perlu ===
DATA_YAML = Path("./alpr-detector/yolo_data/data.yaml")
MODEL_WEIGHTS = "yolov8n.pt"   # alternatif: yolov8s.pt (lebih akurat, lebih berat)
EPOCHS = 100
IMGSZ = 640
BATCH = -1  # auto
PROJECT = "runs/detect"
RUN_NAME = "lp_yolov8n"
# =======================

def main():
    assert DATA_YAML.exists(), f"data.yaml tidak ditemukan: {DATA_YAML}"
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"PyTorch CUDA available: {torch.cuda.is_available()} | device={device}")

    model = YOLO(MODEL_WEIGHTS)  # load pretrained
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        project=PROJECT,
        name=RUN_NAME,
        optimizer="SGD",  # bisa "AdamW"
        workers=4,        # sesuaikan dengan CPU
        verbose=True
        # Catatan: YOLOv8 sudah punya augmentasi bawaan (mosaic, hsv, flip, dll)
        # Untuk kustom augmentasi lanjut, kita bisa tambah "hyp" file terpisah.
    )
    print("Training selesai.")
    print(f"Best weights: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    main()
