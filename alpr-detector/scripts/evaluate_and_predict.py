# scripts/evaluate_and_predict.py
from ultralytics import YOLO
import torch
from pathlib import Path

# === EDIT jika perlu ===
DATA_YAML = Path("./alpr-detector/yolo_data/data.yaml")
WEIGHTS = Path("./runs/detect/lp_yolov8n2/weights/best.pt")  # ganti sesuai nama run
IMGSZ = 640
DEVICE = 0 if torch.cuda.is_available() else "cpu"
# =======================

def main():
    assert DATA_YAML.exists(), f"data.yaml tidak ditemukan: {DATA_YAML}"
    assert WEIGHTS.exists(), f"weights tidak ditemukan: {WEIGHTS}"

    model = YOLO(str(WEIGHTS))

    # 1) Evaluasi di VAL set (default). Untuk TEST set, beberapa versi Ultralytics dukung split='test'.
    print("==> Evaluasi di Validation set:")
    metrics_val = model.val(data=str(DATA_YAML), imgsz=IMGSZ, device=DEVICE)  # mAP50-95, dll
    print("Val metrics:", metrics_val.results_dict)

    # 2) Evaluasi di TEST set (jika versi-mu mendukung argumen split='test')
    try:
        print("==> Evaluasi di Test set:")
        metrics_test = model.val(data=str(DATA_YAML), imgsz=IMGSZ, device=DEVICE, split='test')
        print("Test metrics:", metrics_test.results_dict)
    except TypeError:
        print("[INFO] model.val(split='test') tidak didukung versi Ultralytics ini. "
              "Sebagai alternatif, jalankan prediksi di folder test untuk inspeksi kualitatif.")

    # 3) Prediksi & simpan visualisasi di folder runs/predict
    print("==> Prediksi contoh pada test images (hasil disimpan):")
    preds = model.predict(
        source=str(Path("./alpr-detector/yolo_data/images/test")),
        imgsz=IMGSZ,
        conf=0.25,
        device=DEVICE,
        save=True,     # simpan gambar beranotasi
        save_txt=True  # simpan prediksi .txt
    )
    print("Selesai. Cek folder runs/detect/predict* untuk visualisasi.")

if __name__ == "__main__":
    main()
