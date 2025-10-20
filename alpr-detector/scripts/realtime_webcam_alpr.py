# alpr-detector/scripts/run_webcam_alpr.py
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor, XLMRobertaTokenizer
# Jalankan dengan python alpr-detector/scripts/realtime_webcam_alpr.py --yolo_weights "runs/detect/lp_yolov8n/weights/best.pt" --trocr_dir "runs/ocr_trocr/best" --show --use_fp16

# -----------------------------
# Utils
# -----------------------------
def iou_xyxy(a, b):
    # a,b: [x1,y1,x2,y2]
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_x1 = max(xa1, xb1); inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2); inter_y2 = min(ya2, yb2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    return inter / max(1.0, (area_a + area_b - inter))

def safe_crop_with_pad(img, box_xyxy, pad_ratio=0.10):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_ratio), int(bh * pad_ratio)
    x1p, y1p = max(0, x1 - px), max(0, y1 - py)
    x2p, y2p = min(w, x2 + px), min(h, y2 + py)
    if x2p <= x1p or y2p <= y1p:
        return None
    return img[y1p:y2p, x1p:x2p], (x1p, y1p, x2p, y2p)

class OcrCache:
    """Cache OCR per bbox sederhana untuk hemat komputasi."""
    def __init__(self, ttl_frames=15, iou_thresh=0.5):
        self.items = []  # list of dict: {box, text, last_frame}
        self.ttl = ttl_frames
        self.iou_thresh = iou_thresh

    def get(self, box, frame_idx):
        # kembalikan teks jika ada cache dengan IoU tinggi
        best = None; best_iou = 0.0
        for it in self.items:
            iou = iou_xyxy(it["box"], box)
            if iou > self.iou_thresh and iou > best_iou:
                best = it; best_iou = iou
        if best is not None:
            best["last_frame"] = frame_idx
            return best["text"]
        return None

    def put(self, box, text, frame_idx):
        self.items.append({"box": box, "text": text, "last_frame": frame_idx})
        # bersihkan yang kedaluwarsa
        self.items = [it for it in self.items if frame_idx - it["last_frame"] <= self.ttl]

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo_weights", type=str, default="./runs/detect/lp_yolov8n/weights/best.pt",
                    help="Path YOLOv8 weights (best.pt)")
    ap.add_argument("--trocr_dir", type=str, default="./runs/ocr_trocr/best",
                    help="Folder berisi model & processor TrOCR hasil fine-tune")
    ap.add_argument("--model_name_fallback", type=str, default="microsoft/trocr-small-printed",
                    help="Checkpoint TrOCR base untuk fallback jika processor tidak ada di trocr_dir")
    ap.add_argument("--src", type=str, default="0",
                    help="Sumber video: index webcam (mis. '0') atau path file video")
    ap.add_argument("--conf", type=float, default=0.35, help="Confidence threshold YOLO")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU YOLO")
    ap.add_argument("--ocr_every", type=int, default=3, help="Jalankan OCR setiap N frame untuk semua deteksi")
    ap.add_argument("--show", action="store_true", help="Tampilkan jendela video")
    ap.add_argument("--save", type=str, default="", help="Path output video (mp4). Kosongkan jika tidak disimpan")
    ap.add_argument("--width", type=int, default=1280, help="Lebar frame capture (opsional)")
    ap.add_argument("--height", type=int, default=720, help="Tinggi frame capture (opsional)")
    ap.add_argument("--use_fp16", action="store_true", help="Coba pakai FP16 (GPU) untuk percepatan")
    args = ap.parse_args()

    # --- Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # --- Load YOLO
    yolo = YOLO(args.yolo_weights)
    # ultralytics otomatis pilih device; kita set via environment dengan .predict(..., device=0/CPU)
    yolo_overrides = dict(conf=args.conf, iou=args.iou, verbose=False)

    # --- Load TrOCR (model + processor)
    trocr_dir = Path(args.trocr_dir)
    if not trocr_dir.exists():
        raise FileNotFoundError(f"Folder TrOCR tidak ditemukan: {trocr_dir.resolve()}")

    try:
        processor = TrOCRProcessor.from_pretrained(trocr_dir)
        print("[INFO] Processor loaded from fine-tuned dir.")
    except Exception as e:
        print("[WARN] Gagal load processor dari fine-tuned dir:", e)
        print("[INFO] Fallback ke base model processor:", args.model_name_fallback)
        image_processor = ViTImageProcessor.from_pretrained(args.model_name_fallback)
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name_fallback)
        processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

    trocr = VisionEncoderDecoderModel.from_pretrained(trocr_dir).to(device)
    trocr.eval()
    if args.use_fp16 and device == "cuda":
        try:
            trocr = trocr.half()
            print("[INFO] TrOCR in FP16.")
        except Exception:
            print("[WARN] FP16 untuk TrOCR gagal; lanjut FP32.")

    # --- Video source
    src = 0 if args.src == "0" else args.src
    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # --- Video writer (opsional)
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, 25.0, (int(cap.get(3)), int(cap.get(4))))
        print(f"[INFO] Saving to: {args.save}")

    # --- OCR cache
    cache = OcrCache(ttl_frames=30, iou_thresh=0.5)

    frame_idx = 0
    fps_hist = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[INFO] Video selesai / webcam tidak tersedia.")
                break
            t0 = time.time()

            # Deteksi YOLO (langsung dari numpy BGR)
            # Hasil: list [ultralytics.engine.results.Results]
            results = yolo.predict(
                source=frame,
                device=0 if device == "cuda" else "cpu",
                **yolo_overrides
            )
            res = results[0]

            # Extract boxes
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0,4))
            scores = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.empty((0,))
            texts = []

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)
                # Ambil dari cache kalau ada
                text_cached = cache.get([x1,y1,x2,y2], frame_idx)
                need_ocr = (frame_idx % args.ocr_every == 0) or (text_cached is None)

                if (not need_ocr) and (text_cached is not None):
                    text = text_cached
                else:
                    crop, _ = safe_crop_with_pad(frame, [x1, y1, x2, y2], pad_ratio=0.12)
                    if crop is None or crop.size == 0:
                        text = ""
                    else:
                        # OCR dengan TrOCR
                        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        with torch.inference_mode():
                            pixel_values = processor(images=img_rgb, return_tensors="pt").pixel_values.to(device)
                            if args.use_fp16 and device == "cuda":
                                pixel_values = pixel_values.half()
                            out_ids = trocr.generate(pixel_values, max_length=32)
                            text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
                    cache.put([x1,y1,x2,y2], text, frame_idx)

                texts.append(text)

                # Visualisasi
                color = (0, 180, 0) if len(text.strip()) > 0 else (0, 0, 200)
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                label = f"{text.strip() if text else '...'}  ({scores[i]:.2f})"
                cv2.rectangle(frame, (x1, max(0,y1-26)), (x1 + 8*len(label), y1), color, -1)
                cv2.putText(frame, label, (x1+4, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

            # FPS
            dt = time.time() - t0
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_hist.append(fps)
            if len(fps_hist) > 30:
                fps_hist.pop(0)
            fps_avg = sum(fps_hist) / max(1, len(fps_hist))
            cv2.putText(frame, f"FPS: {fps_avg:.1f} | Det:{len(boxes)}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {fps_avg:.1f} | Det:{len(boxes)}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)

            if writer is not None:
                writer.write(frame)

            if args.show:
                cv2.imshow("ALPR Realtime - YOLOv8 + TrOCR", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    out_img = f"frame_{frame_idx}.jpg"
                    cv2.imwrite(out_img, frame)
                    print(f"[INFO] Frame disimpan: {out_img}")

            frame_idx += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Selesai.")

if __name__ == "__main__":
    main()
