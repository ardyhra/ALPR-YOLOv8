# alpr-detector/scripts/infer_ocr.py
from pathlib import Path
import argparse
import torch
from PIL import Image
import cv2
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== CRNN ====
def infer_crnn(img_path, weights="./runs/ocr_crnn/crnn_best.pt"):
    # Hanya impor simbol yang memang ada
    from train_crnn_ctc import CRNN, resize_pad, IMG_H, IMG_W, BLANK_ID, ctc_decode

    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"CRNN weights tidak ditemukan: {weights}")

    # load model
    model = CRNN().to(DEVICE)
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.eval()

    # load & preprocess image (konsisten dengan training)
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan/korup: {img_path}")

    img = resize_pad(img)                          # (H,W) -> 32xW
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5                        # normalize
    img = np.expand_dims(img, axis=0)              # (1,H,W)
    x = torch.tensor(img).unsqueeze(0).to(DEVICE)  # (1,1,H,W)

    # forward + greedy decode
    with torch.no_grad():
        logits = model(x)                          # (B,T,C)
        text = ctc_decode(logits, blank=BLANK_ID)[0]
    return text

# ==== TrOCR ====
def infer_trocr(img_path, model_dir="./runs/ocr_trocr/best"):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor, XLMRobertaTokenizer

    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Folder model TrOCR tidak ditemukan: {model_dir}")

    # coba load processor tersimpan; fallback ke base model
    try:
        processor = TrOCRProcessor.from_pretrained(model_dir)
    except Exception:
        image_processor = ViTImageProcessor.from_pretrained("microsoft/trocr-small-printed")
        tokenizer = XLMRobertaTokenizer.from_pretrained("microsoft/trocr-small-printed")
        processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(DEVICE)
    model.eval()

    img = Image.open(img_path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        ids = model.generate(pixel_values, max_length=32)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return text

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=str, required=True)
    ap.add_argument("--model", type=str, choices=["crnn","trocr"], default="trocr")
    ap.add_argument("--crnn_weights", type=str, default="./runs/ocr_crnn/crnn_best.pt")
    ap.add_argument("--trocr_dir", type=str, default="./runs/ocr_trocr/best")
    args = ap.parse_args()

    if args.model == "crnn":
        print("CRNN:", infer_crnn(args.img, weights=args.crnn_weights))
    else:
        print("TrOCR:", infer_trocr(args.img, model_dir=args.trocr_dir))
