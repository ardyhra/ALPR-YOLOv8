# scripts/infer_ocr.py
from pathlib import Path
import argparse
import torch
from PIL import Image
import cv2
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def infer_crnn(img_path, weights="./runs/ocr_crnn/crnn_best.pt"):
    from train_crnn_ctc import CRNN, resize_pad, IDX, IMG_H, IMG_W, BLANK_ID, indices_to_text
    from train_crnn_ctc import ctc_decode
    model = CRNN().to(DEVICE)
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.eval()

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = resize_pad(img)
    img = (img.astype(np.float32)/255.0 - 0.5)/0.5
    img = np.expand_dims(img, axis=0)  # (1,H,W)
    x = torch.tensor(img).unsqueeze(0).to(DEVICE)  # (1,1,H,W)
    with torch.no_grad():
        logits = model(x)
        text = ctc_decode(logits, blank=BLANK_ID)[0]
    return text

def infer_trocr(img_path, model_dir="./runs/ocr_trocr/best"):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    processor = TrOCRProcessor.from_pretrained(model_dir)
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
    args = ap.parse_args()

    if args.model=="crnn":
        print("CRNN:", infer_crnn(args.img))
    else:
        print("TrOCR:", infer_trocr(args.img))
