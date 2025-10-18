# scripts/evaluate_ocr.py
from pathlib import Path
import numpy as np
import torch
import editdistance
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

DATA_ROOT = Path("./alpr-detector/ocr_data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === CRNN ===
from train_crnn_ctc import CRNN, OCRDataset, collate_fn, BLANK_ID, indices_to_text, IMG_H, IMG_W, resize_pad

def cer(ref, hyp): return editdistance.eval(list(ref), list(hyp))/max(1,len(ref))
def wer(ref, hyp): return editdistance.eval(ref.split(), hyp.split())/max(1,len(ref.split()))

def eval_crnn(weights_path=Path("./runs/ocr_crnn/crnn_best.pt")):
    model = CRNN().to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    test_set = OCRDataset(DATA_ROOT/"test", DATA_ROOT/"labels_test.txt")
    loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=2, collate_fn=collate_fn)

    from train_crnn_ctc import nn
    criterion = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)

    # decode greedy yang sama dengan train_crnn_ctc
    from train_crnn_ctc import ctc_decode

    chars_acc, words_acc, cers, wers = [], [], [], []
    with torch.no_grad():
        for imgs, _, _, raw_labels in loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            preds = ctc_decode(logits, blank=BLANK_ID)
            for gt, pr in zip(raw_labels, preds):
                ca = (sum(1 for i,(a,b) in enumerate(zip(gt,pr)) if a==b) / max(1,len(gt)))
                wa = float(gt == pr)
                chars_acc.append(ca); words_acc.append(wa); cers.append(cer(gt,pr)); wers.append(wer(gt,pr))
    return {
        "char_acc": float(np.mean(chars_acc)),
        "word_acc": float(np.mean(words_acc)),
        "cer": float(np.mean(cers)),
        "wer": float(np.mean(wers)),
    }

def eval_trocr(model_dir=Path("./runs/ocr_trocr/best")):
    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(DEVICE)
    model.eval()

    items = []
    with open(DATA_ROOT/"labels_test.txt", "r", encoding="utf-8") as f:
        for line in f.read().strip().splitlines():
            fname, label = line.split("\t", 1)
            items.append((fname.strip(), label.strip()))

    chars_acc, words_acc, cers, wers = [], [], [], []
    for fname, gt in items:
        img = Image.open(DATA_ROOT/"test"/fname).convert("RGB")
        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(DEVICE)
        output_ids = model.generate(pixel_values, max_length=32)
        pr = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        ca = (sum(1 for i,(a,b) in enumerate(zip(gt,pr)) if a==b) / max(1,len(gt)))
        wa = float(gt == pr)
        chars_acc.append(ca); words_acc.append(wa); cers.append(cer(gt,pr)); wers.append(wer(gt,pr))
    return {
        "char_acc": float(np.mean(chars_acc)),
        "word_acc": float(np.mean(words_acc)),
        "cer": float(np.mean(cers)),
        "wer": float(np.mean(wers)),
    }

if __name__ == "__main__":
    crnn_metrics = eval_crnn()
    trocr_metrics = eval_trocr()
    print("TEST CRNN :", crnn_metrics)
    print("TEST TrOCR:", trocr_metrics)
