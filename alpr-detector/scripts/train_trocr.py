# alpr-detector/scripts/train_trocr.py
from pathlib import Path
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import editdistance
import warnings

# ==== Config ====
DATA_ROOT = Path("./alpr-detector/ocr_data")
MODEL_NAME = "microsoft/trocr-small-printed"
OUT_DIR = Path("./runs/ocr_trocr")
EPOCHS = 5
BATCH_SIZE = 8
LR = 5e-5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 32
# ===============

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)

def read_label_file(label_file: Path):
    items = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f.read().strip().splitlines():
            fname, label = line.split("\t", 1)
            items.append((fname.strip(), label.strip()))
    return items

class PlateOCRDataset(Dataset):
    def __init__(self, img_dir: Path, label_file: Path, processor: TrOCRProcessor):
        self.img_dir = img_dir
        self.samples = read_label_file(label_file)
        self.processor = processor

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        fname, text = self.samples[idx]
        image = Image.open(self.img_dir / fname).convert("RGB")
        # pixel_values sudah diseragamkan ukurannya oleh processor
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values[0]
        # labels DIBIARKAN variabel panjang, nanti dipad di collator
        labels = self.processor.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=MAX_LEN
        ).input_ids[0]
        return {"pixel_values": pixel_values, "labels": labels}

def compute_metrics(eval_preds):
    pred_ids, label_ids = eval_preds
    processor = compute_metrics.processor

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    # ubah -100 → pad_token sebelum decode labels
    label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)
    labels_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    import editdistance
    char_accs, word_accs, cers, wers = [], [], [], []
    for gt, pr in zip(labels_str, pred_str):
        ca = (sum(1 for i,(a,b) in enumerate(zip(gt,pr)) if a==b) / max(1,len(gt)))
        wa = float(gt == pr)
        ce = editdistance.eval(list(gt), list(pr)) / max(1,len(gt))
        we = editdistance.eval(gt.split(), pr.split()) / max(1,len(gt.split()))
        char_accs.append(ca); word_accs.append(wa); cers.append(ce); wers.append(we)
    return {"char_acc": float(np.mean(char_accs)),
            "word_acc": float(np.mean(word_accs)),
            "cer": float(np.mean(cers)),
            "wer": float(np.mean(wers))}

# ==== CUSTOM COLLATOR (kunci perbaikan) ====
def image_seq_collator(features):
    """
    features: list of dicts {"pixel_values": (C,H,W), "labels": (L,)}
    - Stack pixel_values (semua sama ukuran)
    - Pad labels ke panjang max dengan -100
    """
    pixel_values = torch.stack([f["pixel_values"] for f in features], dim=0)
    labels_list = [f["labels"] for f in features]
    max_len = max(l.size(0) for l in labels_list)
    labels = torch.full((len(labels_list), max_len), -100, dtype=torch.long)
    for i, l in enumerate(labels_list):
        labels[i, :l.size(0)] = l
    return {"pixel_values": pixel_values, "labels": labels}
# ===========================================

def main():
    # processor (use_fast kalau tersedia)
    # Gunakan XLMRobertaTokenizer sesuai dengan checkpoint model
    from transformers import ViTImageProcessor, XLMRobertaTokenizer # <-- Perubahan 1

    # Muat pemroses gambar (dulu namanya FeatureExtractor)
    image_processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    # Muat tokenizer dengan kelas yang benar
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME) # <-- Perubahan 2
    # Gabungkan menjadi satu TrOCRProcessor
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(DEVICE)
    # konfigurasi decoding
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = MAX_LEN
    # model.config.early_stopping = True

    # dataset
    train_ds = PlateOCRDataset(DATA_ROOT/"train", DATA_ROOT/"labels_train.txt", processor)
    val_ds   = PlateOCRDataset(DATA_ROOT/"val",   DATA_ROOT/"labels_val.txt",   processor)
    test_ds  = PlateOCRDataset(DATA_ROOT/"test",  DATA_ROOT/"labels_test.txt",  processor)

    compute_metrics.processor = processor

    # TrainingArguments: gunakan eval_strategy baru bila ada, fallback ke evaluation_strategy utk versi lama
    kwargs = dict(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        save_total_limit=2,
        label_names=["labels"],
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
    )
    try:
        args = Seq2SeqTrainingArguments(eval_strategy="epoch", **kwargs)
    except TypeError:
        warnings.warn("Using legacy 'evaluation_strategy' for older transformers.")
        args = Seq2SeqTrainingArguments(evaluation_strategy="epoch", **kwargs)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=image_seq_collator,   # <-- pakai collator custom
        tokenizer=processor,                # supaya trainer tahu cara decode saat generate
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # evaluasi test
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    print("TEST (TrOCR):", test_metrics)
    trainer.save_model(str(OUT_DIR / "best"))

if __name__ == "__main__":
    main()
