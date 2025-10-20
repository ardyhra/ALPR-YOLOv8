# scripts/train_crnn_ctc.py
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import editdistance

# ==== Config ====
DATA_ROOT = Path("./alpr-detector/ocr_data")
IMG_H, IMG_W = 32, 128
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = Path("./runs/ocr_crnn")
# ===============

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(SAVE_DIR, exist_ok=True)

def load_charset(path):
    with open(path, "r", encoding="utf-8") as f:
        s = f.read().strip()
    # indeks: 0..N-1 utk char. CTC butuh blank_id terpisah (letakkan di akhir)
    charset = list(s)
    return charset

CHARSET = load_charset(DATA_ROOT / "charset.txt")
BLANK_ID = len(CHARSET)  # CTC blank di index terakhir
VOCAB_SIZE = len(CHARSET) + 1

char2idx = {c:i for i,c in enumerate(CHARSET)}
idx2char = {i:c for c,i in char2idx.items()}

def label_to_indices(text):
    return [char2idx[c] for c in text if c in char2idx]

def indices_to_text(indices):
    return "".join(idx2char[i] for i in indices)

def read_label_file(label_file):
    items = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f.read().strip().splitlines():
            fname, label = line.split("\t", 1)
            items.append((fname.strip(), label.strip()))
    return items

def resize_pad(img, target_h=IMG_H, target_w=IMG_W):
    h, w = img.shape[:2]
    new_h = target_h
    new_w = int(w * (new_h / h))
    if new_w > target_w:
        new_w = target_w
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w), dtype=img.dtype)
    canvas[:, :new_w] = img
    return canvas

class OCRDataset(Dataset):
    def __init__(self, img_dir: Path, label_file: Path, augment=False):
        self.img_dir = img_dir
        self.items = read_label_file(label_file)
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, label = self.items[idx]
        img_path = self.img_dir / fname
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        img = resize_pad(img)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # normalize
        img = np.expand_dims(img, axis=0)  # (1,H,W)
        label_idx = label_to_indices(label)
        return torch.tensor(img), torch.tensor(label_idx, dtype=torch.long), label

def collate_fn(batch):
    imgs, labels_idx, raw_labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    # concatenate labels untuk CTC
    label_lens = torch.tensor([len(x) for x in labels_idx], dtype=torch.long)
    concat_labels = torch.cat(labels_idx, dim=0) if len(labels_idx[0])>0 else torch.tensor([], dtype=torch.long)
    return imgs, concat_labels, label_lens, raw_labels

class CRNN(nn.Module):
    def __init__(self, img_h=IMG_H, n_channels=1, n_classes=VOCAB_SIZE):
        super().__init__()
        # CNN feature extractor (sederhana)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2),   # 64 x 16 x 64
            nn.Conv2d(64,128,3,1,1), nn.ReLU(), nn.MaxPool2d(2,2),              # 128 x 8 x 32
            nn.Conv2d(128,256,3,1,1), nn.ReLU(),
            nn.Conv2d(256,256,3,1,1), nn.ReLU(), nn.MaxPool2d((2,2),(2,1), (0,1)), # 256 x 4 x 33
            nn.Conv2d(256,512,3,1,1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512,512,3,1,1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2,2),(2,1),(0,1)), # 512 x 2 x 34
            nn.Conv2d(512,512,2,1,0), nn.ReLU() # 512 x 1 x 33
        )
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        # x: (B,1,H,W)
        feats = self.cnn(x)        # (B,512,1,W')
        feats = feats.squeeze(2)   # (B,512,W')
        feats = feats.permute(0,2,1) # (B,W',512) -> time major along width
        seq, _ = self.rnn(feats)   # (B,W',512)
        logits = self.fc(seq)      # (B,W',n_classes)
        return logits

def ctc_decode(logits, blank=BLANK_ID):
    # greedy decode
    probs = logits.softmax(dim=-1)
    idx = probs.argmax(dim=-1)  # (B,T)
    results = []
    for seq in idx:
        prev = blank
        out = []
        for i in seq.cpu().numpy().tolist():
            if i != blank and i != prev:
                if i < BLANK_ID:
                    out.append(i)
            prev = i
        results.append(indices_to_text(out))
    return results

def cer(ref, hyp):
    return editdistance.eval(list(ref), list(hyp)) / max(1,len(ref))

def wer(ref, hyp):
    return editdistance.eval(ref.split(), hyp.split()) / max(1,len(ref.split()))

def evaluate(model, loader, criterion):
    model.eval()
    losses, chars_acc, words_acc, cers, wers = [], [], [], [], []
    with torch.no_grad():
        for imgs, concat_labels, label_lens, raw_labels in loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)  # (B,T,C)
            B, T, C = logits.shape
            log_probs = logits.log_softmax(2).permute(1,0,2)  # (T,B,C) for CTCLoss

            input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long)
            input_lengths = input_lengths.to(DEVICE)

            concat_labels = concat_labels.to(DEVICE)
            label_lens = label_lens.to(DEVICE)

            loss = criterion(log_probs, concat_labels, input_lengths, label_lens)
            losses.append(loss.item())

            preds = ctc_decode(logits, blank=BLANK_ID)
            # metrics
            for gt, pr in zip(raw_labels, preds):
                ca = (sum(1 for i,(a,b) in enumerate(zip(gt,pr)) if a==b) / max(1,len(gt)))
                wa = float(gt == pr)
                chars_acc.append(ca)
                words_acc.append(wa)
                cers.append(cer(gt, pr))
                wers.append(wer(gt, pr))

    return {
        "loss": np.mean(losses),
        "char_acc": float(np.mean(chars_acc)),
        "word_acc": float(np.mean(words_acc)),
        "cer": float(np.mean(cers)),
        "wer": float(np.mean(wers)),
    }

def main():
    train_set = OCRDataset(DATA_ROOT/"train", DATA_ROOT/"labels_train.txt")
    val_set   = OCRDataset(DATA_ROOT/"val",   DATA_ROOT/"labels_val.txt")
    test_set  = OCRDataset(DATA_ROOT/"test",  DATA_ROOT/"labels_test.txt")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)
    test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)

    model = CRNN().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)

    best_val = 1e9
    for epoch in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        losses = []
        for imgs, concat_labels, label_lens, _ in pbar:
            imgs = imgs.to(DEVICE)

            logits = model(imgs)  # (B,T,C)
            B,T,C = logits.shape
            log_probs = logits.log_softmax(2).permute(1,0,2)  # (T,B,C)

            input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long).to(DEVICE)
            concat_labels = concat_labels.to(DEVICE)
            label_lens = label_lens.to(DEVICE)

            loss = criterion(log_probs, concat_labels, input_lengths, label_lens)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=np.mean(losses))

        metrics = evaluate(model, val_loader, criterion)
        print(f"[VAL] loss={metrics['loss']:.4f} char_acc={metrics['char_acc']:.3f} word_acc={metrics['word_acc']:.3f} cer={metrics['cer']:.3f} wer={metrics['wer']:.3f}")

        # simpan terbaik berdasar VAL CER
        if metrics["cer"] < best_val:
            best_val = metrics["cer"]
            torch.save(model.state_dict(), SAVE_DIR/"crnn_best.pt")
            print("✓ Model disimpan:", SAVE_DIR/"crnn_best.pt")

    # evaluasi test
    model.load_state_dict(torch.load(SAVE_DIR/"crnn_best.pt", map_location=DEVICE))
    test_metrics = evaluate(model, test_loader, criterion)
    print("[TEST] CRNN:", test_metrics)

if __name__ == "__main__":
    main()
