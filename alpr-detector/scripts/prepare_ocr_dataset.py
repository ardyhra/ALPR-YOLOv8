# scripts/prepare_ocr_dataset.py
import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import re

# ==== EDIT sesuai lokasi dataset ====
PLATE_TEXT_DIR = Path("./alpr-detector/IndonesianPlate/plate_text_dataset/dataset")
LABEL_CSV = Path("./alpr-detector/IndonesianPlate/plate_text_dataset/label.csv")
OUT_DIR = Path("./alpr-detector/ocr_data")
# ====================================

SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1

# Normalisasi label: uppercase, trim spaces berlebih
def normalize_label(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)   # satukan spasi berlebih
    s = s.upper()
    return s

def main():
    assert PLATE_TEXT_DIR.exists(), f"Gambar crop plat tidak ada: {PLATE_TEXT_DIR}"
    assert LABEL_CSV.exists(), f"label.csv tidak ada: {LABEL_CSV}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sp in ["train", "val", "test"]:
        (OUT_DIR/sp).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(LABEL_CSV)
    assert {"Filename", "Label"}.issubset(set(df.columns)), "CSV harus punya kolom Filename, Label"

    df["Label"] = df["Label"].astype(str).apply(normalize_label)
    df["src"] = df["Filename"].apply(lambda x: str((PLATE_TEXT_DIR / x).resolve()))
    # filter hanya file yang ada
    df = df[df["src"].apply(lambda p: Path(p).exists())].copy()

    # split
    train_df, tmp_df = train_test_split(df, test_size=(1-TRAIN_RATIO), random_state=SEED, stratify=None)
    val_df, test_df = train_test_split(tmp_df, test_size=TEST_RATIO/(VAL_RATIO+TEST_RATIO), random_state=SEED, stratify=None)

    def copy_and_write(split_df, split_name):
        label_lines = []
        for _, row in split_df.iterrows():
            src = Path(row["src"])
            dst = OUT_DIR / split_name / src.name
            shutil.copy2(src, dst)
            label_lines.append(f"{dst.name}\t{row['Label']}")
        with open(OUT_DIR / f"labels_{split_name}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))
        print(f"{split_name}: {len(split_df)} samples")

    copy_and_write(train_df, "train")
    copy_and_write(val_df, "val")
    copy_and_write(test_df, "test")

    # simpan charset (A-Z, 0-9 dan spasi)
    charset = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
    with open(OUT_DIR / "charset.txt", "w", encoding="utf-8") as f:
        f.write("".join(charset))
    print(f"Done. Labels & charset disimpan ke {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
