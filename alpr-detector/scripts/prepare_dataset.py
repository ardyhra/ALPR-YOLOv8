# scripts/prepare_dataset.py
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ==== EDIT SESUAI LOKASI DATASETMU ====
DATASET_ROOT = Path(r"./alpr-detector/IndonesianPlate/plate_detection_dataset")  # folder berisi images/ & annotations/
COCO_JSON = DATASET_ROOT / "annotations" / "annotations.json"
IMG_DIR = DATASET_ROOT / "images"
WORK_DIR = Path(r"./alpr-detector/yolo_data")  # hasil konversi YOLO + split
# ======================================

SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
RANDOM_SEED = 42

def coco_bbox_to_yolo(bbox, img_w, img_h):
    # COCO: [x_min, y_min, width, height]
    x, y, w, h = bbox
    x_c = x + w / 2.0
    y_c = y + h / 2.0
    return [x_c / img_w, y_c / img_h, w / img_w, h / img_h]

def ensure_dirs():
    for split in ["train", "val", "test"]:
        (WORK_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (WORK_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def main():
    assert COCO_JSON.exists(), f"COCO JSON tidak ditemukan: {COCO_JSON}"
    assert IMG_DIR.exists(), f"Folder images tidak ditemukan: {IMG_DIR}"

    print(f"Load COCO: {COCO_JSON}")
    with open(COCO_JSON, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    annos_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        # filter kategori bila perlu (dataset ini umumnya 1 kategori: license_plate)
        annos_by_image[ann["image_id"]].append(ann)

    img_ids = list(images.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(img_ids)

    n = len(img_ids)
    n_train = int(SPLITS["train"] * n)
    n_val = int(SPLITS["val"] * n)
    train_ids = set(img_ids[:n_train])
    val_ids = set(img_ids[n_train:n_train + n_val])
    test_ids = set(img_ids[n_train + n_val:])

    print(f"Total images: {n} | train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    ensure_dirs()

    def process_split(id_set, split):
        pbar = tqdm(id_set, desc=f"Processing {split}")
        for img_id in pbar:
            img_meta = images[img_id]
            file_name = img_meta["file_name"]
            width, height = int(img_meta["width"]), int(img_meta["height"])

            src_img = IMG_DIR / file_name
            if not src_img.exists():
                # fallback cari berdasarkan nama file saja
                candidates = list(IMG_DIR.rglob(Path(file_name).name))
                if len(candidates) == 0:
                    print(f"[WARNING] Gambar hilang: {file_name}")
                    continue
                src_img = candidates[0]

            dst_img = WORK_DIR / "images" / split / Path(file_name).name
            shutil.copy2(src_img, dst_img)

            # tulis label YOLO
            label_lines = []
            for ann in annos_by_image.get(img_id, []):
                # Abaikan anotasi yang tidak punya bbox valid
                bbox = ann.get("bbox", None)
                if bbox is None or len(bbox) != 4:
                    continue

                # clamp bbox agar tidak keluar gambar
                x, y, w, h = bbox
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w = max(1, min(w, width - x))
                h = max(1, min(h, height - y))

                x_c, y_c, w_n, h_n = coco_bbox_to_yolo([x, y, w, h], width, height)
                # class id YOLO mulai dari 0 -> kita set 0 untuk "license_plate"
                label_lines.append("0 " + " ".join(f"{v:.6f}" for v in [x_c, y_c, w_n, h_n]))

            label_path = WORK_DIR / "labels" / split / (Path(file_name).stem + ".txt")
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines))

    process_split(train_ids, "train")
    process_split(val_ids, "val")
    process_split(test_ids, "test")

    # tulis data.yaml untuk Ultralytics
    data_yaml = f"""path: {WORK_DIR.resolve().as_posix()}
train: images/train
val: images/val
test: images/test
names:
  0: license_plate
"""
    with open(WORK_DIR / "data.yaml", "w", encoding="utf-8") as f:
        f.write(data_yaml)

    print(f"Selesai. data.yaml disimpan di: {(WORK_DIR / 'data.yaml').resolve()}")

if __name__ == "__main__":
    main()
