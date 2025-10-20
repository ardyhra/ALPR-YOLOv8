# scripts/offline_augment_yolo.py
import cv2
import glob
import os
from pathlib import Path
import random
from tqdm import tqdm
import albumentations as A

# === EDIT jika perlu ===
YOLO_ROOT = Path("./alpr-detector/yolo_data")
SPLIT = "train"  # hanya augment data train
N_AUG_PER_IMAGE = 1  # berapa versi augment per gambar
RANDOM_SEED = 123
# =======================

# Transformasi "real-world" (pilih secukupnya)
transform = A.Compose([
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        A.CLAHE(p=1.0),
        A.RandomGamma(gamma_limit=(60, 140), p=1.0),
    ], p=0.7),
    A.OneOf([
        A.MotionBlur(blur_limit=7, p=1.0),
        A.GaussianBlur(blur_limit=7, p=1.0),
        A.MedianBlur(blur_limit=5, p=1.0),
    ], p=0.3),
    A.OneOf([
        A.ISONoise(p=1.0),
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
    ], p=0.3),
    A.ToFloat(max_value=255.0)
],
    bbox_params=A.BboxParams(
        format='yolo',  # format YOLO: [x_center, y_center, w, h] (normalized)
        min_visibility=0.4,
        label_fields=['class_labels']
    )
)

def read_yolo_label(label_path):
    bboxes = []
    class_ids = []
    if not label_path.exists():
        return bboxes, class_ids
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f.read().strip().splitlines():
            if not line:
                continue
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = parts
            class_ids.append(int(cls))
            bboxes.append([float(x), float(y), float(w), float(h)])
    return bboxes, class_ids

def write_yolo_label(label_path, class_ids, bboxes):
    with open(label_path, "w", encoding="utf-8") as f:
        for cls, (x, y, w, h) in zip(class_ids, bboxes):
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def main():
    random.seed(RANDOM_SEED)
    img_dir = YOLO_ROOT / "images" / SPLIT
    lbl_dir = YOLO_ROOT / "labels" / SPLIT
    img_paths = sorted(glob.glob(str(img_dir / "*.*")))

    print(f"Augment {len(img_paths)} images dari {img_dir}")
    for img_path in tqdm(img_paths):
        img_path = Path(img_path)
        label_path = lbl_dir / (img_path.stem + ".txt")
        bboxes, class_ids = read_yolo_label(label_path)

        # skip gambar tanpa bbox (opsional)
        if len(bboxes) == 0:
            continue

        # baca image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for i in range(N_AUG_PER_IMAGE):
            aug = transform(image=image, bboxes=bboxes, class_labels=class_ids)
            aug_img = cv2.cvtColor(aug["image"], cv2.COLOR_RGB2BGR)
            aug_bboxes = aug["bboxes"]
            aug_classes = aug["class_labels"]

            new_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
            new_img_path = img_dir / new_name
            new_label_path = lbl_dir / (img_path.stem + f"_aug{i}.txt")

            cv2.imwrite(str(new_img_path), aug_img)
            write_yolo_label(new_label_path, aug_classes, aug_bboxes)

    print("Augmentasi offline selesai.")

if __name__ == "__main__":
    main()
