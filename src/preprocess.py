import os
import json
import glob
import random
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

SEED     = 42
RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
IMG_SIZE = 640
random.seed(SEED)
np.random.seed(SEED)

def make_dirs(base):
    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(base, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(base, split, "masks"),  exist_ok=True)

def bbox_to_mask(bbox, img_w, img_h):
    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], fill=255)
    return mask

def polygon_to_mask(segmentation, img_w, img_h):
    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)
    for seg in segmentation:
        if len(seg) >= 6:
            pts = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
            draw.polygon(pts, fill=255)
    return mask

def process_and_save(raw_path, out_path, prompt_tag, use_polygons,
                     extra_test_from_valid=False, extra_test_from_train=False):
    make_dirs(out_path)
    counts = {"train": 0, "valid": 0, "test": 0}

    ann_files = glob.glob(
        os.path.join(raw_path, "**", "_annotations.coco.json"),
        recursive=True
    )

    # if we need to resplit, collect filenames first (not images)
    valid_ids = []
    if extra_test_from_valid:
        for ann_file in ann_files:
            split = os.path.basename(os.path.dirname(ann_file))
            if split == "valid":
                with open(ann_file) as f:
                    coco = json.load(f)
                valid_ids = [img["id"] for img in coco["images"]]
                random.shuffle(valid_ids)
                break

    train_ids = []
    if extra_test_from_train:
        for ann_file in ann_files:
            split = os.path.basename(os.path.dirname(ann_file))
            if split == "train":
                with open(ann_file) as f:
                    coco = json.load(f)
                train_ids = [img["id"] for img in coco["images"]]
                random.shuffle(train_ids)
                break

    # determine test ids from valid
    n_test_from_valid = max(50, int(0.2 * len(valid_ids))) if valid_ids else 0
    test_ids_set = set(valid_ids[:n_test_from_valid])

    # determine test ids from train
    n_test_from_train = max(50, int(0.1 * len(train_ids))) if train_ids else 0
    test_train_ids_set = set(train_ids[:n_test_from_train])

    for ann_file in ann_files:
        split   = os.path.basename(os.path.dirname(ann_file))
        img_dir = os.path.dirname(ann_file)

        with open(ann_file) as f:
            coco = json.load(f)

        id_to_anns = {}
        for ann in coco["annotations"]:
            id_to_anns.setdefault(ann["image_id"], []).append(ann)

        for img_info in tqdm(coco["images"], desc=f"{prompt_tag} [{split}]"):
            img_id  = img_info["id"]
            img_w   = img_info["width"]
            img_h   = img_info["height"]
            fname   = img_info["file_name"]
            img_path = os.path.join(img_dir, fname)

            if not os.path.exists(img_path):
                continue

            anns = id_to_anns.get(img_id, [])
            if not anns:
                continue

            # determine actual split
            actual_split = split
            if img_id in test_ids_set:
                actual_split = "test"
            elif img_id in test_train_ids_set:
                actual_split = "test"

            # build mask
            combined = np.zeros((img_h, img_w), dtype=np.uint8)
            for ann in anns:
                if use_polygons and ann.get("segmentation"):
                    m = polygon_to_mask(ann["segmentation"], img_w, img_h)
                else:
                    m = bbox_to_mask(ann["bbox"], img_w, img_h)
                combined = np.maximum(combined, np.array(m))

            mask = Image.fromarray(combined)
            img  = Image.open(img_path).convert("RGB")

            if img.size != (IMG_SIZE, IMG_SIZE):
                img  = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)

            stem     = os.path.splitext(fname)[0]
            img_out  = os.path.join(out_path, actual_split, "images", f"{stem}.jpg")
            mask_out = os.path.join(out_path, actual_split, "masks", f"{stem}__{prompt_tag}.png")

            img.save(img_out, quality=95)
            mask.save(mask_out)

            # free memory immediately
            del img, mask, combined
            counts[actual_split] += 1

    return counts



def rebalance_cracks(proc_path, train_ratio=0.80, valid_ratio=0.10):
    """
    Redistribute all processed crack images into 80/10/10 split.
    Works by moving files between split folders.
    """
    import shutil

    # collect all image stems from all splits
    all_stems = []
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(proc_path, split, "images")
        if os.path.exists(img_dir):
            for f in os.listdir(img_dir):
                if f.endswith(".jpg"):
                    all_stems.append((split, os.path.splitext(f)[0]))

    random.shuffle(all_stems)
    total   = len(all_stems)
    n_train = int(total * train_ratio)
    n_valid = int(total * valid_ratio)

    new_splits = (
        [("train", s) for _, s in all_stems[:n_train]] +
        [("valid", s) for _, s in all_stems[n_train:n_train + n_valid]] +
        [("test",  s) for _, s in all_stems[n_train + n_valid:]]
    )

    # rebuild directories
    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(proc_path, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(proc_path, split, "masks"),  exist_ok=True)

    # move files to correct split
    for new_split, stem in tqdm(new_splits, desc="Rebalancing cracks"):
        for old_split in ["train", "valid", "test"]:
            src_img  = os.path.join(proc_path, old_split, "images", f"{stem}.jpg")
            src_mask = os.path.join(proc_path, old_split, "masks",  f"{stem}__segment_crack.png")
            if os.path.exists(src_img):
                dst_img  = os.path.join(proc_path, new_split, "images", f"{stem}.jpg")
                dst_mask = os.path.join(proc_path, new_split, "masks",  f"{stem}__segment_crack.png")
                if src_img != dst_img:
                    shutil.move(src_img,  dst_img)
                    shutil.move(src_mask, dst_mask)
                break

    # count final
    counts = {}
    for split in ["train", "valid", "test"]:
        counts[split] = len(os.listdir(os.path.join(proc_path, split, "images")))
    return counts

if __name__ == "__main__":

    print("\nProcessing Dataset 1: Drywall Taping...")
    counts = process_and_save(
        raw_path             = os.path.join(RAW_DIR,  "drywall_taping"),
        out_path             = os.path.join(PROC_DIR, "drywall_taping"),
        prompt_tag           = "segment_taping_area",
        use_polygons         = False,
        extra_test_from_train= True
    )
    print(f"  Drywall → train:{counts['train']} valid:{counts['valid']} test:{counts['test']}")

    print("\nProcessing Dataset 2: Cracks...")
    counts = process_and_save(
        raw_path              = os.path.join(RAW_DIR,  "cracks"),
        out_path              = os.path.join(PROC_DIR, "cracks"),
        prompt_tag            = "segment_crack",
        use_polygons          = True,
        extra_test_from_valid = False,  # we'll rebalance manually below
        extra_test_from_train = False
    )
    print(f"  Cracks → train:{counts['train']} valid:{counts['valid']} test:{counts['test']}")

    print(f"\nDone. Output: {os.path.abspath(PROC_DIR)}")

    print("\nRebalancing Cracks to 80/10/10...")
    counts = rebalance_cracks(os.path.join(PROC_DIR, "cracks"))
    print(f"  Cracks rebalanced → train:{counts['train']} valid:{counts['valid']} test:{counts['test']}")