import os
import json
import glob
import matplotlib.pyplot as plt
from PIL import Image

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

def explore_dataset(name, path):
    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    print(f"{'='*50}")

    ann_files = glob.glob(os.path.join(path, "**", "_annotations.coco.json"), recursive=True)
    if not ann_files:
        print("No annotation file found")
        return

    for ann_file in ann_files:
        split = os.path.basename(os.path.dirname(ann_file))
        with open(ann_file) as f:
            coco = json.load(f)

        images     = coco["images"]
        anns       = coco["annotations"]
        categories = coco["categories"]

        print(f"\n  Split     : {split}")
        print(f"  Images    : {len(images)}")
        print(f"  Annotations: {len(anns)}")
        print(f"  Categories : {[c['name'] for c in categories]}")

        # check if segmentation polygons exist
        has_seg = any(len(a.get("segmentation", [])) > 0 for a in anns)
        print(f"  Has segmentation polygons: {has_seg}")

        if images:
            print(f"  Sample image size: {images[0].get('width')}x{images[0].get('height')}")

def visualize_sample(dataset_path, dataset_name, n=3):
    ann_files = glob.glob(os.path.join(dataset_path, "train", "_annotations.coco.json"))
    if not ann_files:
        print(f"No train annotations found for {dataset_name}")
        return

    with open(ann_files[0]) as f:
        coco = json.load(f)

    images  = coco["images"][:n]
    img_dir = os.path.dirname(ann_files[0])

    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 4))
    if len(images) == 1:
        axes = [axes]

    for ax, img_info in zip(axes, images):
        img_path = os.path.join(img_dir, img_info["file_name"])
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
            ax.set_title(f"{dataset_name}\n{img_info['file_name'][:25]}")
            ax.axis("off")
        else:
            ax.set_title("Image not found")

    plt.tight_layout()
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "data",
        f"sample_{dataset_name.replace(' ', '_')}.png"
    )
    plt.savefig(out_path)
    print(f"\n  Sample visualization saved to: {out_path}")
    plt.show()

if __name__ == "__main__":
    drywall_path = os.path.join(RAW_DIR, "drywall_taping")
    cracks_path  = os.path.join(RAW_DIR, "cracks")

    explore_dataset("Drywall Taping (Dataset 1)", drywall_path)
    explore_dataset("Cracks (Dataset 2)", cracks_path)

    visualize_sample(drywall_path, "Drywall Taping")
    visualize_sample(cracks_path, "Cracks")