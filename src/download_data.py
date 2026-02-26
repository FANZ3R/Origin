import os
from roboflow import Roboflow
from pathlib import Path
from dotenv import load_dotenv


load_dotenv(Path(__file__).parent.parent / ".env")
API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY:
    raise ValueError("ROBOFLOW_API_KEY not found in .env file")

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(SAVE_DIR, exist_ok=True)

rf = Roboflow(api_key=API_KEY)

# Dataset 1: Drywall
print("Downloading Dataset 1: Drywall-Join-Detect...")
project1 = rf.workspace("objectdetect-pu6rn").project("drywall-join-detect")
dataset1 = project1.version(1).download(
    model_format="coco",
    location=os.path.join(SAVE_DIR, "drywall_taping")
)
print("Dataset 1 done.")

# Dataset 2: Cracks 
print("\nDownloading Dataset 2: Cracks...")
project2 = rf.workspace("burz4ms-workspace").project("cracks-3ii36-anowy")
dataset2 = project2.version(1).download(
    model_format="coco",
    location=os.path.join(SAVE_DIR, "cracks")
)
print("Dataset 2 done.")

print(f"\nAll done. Saved to: {os.path.abspath(SAVE_DIR)}")