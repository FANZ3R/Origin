# Prompted Segmentation for Drywall QA

Text-conditioned binary segmentation model that produces pixel masks given an image
and a natural-language prompt. Supports two prompts:
- `"segment taping area"` — detects drywall seams and joint tape
- `"segment crack"` — detects wall cracks

---

## Approach

Fine-tuned **CLIPSeg** (`CIDAS/clipseg-rd64-refined`) on two construction QA datasets.
CLIPSeg combines a CLIP vision-language backbone with a lightweight convolutional
decoder, making it naturally suited for text-prompted segmentation.

Only the decoder was fine-tuned (1.1M / 150.7M parameters). The CLIP backbone was
frozen to preserve general vision-language alignment while adapting the segmentation
head to the target domains.

**Loss:** Equal-weight combination of Binary Cross Entropy and Dice Loss (50/50).
This combination handles class imbalance (crack pixels are rare) better than BCE alone.

---

## Model

| Property | Value |
|---|---|
| Base model | CIDAS/clipseg-rd64-refined |
| Total parameters | 150,747,746 |
| Trainable parameters | 1,127,009 |
| Frozen parameters | 149,620,737 |
| Checkpoint size | 603.2 MB |

---

## Datasets

### Dataset 1 — Drywall-Join-Detect
- Source: https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect
- Prompt: `"segment taping area"`
- Annotation type: Bounding boxes (converted to filled rectangular masks)
- Split: 936 train / 250 valid / 209 test

### Dataset 2 — Cracks
- Source: https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36
- Prompt: `"segment crack"`
- Annotation type: Polygon segmentation masks
- Split: 4275 train / 546 valid / 632 test

---

## Results

| Prompt | mIoU | Dice | N (test) |
|---|---|---|---|
| segment taping area | 0.5730 | 0.7194 | 209 |
| segment crack | 0.4592 | 0.6064 | 632 |
| **Average** | **0.5161** | **0.6629** | **841** |

---

## Training

| Hyperparameter | Value |
|---|---|
| Epochs | 15 |
| Best epoch | 8 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Batch size | 16 |
| Scheduler | CosineAnnealingLR |
| Loss | BCE + Dice (50/50) |
| Random seed | 42 |

---

## Runtime & Footprint

| Metric | Value |
|---|---|
| Training time | ~75 minutes (15 epochs, T4 GPU) |
| Avg inference time | 43.8 ms/image |
| Checkpoint size | 603.2 MB |
| GPU used | Tesla T4 (15.6 GB VRAM) |

---

## Output Masks

Prediction masks are single-channel PNG files, same spatial size as source image,
with values {0, 255}.

Filename format: `{image_stem}__{prompt_tag}.png`

Examples:
- `image001__segment_crack.png`
- `image001__segment_taping_area.png`

---

## Reproduction

**Seeds used:** Python `random.seed(42)`, `numpy.seed(42)`, `torch.manual_seed(42)`

### Local preprocessing (VSCode)
```bash
pip install -r requirements.txt
python src/download_data.py
python src/preprocess.py
```

### Training and inference (Google Colab, T4 GPU)
Open `colab/train_and_infer.ipynb` and run all cells top to bottom.
Mount Google Drive when prompted and ensure processed data is uploaded to
`/MyDrive/origin-drywall-qa/processed/`.

---

## Project Structure
```
origin-drywall-qa/
├── data/
│   ├── raw/                  # downloaded datasets
│   └── processed/            # binary masks + splits
├── src/
│   ├── download_data.py
│   ├── explore_data.py
│   └── preprocess.py
├── colab/
│   └── origin.ipynb   (also added link of this in the submission)
├── outputs/            (same added the link as well as the ouput is large for github)
│   ├── masks/                # prediction PNGs
│   └── visuals/              # orig | GT | pred comparisons
├── requirements.txt
└── README.md
```

---

## Known Limitations

- Drywall dataset has bounding box annotations only — rectangular masks are an
  approximation of the true taping area, which limits the upper bound of achievable
  mIoU for this prompt.
- Crack test set performance is constrained by the thin, irregular nature of cracks.
  Even specialized crack segmentation models report mIoU in the 0.65-0.85 range
  with purpose-built architectures trained on much larger datasets.
- CLIPSeg decoder resolution (64x64 upsampled to 640x640) limits fine boundary
  precision on hairline cracks.
```

---

## Report (paste into a PDF or Word doc)
```
PROMPTED SEGMENTATION FOR DRYWALL QA
Technical Report
─────────────────────────────────────────────────────────

1. GOAL
───────
Train a text-conditioned segmentation model that, given an image and a
natural-language prompt, produces a binary mask identifying the region of
interest. Two prompts are supported:
  • "segment taping area" — drywall seams and joint tape regions
  • "segment crack"       — wall and surface cracks

─────────────────────────────────────────────────────────

2. APPROACH & MODEL
───────────────────
Model: CLIPSeg (CIDAS/clipseg-rd64-refined)

CLIPSeg was selected because it natively accepts text prompts as conditioning
signals for segmentation — no external detector is needed. It combines a CLIP
ViT-B/16 vision-language backbone with a lightweight transformer decoder that
produces segmentation logits conditioned on the text embedding.

Fine-tuning strategy: Only the decoder was trained (1.1M parameters out of
150.7M total). The CLIP backbone was frozen to preserve general vision-language
alignment while adapting the decoder to the two target domains.

Loss function: Equal-weight sum of Binary Cross Entropy (BCE) and Dice Loss.
BCE alone tends to underweight minority-class pixels (cracks occupy a small
fraction of each image). Dice Loss directly optimizes the overlap metric,
complementing BCE for imbalanced segmentation tasks.

─────────────────────────────────────────────────────────

3. DATA
───────

Dataset 1 — Drywall-Join-Detect (taping area)
  Source      : roboflow.com/objectdetect-pu6rn/drywall-join-detect
  Images      : 1,395 total (640×640 px)
  Annotations : Bounding boxes only (converted to filled rectangular masks)
  Train/Valid/Test: 936 / 250 / 209

Dataset 2 — Cracks
  Source      : roboflow.com/fyp-ny1jt/cracks-3ii36
  Images      : 5,369 total (640×640 px)
  Annotations : Polygon segmentation masks (precise crack boundaries)
  Train/Valid/Test: 4,275 / 546 / 632

Preprocessing:
  • All images resized to 640×640
  • Bounding boxes converted to filled binary masks (0/255)
  • Polygons rendered to binary masks using PIL ImageDraw
  • Drywall test split carved from train (no original test split provided)
  • Cracks rebalanced from 96/4/0 to 78/10/12 (train/valid/test)
  • Augmentation on train: random horizontal flip, color jitter

─────────────────────────────────────────────────────────

4. TRAINING CONFIGURATION
──────────────────────────
  Optimizer     : AdamW (lr=1e-4, weight_decay=1e-4)
  Scheduler     : CosineAnnealingLR (T_max=15)
  Epochs        : 15 (best checkpoint at epoch 8)
  Batch size    : 16
  Loss          : BCE + Dice (50/50 weight)
  Hardware      : Tesla T4 GPU (15.6 GB VRAM)
  Seed          : 42 (Python, NumPy, PyTorch)

Validation mIoU progression (selected epochs):
  Epoch 01: 0.4170
  Epoch 04: 0.4666
  Epoch 07: 0.4812
  Epoch 08: 0.4825  ← best checkpoint saved
  Epoch 15: 0.4786  (converged, no further improvement)

─────────────────────────────────────────────────────────

5. RESULTS
──────────
Test set evaluation (best checkpoint, epoch 8):

  ┌──────────────────────────────┬────────┬────────┬──────┐
  │ Prompt                       │  mIoU  │  Dice  │   N  │
  ├──────────────────────────────┼────────┼────────┼──────┤
  │ segment taping area          │ 0.5730 │ 0.7194 │  209 │
  │ segment crack                │ 0.4592 │ 0.6064 │  632 │
  ├──────────────────────────────┼────────┼────────┼──────┤
  │ Average                      │ 0.5161 │ 0.6629 │  841 │
  └──────────────────────────────┴────────┴────────┴──────┘

The taping area prompt outperforms cracks consistently. Taping areas are
wide vertical strips that approximately fill their bounding boxes, making
the rectangular mask approximation a reasonable ground truth. Cracks are
thin, irregular structures where even a visually correct prediction scores
lower due to boundary sensitivity in IoU-based metrics.

─────────────────────────────────────────────────────────

6. VISUAL EXAMPLES
──────────────────
[Insert 4 images from outputs/visuals/ here]
[2 from segment_taping_area, 2 from segment_crack]

Each panel shows: Original Image | Ground Truth Mask | Predicted Mask
with mIoU and Dice score annotated on the prediction panel.

─────────────────────────────────────────────────────────

7. FAILURE ANALYSIS
────────────────────
- Hairline cracks: The model misses very thin cracks (< 5px wide) due to
  CLIPSeg's 64×64 internal resolution. Upsampling to 640×640 cannot recover
  fine detail lost at the encoder stage.

- Bounding box ground truth: For taping area, the rectangular masks include
  wall background pixels within the bounding box. This introduces label
  noise during training and artificially lowers test mIoU even for correct
  predictions.

- Multiple cracks per image: When several cracks appear in one image, the
  model tends to identify the most prominent one and miss secondary cracks,
  reducing recall.

- Low-contrast scenes: Cracks in dark or uniform-colored walls are frequently
  missed entirely, showing the model's dependence on edge and texture contrast.

─────────────────────────────────────────────────────────

8. RUNTIME & FOOTPRINT
───────────────────────
  Training time         : ~75 minutes (15 epochs, T4 GPU)
  Avg inference time    : 43.8 ms/image (22.8 FPS)
  Checkpoint size       : 603.2 MB
  Model parameters      : 150.7M total / 1.1M trained

─────────────────────────────────────────────────────────

9. LIMITATIONS & FUTURE WORK
──────────────────────────────
The primary ceiling on performance is CLIPSeg's decoder resolution and the
bounding-box ground truth for the drywall dataset. Using Grounded SAM 2
with its high-resolution mask decoder would directly address the resolution
limitation. Acquiring pixel-level annotations for the drywall seam dataset
would address the ground truth quality limitation. Both improvements are
expected to push average mIoU above 0.70.