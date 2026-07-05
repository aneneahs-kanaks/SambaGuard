# SambaGuard — Full Pipeline Summary

This document summarizes every step taken to build the SambaGuard (Angalia) FAW detection pipeline from raw dataset to a fully organized, training-ready YOLOv8 dataset.

For setup and download steps, see [`dataset-setup-guide.md`](./dataset-setup-guide.md).
For annotation conversion details, see [`annotation-conversion-guide.md`](./annotation-conversion-guide.md).
For the project's research-backed problem statement, see [`problem-statement.md`](./problem-statement.md).

---

## Phase 1: Dataset Acquisition

- Identified the **KaraAgro AI Maize dataset** on Dataset Ninja (Supervisely format, 31.5GB, 16,552 images, 8 classes)
- Downloaded directly into Lightning AI Studio using the `dataset-tools` Python package — server-to-server transfer, no local upload needed:

```python
import dataset_tools as dtools
dtools.download(dataset='KaraAgro AI Maize', dst_dir='~/dataset-ninja/')
```

- Dataset auto-extracted (49,664 files) during download — no manual unpacking required
- Verified: **16,552 images**, ~31GB on disk after cleanup of the leftover `.tar` archive

---

## Phase 2: Project Scoping Decision

The original dataset has 8 classes:
```
0: fall armyworm egg
1: fall armyworm frass
2: fall armyworm larva
3: fall armyworm larval damage
4: healthy images
5: healthy maize
6: maize streak disease
7: none healthy
```

**Initial decision:** scope to 4 FAW-specific classes only (0–3), dropping non-FAW images entirely.

**Revised decision:** keep a random **10% of non-FAW images as background/negative samples** (written as empty label files with no bounding boxes). This reduces false positives in the field — backed by research showing smallholder farmers regularly confuse FAW with look-alike pests like stalk borer and African armyworm.

**Final class mapping used for training:**
```
0: fall armyworm egg
1: fall armyworm frass
2: fall armyworm larva
3: fall armyworm larval damage
```

---

## Phase 3: Problem Statement & Research

Before committing to the build, the project's value proposition was stress-tested against the question: *"If farmers already know FAW exists and know the control methods, are we just repeating what they know?"*

Literature review confirmed the real gaps are **not** awareness gaps but execution gaps:

- **Misidentification** — farmers in Kenya and Ethiopia confuse FAW with stalk borer, spotted stalk borer, and African armyworm, leading to wrong pesticide choices that fail to control the actual infestation
- **Detection timing** — early FAW life stages are cryptic and hard to spot manually; smallholders lack time for frequent enough scouting to catch the narrow effective intervention window
- **Generic advice** — existing guidance is not localized or severity-graded enough to tell a farmer what to do right now given exactly what's in their field

**Reframed value proposition:**
> *"Frequent, severity-aware monitoring that catches the narrow FAW intervention window smallholder farmers structurally cannot catch through manual scouting alone — then delivers specific, timely, localized guidance based on exactly what is detected."*

See [`problem-statement.md`](./problem-statement.md) for full sources and citations.

---

## Phase 4: Understanding Supervisely Annotation Format

Supervisely format stores annotations as per-image `.json` files under `ann/` folders, with images in a parallel `img/` folder. The actual dataset structure:

```
dataset-ninja/
└── karaagro-ai-maize/
    └── ds/
        ├── ann/   ← one .json per image
        ├── img/   ← actual image files
        └── meta/
```

Each annotation JSON has an `objects` list where each entry contains:
- `classTitle` — class name as a string (e.g. `"fall armyworm larva"`)
- `points.exterior` — two `[x, y]` corner coordinates in absolute pixels defining the bounding box

A sample image was visualized with bounding boxes drawn using PIL to confirm annotations mapped correctly before writing conversion code.

---

## Phase 5: Annotation Conversion (Supervisely → YOLOv8)

YOLOv8 expects normalized, center-based bounding box format per line in a `.txt` file:
```
class_id x_center y_center width height
```

**Conversion math (absolute pixel corners → normalized center-based):**
```
x_center = (x1 + x2) / 2 / image_width
y_center = (y1 + y2) / 2 / image_height
width    = (x2 - x1) / image_width
height   = (y2 - y1) / image_height
```

**Key design decisions in the conversion loop:**

- **FAW-only filtering** — skips non-FAW class objects within mixed images
- **10% background sampling** — keeps a random 10% of non-FAW images as empty label files (negative examples)
- **`min/max` geometry fix** — handles reverse-drawn bounding boxes where annotators drew from bottom-right to top-left, which would produce negative widths/heights without this correction:
```python
xmin, xmax = min(x1, x2), max(x1, x2)
ymin, ymax = min(y1, y2), max(y1, y2)
```
- **Per-image dimension reading** — images have mixed resolutions across the dataset, so width/height is read individually per image rather than assumed
- **`try/except` wrapping** — one malformed annotation file doesn't crash the entire loop; errors are logged and the loop continues

**Conversion results:**
```
Converted (FAW images kept):       ~11,036 images
Kept backgrounds (healthy/streak):  1,090 images
Skipped completely:                 ~4,426 images
Errors:                             0
Total usable images:               12,126
```

---

## Phase 6: Class Distribution Analysis

Before splitting, per-class image counts were checked:

```
class -1 (background):                1,090 images
class  0 (fall armyworm egg):           404 images  ← underrepresented
class  1 (fall armyworm frass):       3,504 images
class  2 (fall armyworm larva):       2,704 images
class  3 (fall armyworm larval damage): 4,424 images
```

The egg class (404 images) is significantly underrepresented compared to the others. At a 70/20/10 split, this would leave only ~283 egg images in training — workable, but likely to produce weaker detection performance for that class.

**Decision:** oversample the egg class in the training split only (val/test kept clean and unmodified) to bring the egg training count up to approximately the frass level as a reference target.

---

## Phase 7: Stratified Train/Val/Test Split

**Split ratio:** 70 / 20 / 10 (train / val / test)

**Stratification approach:** split performed within each class group separately (not a global random shuffle), so every class is proportionally represented in each split — particularly important given the class imbalance.

**Dependency note:** `sklearn.model_selection.train_test_split` was attempted but failed at import due to a numpy/scikit-learn binary ABI incompatibility:
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
Expected 96 from C header, got 88 from PyObject
```
Root cause: `dataset-tools` pulled in numpy 2.3.3, but the installed scikit-learn was compiled against numpy 1.x. Rather than chase the dependency conflict, a manual stratified split was implemented using only `random` and `collections.defaultdict` — same outcome, zero dependency risk.

**Egg class oversampling:** after the stratified split, egg images in the training set were oversampled using `random.choices()` with replacement, targeting the frass training count as the reference level. Duplicate files written with `_dup0`, `_dup1` suffixes to avoid overwriting originals.

**Final split counts:**
```
Train: 10,655  (includes 2,170 oversampled egg images)
Val:    2,422
Test:   1,219
Total: 14,296  (across all splits including duplicates)
```

---

## Phase 8: YOLOv8 Folder Organization

Organized into the standard YOLOv8 directory structure:

```
faw_dataset/
├── images/
│   ├── train/   (10,655 images)
│   ├── val/     (2,422 images)
│   └── test/    (1,219 images)
└── labels/
    ├── train/   (10,655 .txt label files)
    ├── val/     (2,422 .txt label files)
    └── test/    (1,219 .txt label files)
```

**Bug encountered and fixed — double file extension:**

The dataset contains two filename types:
- **Type A:** `IMG_20220720_081801_257` — no extension in stem, needs one appended when searching
- **Type B:** `1628014182456_jpg.rf.d55ccb44f16a438d1374a22fcbf6bdb3.jpg` — extension already baked into the stem

The original `find_image()` function always appended an extension, causing Type B filenames to search for `.jpg.jpg` — a double extension that never matched anything, explaining the high initial missing count (5,717 missing in train alone).

**Fix — try direct match first, then extension-appending fallback:**
```python
def find_image(base):
    # Case 1: stem already has extension baked in
    matches = glob.glob(os.path.join(base_dir, '**', 'img', base), recursive=True)
    if matches:
        return matches[0]
    # Case 2: stem has no extension — try appending common ones
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
        matches = glob.glob(os.path.join(base_dir, '**', 'img', base + ext), recursive=True)
        if matches:
            return matches[0]
    return None
```

**Final copy results:**
```
train: copied 10,655, missing 0
val:   copied  2,422, missing 0
test:  copied  1,219, missing 0
```

---

## Phase 9: data.yaml Generation

```yaml
train: /teamspace/studios/this_studio/faw_dataset/images/train
val: /teamspace/studios/this_studio/faw_dataset/images/val
test: /teamspace/studios/this_studio/faw_dataset/images/test

nc: 4
names: ['fall armyworm egg', 'fall armyworm frass', 'fall armyworm larva', 'fall armyworm larval damage']
```

---

## Incident Log: Data Loss During Platform Migration

**What happened:** during an attempt to move the ~63GB dataset from the Studio's local disk to Lightning AI's teamspace Drive (`/teamspace/datasets/`), the standard `mkdir` command failed with `Permission denied`. A `sudo mkdir` workaround was used instead of going through Lightning AI's proper "Add Data" UI flow. This coincided with an active personal-teamspace-to-organization migration on Lightning AI's platform (June 29, 2026). The full dataset was subsequently lost — not present in either the original location or the intended destination.

**Recovery:** re-ran the original `dataset-tools` download script, which completed in ~15 minutes. No code or documentation was lost — only the data itself, since all conversion and split logic was already committed to GitHub and saved in the project notebook.

**Lessons learned:**
1. Never use `sudo` to force-create platform-managed storage paths — if `mkdir` fails, use the platform's official UI flow instead
2. Avoid storage migrations during active platform-side infrastructure changes (e.g. announced org migrations)
3. Confirm a destination path is genuinely persistent and visible in the platform's official Datasets UI before moving or deleting the only copy of data
4. Keep all scripts and notebooks committed to GitHub early — makes data-loss recovery a re-run, not a rebuild

---

## What's Next

- **YOLOv8 fine-tuning** — train `yolov8s.pt` on `faw_dataset/` using `ultralytics`
- **Evaluation** — assess per-class mAP, particularly egg class (class 0) which has the fewest training examples even after oversampling
- **Edge export** — convert best checkpoint to TFLite or ONNX for Raspberry Pi 4 deployment
- **LLM advisory layer** — integrate UlizaLlama (7B Swahili-capable model) as the SMS advisory layer, triggered by detection output

---
*Last updated: June 2026*
