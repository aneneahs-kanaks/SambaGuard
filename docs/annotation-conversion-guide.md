# Dataset Annotation Conversion & Scoping Guide

This documents the process of converting the KaraAgro AI Maize dataset's Supervisely-format annotations into YOLOv8-compatible labels, scoped specifically to Fall Armyworm (FAW) classes.

Continues from [`dataset-setup-guide.md`](./dataset-setup-guide.md), which covers downloading and unpacking the raw dataset.

---

## Scoping Decision: FAW-Only Classes

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

**Decision: train only on the 4 FAW-specific classes (0–3).** Images containing only non-FAW classes (healthy maize, maize streak disease, etc.) are dropped entirely from the training set, matching the project's scope as a dedicated FAW detector.

> **Note for future reference:** this was discussed and weighed against keeping non-FAW images as negative examples (to reduce false positives on look-alike pests/diseases, per the project's [problem statement](./problem-statement.md)). The team chose the simpler FAW-only scope for this phase. If false-positive rates on non-FAW maize images become an issue during evaluation, revisit this decision and consider reintroducing negative examples.

---

## Step 1: Understanding Supervisely Annotation Format

Each image has a matching `.json` annotation file under an `ann/` folder (with the image itself in a parallel `img/` folder). The key field is `objects`, a list where each entry has:
- `classTitle` — the class name as a string (e.g. `"fall armyworm larva"`)
- `points.exterior` — two `[x, y]` corner coordinates defining the bounding box, in absolute pixel values

```python
import os
import json
import glob

base_dir = os.path.expanduser('~/dataset-ninja')

ann_paths = glob.glob(f'{base_dir}/**/ann/*.json', recursive=True)
print(f"Total annotation files: {len(ann_paths)}")

# inspect one sample
with open(ann_paths[0], 'r') as f:
    sample_ann = json.load(f)
print(json.dumps(sample_ann, indent=2))
```

---

## Step 2: Visualizing a Sample (sanity check)

Before writing conversion logic, confirm the annotation maps correctly onto its image:

```python
from PIL import Image, ImageDraw

ann_path = ann_paths[0]
img_dir = os.path.dirname(ann_path).replace('/ann', '/img')
img_filename = os.path.basename(ann_path).replace('.json', '')
img_path = os.path.join(img_dir, img_filename)

with open(ann_path, 'r') as f:
    ann = json.load(f)

img = Image.open(img_path).convert("RGB")
draw = ImageDraw.Draw(img)

for obj in ann['objects']:
    class_title = obj['classTitle']
    (x1, y1), (x2, y2) = obj['points']['exterior']
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    draw.text((x1, max(y1 - 10, 0)), class_title, fill="red")

img
```

---

## Step 3: Converting Supervisely Boxes → YOLOv8 Format

Supervisely stores boxes as two absolute-pixel corners. YOLOv8 needs normalized (0–1), center-based format: `class_id x_center y_center width height`.

Conversion math:
```
x_center = (x1 + x2) / 2 / image_width
y_center = (y1 + y2) / 2 / image_height
width    = (x2 - x1) / image_width
height   = (y2 - y1) / image_height
```

### FAW-only class mapping
```python
faw_class_names = [
    'fall armyworm egg',
    'fall armyworm frass',
    'fall armyworm larva',
    'fall armyworm larval damage'
]
class_to_id = {name: idx for idx, name in enumerate(faw_class_names)}
```

### Full conversion loop (all images, FAW-filtered)

This processes every annotation file, **keeps only FAW-class objects**, and **skips images entirely if they contain no FAW objects** (e.g. pure healthy-maize or maize-streak-disease images are dropped from the training set).

```python
labels_out_dir = os.path.expanduser('~/labels_converted_faw_only')
os.makedirs(labels_out_dir, exist_ok=True)

success_count = 0
skipped_no_faw = 0
error_count = 0
errors = []

for ann_path in ann_paths:
    try:
        img_dir = os.path.dirname(ann_path).replace('/ann', '/img')
        img_filename = os.path.basename(ann_path).replace('.json', '')
        img_path = os.path.join(img_dir, img_filename)

        with open(ann_path, 'r') as f:
            ann_data = json.load(f)

        faw_objects = [obj for obj in ann_data['objects'] if obj['classTitle'] in class_to_id]

        if not faw_objects:
            skipped_no_faw += 1
            continue

        with Image.open(img_path) as img_obj:
            iw, ih = img_obj.size

        yolo_lines = []
        for obj in faw_objects:
            class_id = class_to_id[obj['classTitle']]
            (x1, y1), (x2, y2) = obj['points']['exterior']

            x_center = (x1 + x2) / 2 / iw
            y_center = (y1 + y2) / 2 / ih
            width    = (x2 - x1) / iw
            height   = (y2 - y1) / ih

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        with open(os.path.join(labels_out_dir, label_filename), 'w') as f:
            f.write('\n'.join(yolo_lines))

        success_count += 1

    except Exception as e:
        error_count += 1
        errors.append((ann_path, str(e)))

print(f"Converted (FAW images kept): {success_count}")
print(f"Skipped (no FAW objects):    {skipped_no_faw}")
print(f"Errors:                      {error_count}")
```

**Why each image is reopened individually:** image resolutions are not uniform across the dataset, so width/height must be read per-image rather than assumed.

**Why the loop is wrapped in try/except:** one malformed annotation shouldn't crash the entire run — errors are logged and the loop continues, with a final count for review.

---

## Step 4: Stratified Train/Val/Test Split

Once labels are converted, split by class proportionally (stratified) rather than a pure random shuffle, so rare classes aren't underrepresented in any split.

**Note:** `sklearn.model_selection.train_test_split` was attempted first but hit a numpy/scikit-learn binary incompatibility (`numpy.dtype size changed` — a compiled-extension ABI mismatch from competing numpy version requirements across installed packages). Rather than chase dependency versions, a manual stratified split was used instead — same outcome, zero dependency risk.

```python
import random
from collections import defaultdict

random.seed(42)

class_to_stems = defaultdict(list)
for stem, class_id in image_classes:  # image_classes: list of (filename_stem, dominant_class_id)
    class_to_stems[class_id].append(stem)

train_stems, val_stems, test_stems = [], [], []

for class_id, stems_list in class_to_stems.items():
    random.shuffle(stems_list)
    n = len(stems_list)
    n_train = int(n * 0.70)
    n_val = int(n * 0.20)

    train_stems.extend(stems_list[:n_train])
    val_stems.extend(stems_list[n_train:n_train + n_val])
    test_stems.extend(stems_list[n_train + n_val:])

print(f"Train: {len(train_stems)}")
print(f"Val:   {len(val_stems)}")
print(f"Test:  {len(test_stems)}")
```

Split ratio used: **70/10/20** train/val/test (decided after reviewing project priorities — adjust as needed).

---

## Step 5: Organizing into YOLOv8 Folder Structure

YOLOv8 expects:
```
yolo_dataset/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```
with matching filenames between an image and its label (e.g. `img001.jpg` ↔ `img001.txt`).

```python
import shutil

yolo_base = os.path.expanduser('~/yolo_dataset')

splits = {'train': train_stems, 'val': val_stems, 'test': test_stems}

for split_name, stems_list in splits.items():
    img_out_dir = os.path.join(yolo_base, 'images', split_name)
    lbl_out_dir = os.path.join(yolo_base, 'labels', split_name)
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    copied, missing = 0, []

    for stem in stems_list:
        possible_exts = ['.jpg', '.jpeg', '.png']
        src_img_path = None
        for ext in possible_exts:
            matches = glob.glob(os.path.join(base_dir, '**', 'img', stem + ext), recursive=True)
            if matches:
                src_img_path = matches[0]
                break

        src_lbl_path = os.path.join(labels_out_dir, stem + '.txt')

        if src_img_path and os.path.exists(src_lbl_path):
            shutil.copy(src_img_path, os.path.join(img_out_dir, os.path.basename(src_img_path)))
            shutil.copy(src_lbl_path, os.path.join(lbl_out_dir, stem + '.txt'))
            copied += 1
        else:
            missing.append(stem)

    print(f"{split_name}: copied {copied}, missing {len(missing)}")
```

---

## Step 6: Creating `data.yaml`

```python
yaml_content = f"""train: {os.path.join(yolo_base, 'images', 'train')}
val: {os.path.join(yolo_base, 'images', 'val')}
test: {os.path.join(yolo_base, 'images', 'test')}

nc: {len(faw_class_names)}
names: {faw_class_names}
"""

yaml_path = os.path.join(yolo_base, 'data.yaml')
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print(yaml_content)
```

---

## Incident Log: Data Loss During Platform Migration

**What happened:** while attempting to move the dataset from the Studio's local disk to Lightning AI's teamspace Drive storage (`/teamspace/datasets/`), the target path did not exist and `mkdir` failed with `Permission denied`. A `sudo mkdir` workaround was attempted instead of using the platform's proper "Add Data" UI flow. This coincided with an active personal-teamspace-to-organization migration on Lightning AI's platform. The dataset (~63GB) was subsequently lost — not present in either the original location or the intended destination.

**Root cause:** not confirmed with certainty (no visibility into Lightning's storage backend), but most likely an ad-hoc `sudo`-created directory was not properly registered as a persistent Drive asset, and was dropped or orphaned during the platform-side migration.

**Recovery:** re-ran the original download script (`dataset-tools`), which completed successfully in ~15 minutes. No code or documentation was lost, only the data itself, since the conversion/split logic was already written and saved in the project notebook.

**Lessons for future reference:**
1. Do not use `sudo` to force-create platform-managed storage paths (e.g. `/teamspace/datasets/`) when the standard method is rejected — that rejection is usually a signal the path should be provisioned through the UI instead.
2. Avoid storage migrations during active platform-side infrastructure changes (e.g. announced org migrations) — wait until they've completed.
3. Keep a copy of `download.py` and conversion notebooks committed to GitHub early, so recovery from data loss only requires re-running scripts, not rewriting logic.
4. Confirm a destination path is genuinely persistent (e.g. visible via the platform's official "Datasets" UI) before deleting or moving the only copy of data into it.

---
*Last updated: June 2026*
