"""
test_image.py
SambaGuard AI — Single Image Inference Test

Run this script first to confirm the model loads and produces
correct output before attempting live camera inference.

Usage:
    python3 scripts/test_image.py

Place a test image named 'test_image.jpg' in the same directory,
or update IMAGE_PATH below to point at your image.
"""

import cv2
import numpy as np
import onnxruntime as ort
import os

# ── Configuration ────────────────────────────────────────────
MODEL_PATH  = 'weights/best.onnx'
IMAGE_PATH  = 'test_image.jpg'
INPUT_SIZE  = 640

CLASS_NAMES = [
    'fall armyworm egg',
    'fall armyworm frass',
    'fall armyworm larva',
    'fall armyworm larval damage'
]

CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD        = 0.45

# ── Load Model ───────────────────────────────────────────────
print(f"Loading model from: {MODEL_PATH}")
assert os.path.exists(MODEL_PATH), f"Model not found at {MODEL_PATH}"

session    = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
print(f"Model loaded successfully")
print(f"Input name:  {input_name}")
print(f"Input shape: {session.get_inputs()[0].shape}")

# ── Load Image ───────────────────────────────────────────────
print(f"\nLoading image from: {IMAGE_PATH}")
assert os.path.exists(IMAGE_PATH), f"Image not found at {IMAGE_PATH}"

img    = cv2.imread(IMAGE_PATH)
orig_h, orig_w = img.shape[:2]
print(f"Image size: {orig_w} x {orig_h}")

# ── Preprocess ───────────────────────────────────────────────
img_resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
img_input   = img_resized.astype(np.float32) / 255.0
img_input   = np.transpose(img_input, (2, 0, 1))  # HWC to CHW
img_input   = np.expand_dims(img_input, axis=0)   # add batch dimension

# ── Run Inference ────────────────────────────────────────────
print("\nRunning inference...")
import time
start   = time.time()
outputs = session.run(None, {input_name: img_input})
elapsed = time.time() - start

print(f"Inference complete in {elapsed*1000:.1f} ms")
print(f"Output shape: {outputs[0].shape}")

# ── Parse Detections ─────────────────────────────────────────
predictions = outputs[0].squeeze(0).T  # [8400, 8]
boxes        = predictions[:, :4]
class_scores = predictions[:, 4:]

confidences = np.max(class_scores, axis=1)
class_ids   = np.argmax(class_scores, axis=1)

mask        = confidences >= CONFIDENCE_THRESHOLD
boxes       = boxes[mask]
confidences = confidences[mask]
class_ids   = class_ids[mask]

print(f"\nDetections above {CONFIDENCE_THRESHOLD} confidence: {len(boxes)}")

if len(boxes) > 0:
    scale_x = orig_w / INPUT_SIZE
    scale_y = orig_h / INPUT_SIZE

    x1 = (boxes[:, 0] - boxes[:, 2] / 2) * scale_x
    y1 = (boxes[:, 1] - boxes[:, 3] / 2) * scale_y
    x2 = (boxes[:, 0] + boxes[:, 2] / 2) * scale_x
    y2 = (boxes[:, 1] + boxes[:, 3] / 2) * scale_y

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(),
        confidences.tolist(),
        CONFIDENCE_THRESHOLD,
        IOU_THRESHOLD
    )

    print("\nFinal detections after NMS:")
    for i in indices:
        print(f"  {CLASS_NAMES[class_ids[i]]:<35} "
              f"conf={confidences[i]:.3f}  "
              f"box=[{int(x1[i])}, {int(y1[i])}, {int(x2[i])}, {int(y2[i])}]")
else:
    print("No FAW objects detected in this image.")

print("\nTest complete — model is working correctly.")
