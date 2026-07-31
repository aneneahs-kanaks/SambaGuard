"""
inference.py
SambaGuard AI — Live Camera Inference

Runs real-time Fall Armyworm detection on the Raspberry Pi Camera Module v3
using the exported ONNX model. Prints detections to terminal and triggers
an alert when FAW is detected above the confidence threshold.

Usage:
    python3 scripts/inference.py

Press Ctrl+C to stop.

Requirements:
    pip install opencv-python-headless numpy onnxruntime picamera2
"""

import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import time

# ── Configuration ────────────────────────────────────────────
MODEL_PATH           = 'weights/best.onnx'
INPUT_SIZE           = 640
CONFIDENCE_THRESHOLD = 0.5    # ignore detections below 50% confidence
IOU_THRESHOLD        = 0.45   # threshold for Non-Maximum Suppression

CLASS_NAMES = [
    'fall armyworm egg',
    'fall armyworm frass',
    'fall armyworm larva',
    'fall armyworm larval damage'
]

# colours per class in BGR format for OpenCV
CLASS_COLORS = {
    'fall armyworm egg'          : (0, 255, 255),   # yellow
    'fall armyworm frass'        : (0, 165, 255),   # orange
    'fall armyworm larva'        : (0, 0, 255),     # red
    'fall armyworm larval damage': (255, 0, 0),     # blue
}

# ── Load Model ───────────────────────────────────────────────
print(f"Loading model from: {MODEL_PATH}")
session    = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
print("Model loaded successfully")


# ── Parsing ──────────────────────────────────────────────────
def parse_detections(output, orig_h, orig_w):
    """
    Convert raw YOLOv8 ONNX output into a list of detections.

    The model returns a tensor of shape [1, 8, 8400]:
      - 1    : batch size (one image)
      - 8    : 4 box coordinates + 4 class scores (one per FAW class)
      - 8400 : candidate detection locations across the image grid

    This function filters candidates by confidence, converts coordinates
    back to pixel positions on the original image, and removes duplicate
    overlapping boxes via Non-Maximum Suppression (NMS).

    Args:
        output : raw model output, shape [1, 8, 8400]
        orig_h : original image height in pixels
        orig_w : original image width in pixels

    Returns:
        list of dicts: [{'class_name', 'confidence', 'box':[x1,y1,x2,y2]}, ...]
    """
    # squeeze batch dimension: [1, 8, 8400] -> [8, 8400]
    predictions = output[0].squeeze(0)

    # transpose to [8400, 8] — process row by row
    predictions = predictions.T

    # split into box coordinates and class scores
    boxes        = predictions[:, :4]   # [8400, 4] x_center, y_center, w, h
    class_scores = predictions[:, 4:]   # [8400, 4] one score per FAW class

    # get best class and confidence for each candidate
    confidences = np.max(class_scores, axis=1)
    class_ids   = np.argmax(class_scores, axis=1)

    # filter out low-confidence candidates
    mask        = confidences >= CONFIDENCE_THRESHOLD
    boxes       = boxes[mask]
    confidences = confidences[mask]
    class_ids   = class_ids[mask]

    if len(boxes) == 0:
        return []

    # convert from center format (x_center, y_center, w, h)
    # to corner format (x1, y1, x2, y2) in original image pixels
    scale_x = orig_w / INPUT_SIZE
    scale_y = orig_h / INPUT_SIZE

    x1 = (boxes[:, 0] - boxes[:, 2] / 2) * scale_x
    y1 = (boxes[:, 1] - boxes[:, 3] / 2) * scale_y
    x2 = (boxes[:, 0] + boxes[:, 2] / 2) * scale_x
    y2 = (boxes[:, 1] + boxes[:, 3] / 2) * scale_y

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # Non-Maximum Suppression — keeps only the best box per object,
    # removes lower-confidence duplicates that overlap the same region
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(),
        confidences.tolist(),
        CONFIDENCE_THRESHOLD,
        IOU_THRESHOLD
    )

    detections = []
    for i in indices:
        detections.append({
            'class_name' : CLASS_NAMES[class_ids[i]],
            'confidence' : float(confidences[i]),
            'box'        : [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])]
        })

    return detections


# ── Drawing ──────────────────────────────────────────────────
def draw_detections(frame, detections):
    """Draw bounding boxes and class labels on the frame."""
    for det in detections:
        x1, y1, x2, y2 = det['box']
        label           = det['class_name']
        conf            = det['confidence']
        color           = CLASS_COLORS.get(label, (0, 255, 0))

        # bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # label background
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)

        # label text
        cv2.putText(frame, text, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


# ── Alert ────────────────────────────────────────────────────
def check_alert(detections):
    """
    Check if FAW has been detected and return a summary string
    for the SMS advisory layer.

    Returns None if no FAW detected, otherwise a summary string.
    """
    if not detections:
        return None

    class_counts = {}
    for det in detections:
        name = det['class_name']
        class_counts[name] = class_counts.get(name, 0) + 1

    summary = "FAW DETECTED: " + ", ".join(
        f"{count} {name}" for name, count in class_counts.items()
    )
    return summary


# ── Main Loop ────────────────────────────────────────────────
def main():
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (1280, 960)}
    ))
    picam2.start()
    time.sleep(2)  # allow camera to warm up
    print("Camera started — press Ctrl+C to stop\n")

    try:
        while True:
            # capture frame from Pi Camera
            frame          = picam2.capture_array()
            orig_h, orig_w = frame.shape[:2]

            # preprocess frame for model input
            img_resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
            img_input   = img_resized.astype(np.float32) / 255.0
            img_input   = np.transpose(img_input, (2, 0, 1))  # HWC to CHW
            img_input   = np.expand_dims(img_input, axis=0)   # add batch dim

            # run inference
            start   = time.time()
            outputs = session.run(None, {input_name: img_input})
            elapsed = time.time() - start

            # parse raw output into structured detections
            detections = parse_detections(outputs, orig_h, orig_w)

            # draw boxes and labels on frame
            frame = draw_detections(frame, detections)

            # print results to terminal
            fps = 1 / elapsed
            print(f"FPS: {fps:.1f} | Detections: {len(detections)}")
            for det in detections:
                print(f"  {det['class_name']:<35} "
                      f"conf={det['confidence']:.3f}  "
                      f"box={det['box']}")

            # check for FAW alert
            alert = check_alert(detections)
            if alert:
                print(f"\nALERT: {alert}")
                # TODO: pass alert to Swahili LLM advisory layer
                # TODO: send SMS via GSM module

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        picam2.stop()


if __name__ == "__main__":
    main()
