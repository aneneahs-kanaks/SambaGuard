# SambaGuard AI — Raspberry Pi Deployment Guide

This guide covers everything needed to deploy the SambaGuard FAW detection model on a Raspberry Pi for real-time inference in the field.

---

## Hardware

### Recommended: Raspberry Pi 5

Use the **Raspberry Pi 5** over the Pi 4 for this project. Key reasons:

- The Pi 5 CPU (Cortex-A76) is roughly 2-3x faster than the Pi 4 (Cortex-A72) for CPU-bound tasks like TFLite inference
- Expected FPS on Pi 5 with YOLOv8s ONNX: approximately 8-12 FPS vs 3-6 FPS on Pi 4
- The Pi 5 has enough headroom to run both the detection model and the Swahili LLM advisory layer simultaneously — the Pi 4 would struggle with both at once
- Setup process is identical on both boards

### Full Hardware List

| Item | Specification | Notes |
|---|---|---|
| Single-board computer | Raspberry Pi 5 (8GB RAM) | Use the 8GB variant |
| MicroSD card | 64GB, SanDisk or Samsung | Cheap cards cause random crashes — brand matters |
| Power supply | 27W USB-C (official Pi 5 PSU) | Pi 5 requires more power than Pi 4 — use the correct PSU |
| Camera | Raspberry Pi Camera Module v3 | Plugs into the CSI/FPC camera port |
| Camera ribbon cable | Pi 5 FPC cable | Pi 5 uses a smaller connector than Pi 4 — confirm cable compatibility |
| Display adapter | Micro-HDMI to VGA adapter | Pi 5 has micro-HDMI ports; needed for initial setup only |
| Keyboard + mouse | Any USB | Needed for first boot only |
| Case (optional) | Pi 5 compatible case with fan | Pi 5 runs hot under sustained inference — cooling recommended |

> **Note on the display:** VGA monitors work fine with a micro-HDMI to VGA adapter. Once WiFi and SSH are configured during first boot, the monitor is no longer needed — all subsequent work can be done remotely from your laptop via SSH.

---

## Model Format

This guide uses the **ONNX** export of the SambaGuard model (`best.onnx`).

ONNX was chosen over TFLite for this deployment because:
- ONNX Runtime is actively maintained and well-supported on Pi 5
- Slightly better compatibility with the Pi 5's newer ARM architecture
- Easier to debug and inspect than TFLite flatbuffers

The model file (`best.onnx`) is available in the SambaGuard Hugging Face repository under `weights/`.

---

## Part 1: Setting Up the Raspberry Pi 5

### Step 1 — Flash the Operating System

On your laptop:

1. Download **Raspberry Pi Imager** from [raspberrypi.com/software](https://www.raspberrypi.com/software/)
2. Insert your 64GB microSD card into your laptop
3. Open Raspberry Pi Imager
4. Select device: **Raspberry Pi 5**
5. Select OS: **Raspberry Pi OS (64-bit)** — the full version, not Lite
6. Select your microSD card as the storage target
7. Click the **settings icon** before flashing and configure:
   - Hostname: `sambaguard`
   - Enable SSH (use password authentication)
   - Set username and password (remember these)
   - Configure your WiFi network name and password
8. Click **Save** then **Write**

> Configuring WiFi and SSH before flashing saves significant setup time — you will be able to connect remotely immediately after first boot.

### Step 2 — First Boot

1. Insert the flashed microSD card into the Pi 5
2. Connect the Camera Module v3 ribbon cable to the CSI/FPC port
3. Connect micro-HDMI to VGA adapter and plug into your monitor
4. Connect keyboard and mouse via USB
5. Connect the official 27W USB-C power supply last — this powers it on

The Pi will boot into Raspberry Pi OS desktop. Wait for it to finish first-boot setup (takes 1-2 minutes).

### Step 3 — Update the System

Open a terminal on the Pi and run:

```bash
sudo apt update && sudo apt upgrade -y
sudo reboot
```

This ensures all system packages are current before installing dependencies.

### Step 4 — Enable the Camera

```bash
sudo raspi-config
```

Navigate to: **Interface Options** → **Camera** → **Enable**

Select **Finish** and reboot when prompted:

```bash
sudo reboot
```

Verify the camera is detected after reboot:

```bash
libcamera-hello --list-cameras
```

You should see your Camera Module v3 listed. If nothing appears, check the ribbon cable connection.

### Step 5 — Connect via SSH (Optional but Recommended)

Once WiFi is connected, find the Pi's IP address:

```bash
hostname -I
```

From your laptop terminal, connect remotely:

```bash
ssh pi@sambaguard.local
# or use the IP address directly
ssh pi@192.168.x.x
```

From this point forward, you can work entirely from your laptop without the monitor.

---

## Part 2: Installing SambaGuard Dependencies

### Step 1 — Create a Virtual Environment

```bash
cd ~
python3 -m venv sambaguard_env
source sambaguard_env/bin/activate
```

Always activate this environment before running SambaGuard scripts:

```bash
source ~/sambaguard_env/bin/activate
```

### Step 2 — Install Required Packages

```bash
pip install --upgrade pip

# computer vision and image processing
pip install opencv-python-headless
pip install numpy pillow

# ONNX inference runtime
pip install onnxruntime

# Raspberry Pi camera support
pip install picamera2
```

> **Why `opencv-python-headless`?** The standard `opencv-python` tries to install GUI dependencies that conflict with Pi's display stack. The headless version handles all camera capture and image processing without the GUI overhead.

> **Why `onnxruntime` and not the full ONNX package?** `onnxruntime` is the lightweight inference-only runtime — it is all you need to run the model. The full ONNX package is for model conversion, which you have already done on your development machine.

### Step 3 — Verify Installations

```bash
python3 -c "import cv2; print('OpenCV:', cv2.__version__)"
python3 -c "import onnxruntime as ort; print('ONNX Runtime:', ort.__version__)"
python3 -c "import numpy as np; print('NumPy:', np.__version__)"
```

All three should print version numbers without errors.

---

## Part 3: Transferring the Model and Scripts

### Option A — Clone from GitHub (Recommended)

```bash
cd ~
git clone https://github.com/aneneahs-kanaks/SambaGuard.git
cd SambaGuard
```

### Option B — Copy from Your Laptop via SCP

From your laptop terminal:

```bash
# create the deployment folder on the Pi
ssh pi@sambaguard.local "mkdir -p ~/sambaguard"

# copy the model
scp best.onnx pi@sambaguard.local:~/sambaguard/

# copy the inference script (once written)
scp inference.py pi@sambaguard.local:~/sambaguard/
```

---

## Part 4: Running Inference

### Step 1 — Test on a Single Image First

Before running the live camera feed, confirm the model loads and runs correctly on a static test image:

```python
import cv2
import numpy as np
import onnxruntime as ort

# class names in the same order as training
CLASS_NAMES = [
    'fall armyworm egg',
    'fall armyworm frass',
    'fall armyworm larva',
    'fall armyworm larval damage'
]

# load model
session = ort.InferenceSession('best.onnx')
input_name = session.get_inputs()[0].name
print("Model loaded successfully")

# load and preprocess a test image
img = cv2.imread('test_image.jpg')
img_resized = cv2.resize(img, (640, 640))
img_input = img_resized.astype(np.float32) / 255.0
img_input = np.transpose(img_input, (2, 0, 1))  # HWC to CHW
img_input = np.expand_dims(img_input, axis=0)   # add batch dimension

# run inference
outputs = session.run(None, {input_name: img_input})
print("Inference complete")
print("Output shape:", outputs[0].shape)
```

### Step 2 — Live Camera Inference

Once the static test works, run on the live camera feed:

```python
import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import time

CLASS_NAMES = [
    'fall armyworm egg',
    'fall armyworm frass',
    'fall armyworm larva',
    'fall armyworm larval damage'
]

CONFIDENCE_THRESHOLD = 0.5

# load model
session = ort.InferenceSession('best.onnx')
input_name = session.get_inputs()[0].name
print("Model loaded")

# initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (1280, 960)}
))
picam2.start()
time.sleep(2)  # allow camera to warm up

print("Camera started — press Ctrl+C to stop")

while True:
    # capture frame
    frame = picam2.capture_array()

    # preprocess
    img_resized = cv2.resize(frame, (640, 640))
    img_input = img_resized.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0)

    # run inference and time it
    start = time.time()
    outputs = session.run(None, {input_name: img_input})
    elapsed = time.time() - start

    print(f"Inference time: {elapsed*1000:.1f} ms ({1/elapsed:.1f} FPS)")

    # parse detections (format depends on YOLOv8 ONNX output)
    # full parsing logic to be added here

picam2.stop()
```

---

## Part 5: Expected Performance

| Metric | Expected Value on Pi 5 |
|---|---|
| Model format | ONNX |
| Inference time per frame | 80-150 ms |
| Frames per second | 8-12 FPS |
| RAM usage (model + runtime) | ~500 MB |
| CPU usage during inference | 60-80% |

These are estimates based on published Pi 5 benchmarks for similar YOLOv8s ONNX models. Actual performance will be measured and recorded once the Pi is running.

---

## Part 6: What to Test on First Run

Work through these in order — do not skip ahead:

1. Camera detected by the system (`libcamera-hello --list-cameras`)
2. Model loads without errors (static image test)
3. Inference completes on a single static image
4. Output shape is correct (should be `[1, 8, 8400]` for YOLOv8s with 4 classes)
5. Live camera feed opens and captures frames
6. Inference runs on live frames with reasonable FPS
7. Detections appear correctly on test images of maize plants
8. Confidence threshold filters out low-confidence noise
9. FAW detection correctly triggers the alert logic

---

## Troubleshooting

| Issue | Likely Cause | Fix |
|---|---|---|
| Camera not detected | Ribbon cable not fully seated | Power off, reseat cable, reboot |
| `onnxruntime` install fails | ARM architecture package unavailable | Try `pip install onnxruntime` with `--extra-index-url https://pkgs.dev.azure.com/onnxruntime/onnxruntime/_packaging/onnxruntime-arm/pypi/simple/` |
| Very slow inference (>500ms) | Running on CPU with wrong backend | Confirm `onnxruntime` is using the ARM optimized build |
| Pi gets very hot | No cooling | Add heatsinks or a fan — sustained inference generates significant heat on Pi 5 |
| SSH connection refused | SSH not enabled or wrong IP | Re-flash with SSH enabled in Imager settings |
| Model output shape unexpected | Wrong ONNX export settings | Re-export from Kaggle with `model.export(format='onnx', imgsz=640)` |

---

## Next Steps After Successful Deployment

Once real-time inference is confirmed working:

1. Write the full detection parsing and bounding box drawing logic
2. Implement the FAW detection alert trigger (confidence threshold + minimum detection count)
3. Integrate the Swahili LLM advisory layer
4. Connect GSM module for SMS delivery to farmers
5. Field test with actual maize crop images
6. Benchmark and document FPS, accuracy, and latency in real conditions

---

## Related Resources

- SambaGuard GitHub: https://github.com/aneneahs-kanaks/SambaGuard
- SambaGuard Model (Hugging Face): https://huggingface.co/ndunge23/sambaguard-faw-detector
- Raspberry Pi 5 Documentation: https://www.raspberrypi.com/documentation/
- ONNX Runtime ARM Documentation: https://onnxruntime.ai/docs/build/eps.html
- Picamera2 Documentation: https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf

---

*Last updated: July 2026*
