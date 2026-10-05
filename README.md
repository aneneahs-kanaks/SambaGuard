# SambaGuard AI — Maize Pest and Disease Detection System

---

## Overview

SambaGuard AI is an edge AI system for early detection of Fall Armyworm (FAW) infestations and Maize Streak Disease in smallholder maize farms. It runs a fine-tuned YOLOv8s object detection model on a Raspberry Pi 5, detects FAW life stages, damage signatures, and disease symptoms in real time using a camera module, and is designed to trigger an AI-generated agricultural advisory upon detection.

Fall Armyworm is one of the most destructive pests affecting maize across Sub-Saharan Africa. The effective window for low-cost biological intervention is narrow, and smallholder farmers cannot manually scout frequently enough to catch early-stage infestations in time. SambaGuard AI automates this monitoring layer — providing severity-aware, localized detection that does not depend on internet connectivity or expensive infrastructure.

---

## Live Demo

Try the cloud demo on Hugging Face Spaces:

[https://huggingface.co/spaces/ndunge23/SambaGuard](https://huggingface.co/spaces/ndunge23/SambaGuard)

Supports image upload, webcam capture, and URL input. Detections are accompanied by an AI-generated agricultural advisory.

---

## Features

- Real-time pest and disease detection on edge hardware with no cloud dependency
- Five-class detection: FAW egg, frass, larva, larval damage, and maize streak disease
- ONNX and TFLite export for flexible deployment on constrained hardware
- Raspberry Pi Camera Module v3 support via picamera2
- AI-generated agricultural advisory via Gemini
- Designed for offline use in low-connectivity field environments
- Planned: Swahili-language advisory SMS via a quantized LLM

---

## System Architecture

```
Camera Module (Raspberry Pi Camera v3)
                |
                v
       Frame Capture (picamera2)
                |
                v
    Preprocessing — resize 640x640, normalize
                |
                v
   YOLOv8s Detection Model (ONNX / TFLite)
                |
                v
     Detection Parsing + NMS
      class · confidence · bounding box
                |
         -------+--------
         |               |
         v               v
    No detection      Pest/Disease detected
    (continue)             |
                           v
               AI Advisory Layer (Gemini)
               Severity-graded advice
                           |
                           v
               Advisory delivered to farmer
               (Planned: Swahili SMS)
```

---

## Model

| Property | Value |
|---|---|
| Architecture | YOLOv8s |
| Framework | Ultralytics / PyTorch |
| Input size | 640 x 640 pixels |
| Parameters | 11.1M |
| GFLOPs | 28.4 |
| Export formats | ONNX, TFLite |
| Number of classes | 5 |

### Detected Classes

| Class ID | Class Name |
|---|---|
| 0 | Fall Armyworm Egg |
| 1 | Fall Armyworm Frass |
| 2 | Fall Armyworm Larva |
| 3 | Fall Armyworm Larval Damage |
| 4 | Maize Streak Disease |

---

## Dataset

The model was trained on a subset of the [KaraAgro AI Maize dataset](https://datasetninja.com/kara-agro-ai-maize), accessed via Dataset Ninja in Supervisely format. The original dataset was collected by KaraAgro AI from maize farms across Ghana and is available on [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CXUMDS) (DOI: 10.7910/DVN/CXUMDS, License: CC0 1.0).

Full dataset preparation process is documented in [`docs/annotation-conversion-guide.md`](docs/annotation-conversion-guide.md).

| Split | Images | Labels |
|---|---|---|
| Train | 5,084 | 5,084 |
| Val | 1,715 | 1,715 |
| Test | 664 | 664 |

---

## Performance

### Experiment Log

| Version | Classes | Key Change | Epochs | mAP50 | Status |
|---|---|---|---|---|---|
| v1 | 4 | Baseline FAW-only | 50 | 0.368 | Complete |
| v2 | 4 | Clean dataset | 100 (best 55) | 0.347 | Complete |
| v3 | 6 | Roboflow dataset | 100 (best 49) | 0.161 | Complete |
| v4 | 6 | Roboflow + oversampling | 100 (best 56) | 0.166 | Complete |
| **Current** | **5** | **KaraAgro 5-class** | **50** | **0.459** | **Complete** |

### Current Model — Overall Metrics

| Metric | Value |
|---|---|
| Precision | 0.500 |
| Recall | 0.466 |
| mAP50 | 0.459 |
| mAP50-95 | 0.220 |
| Best Epoch | 49 |

### Current Model — Per-Class Performance

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Fall Armyworm Egg | — | 0.135 | 0.090 | 0.030 |
| Fall Armyworm Frass | 0.404 | 0.249 | 0.228 | 0.074 |
| Fall Armyworm Larva | 0.796 | 0.875 | 0.893 | 0.352 |
| Fall Armyworm Larval Damage | 0.464 | 0.328 | 0.316 | 0.110 |
| Maize Streak Disease | 0.645 | 0.743 | 0.767 | 0.535 |

Training plots, confusion matrix, PR curves, and model weights are available on [Hugging Face](https://huggingface.co/ndunge23/disease-detector).

---

## Repository Structure

```
SambaGuard/
├── app/
│   ├── app.py
│   └── requirements.txt
├── docs/
│   ├── dataset-setup-guide.md
│   ├── annotation-conversion-guide.md
│   ├── pipeline-summary.md
│   ├── problem-statement.md
│   └── pi-deployment-guide.md
├── scripts/
│   ├── inference.py
│   ├── test_image.py
│   └── train.py
└── README.md
```

---

## Quick Start

**Clone the repository**

```bash
git clone https://github.com/aneneahs-kanaks/SambaGuard.git
cd SambaGuard
```

**Install requirements**

```bash
pip install ultralytics onnxruntime numpy pillow
```

**Run inference on a single image**

```bash
python3 scripts/test_image.py --model best.onnx --image test_image.jpg
```

**Run live camera inference on Raspberry Pi**

```bash
python3 scripts/inference.py --model best.onnx
```

---

## Deployment

Target hardware: Raspberry Pi 5 (8GB RAM) with Raspberry Pi Camera Module v3.

| Step | Action |
|---|---|
| 1 | Flash Raspberry Pi OS (64-bit) and configure SSH |
| 2 | Install dependencies: onnxruntime, picamera2, numpy, pillow |
| 3 | Clone this repository and transfer best.onnx |
| 4 | Run scripts/test_image.py to verify the model loads correctly |
| 5 | Run scripts/inference.py for live camera inference |

Expected inference speed on Pi 5 with ONNX: approximately 8-12 FPS.

Full setup walkthrough: [docs/pi-deployment-guide.md](docs/pi-deployment-guide.md)

---

## Future Roadmap

- Swahili-language LLM advisory layer (UlizaLlama or equivalent quantized model)
- GSM module integration for SMS delivery to farmers
- Model quantization for further edge optimization
- Field validation with smallholder farmers in Kenya
- GPS tagging and regional FAW spread mapping
- SMS integration via Africa's Talking API

---

## Models on Hugging Face

| Model | Classes | mAP50 | Link |
|---|---|---|---|
| SambaGuard v2 (FAW only) | 4 | 0.347 | [ndunge23/SambaGuard-v2](https://huggingface.co/ndunge23/SambaGuard-v2) |
| Pest Disease Detector v4 | 6 | 0.166 | [ndunge23/Pest_Disease_Detector](https://huggingface.co/ndunge23/Pest_Disease_Detector) |
| Current model (best) | 5 | 0.459 | [ndunge23/disease-detector](https://huggingface.co/ndunge23/disease-detector) |

---

## Citation

```bibtex
@misc{sambaguard2026,
  author       = {Annastacia Ndunge},
  title        = {SambaGuard AI: Edge AI for Maize Pest and Disease Detection},
  year         = {2026},
  howpublished = {\url{https://github.com/aneneahs-kanaks/SambaGuard}},
  note         = {Work in progress}
}
```

---

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

---

## References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ONNX Runtime](https://onnxruntime.ai)
- [Dataset Ninja — KaraAgro AI Maize](https://datasetninja.com/kara-agro-ai-maize)
- [KaraAgro AI Maize on Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CXUMDS)
- [picamera2 Documentation](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)

---

## Author

Annastacia Ndunge
Electrical and Electronic Engineering, Dedan Kimathi University of Technology, Kenya

GitHub: [aneneahs-kanaks](https://github.com/aneneahs-kanaks)
Hugging Face: [ndunge23](https://huggingface.co/ndunge23)
LinkedIn: [Annastacia Ndunge](https://www.linkedin.com/in/annastacia-ndunge-809a21361)
