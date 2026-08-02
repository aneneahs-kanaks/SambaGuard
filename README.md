# Ubuntu-Informed Edge AI for Real-Time Fall Armyworm Detection with Automated Swahili SMS Advisory

---

## Overview

SambaGuard AI is an edge AI system for early detection of Fall Armyworm (FAW) infestations in smallholder maize farms. It runs a fine-tuned YOLOv8s object detection model on a Raspberry Pi 5, detects FAW life stages and damage signatures in real time using a camera module, and is designed to trigger a Swahili-language advisory SMS to the farmer upon detection.

Fall Armyworm is one of the most destructive pests affecting maize across Sub-Saharan Africa. The effective window for low-cost biological intervention is narrow, and smallholder farmers cannot manually scout frequently enough to catch early-stage infestations in time. SambaGuard AI automates this monitoring layer — providing severity-aware, localized detection that does not depend on internet connectivity or expensive infrastructure.

---

## Features

- Real-time FAW detection on edge hardware with no cloud dependency
- Four-class detection: egg, frass, larva, and larval damage
- ONNX and TFLite export for flexible deployment on constrained hardware
- Raspberry Pi Camera Module v3 support via picamera2
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
    No detection      FAW detected
    (continue)             |
                           v
               Swahili LLM Advisory Layer
               UlizaLlama — severity-graded
               advice in Swahili (planned)
                           |
                           v
               SMS alert delivered to farmer
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
| Number of classes | 4 |

### Detected Classes

| Class ID | Class Name |
|---|---|
| 0 | Fall Armyworm Egg |
| 1 | Fall Armyworm Frass |
| 2 | Fall Armyworm Larva |
| 3 | Fall Armyworm Larval Damage |

---

## Dataset

The model was trained on a cleaned and validated subset of the [KaraAgro AI Maize dataset](https://datasetninja.com/kara-agro-ai-maize), accessed via Dataset Ninja in Supervisely format. The original dataset was published by KaraAgro AI and is available on [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CXUMDS) (DOI: 10.7910/DVN/CXUMDS, License: CC0 1.0).

Full dataset preparation process is documented in [`docs/annotation-conversion-guide.md`](docs/annotation-conversion-guide.md).

| Split | Images | Labels |
|---|---|---|
| Train | 5,709 | 5,709 |
| Val | 1,320 | 1,320 |
| Test | 664 | 664 |

---

## Performance

Results from the v2 training run. Best checkpoint obtained at epoch 55.

### Overall Metrics

| Metric | v1 Baseline | v2 Clean Dataset |
|---|---|---|
| Precision | 0.486 | 0.479 |
| Recall | 0.401 | 0.376 |
| mAP50 | 0.368 | 0.347 |
| mAP50-95 | 0.144 | 0.137 |
| Best Epoch | 50 | 55 |

### Per-Class Performance (v2)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Fall Armyworm Egg | 0.336 | 0.250 | 0.198 | 0.085 |
| Fall Armyworm Frass | 0.374 | 0.233 | 0.196 | 0.066 |
| Fall Armyworm Larva | 0.782 | 0.716 | 0.726 | 0.303 |
| Fall Armyworm Larval Damage | 0.425 | 0.304 | 0.267 | 0.093 |

Larva detection achieves the strongest performance (mAP50 = 0.726), consistent with its larger visual signature and stronger representation in the training data. Egg and frass remain areas for improvement in subsequent versions.

Training plots, confusion matrix, PR curves, and model weights are available on [Hugging Face](https://huggingface.co/ndunge23/SambaGuard-v2/tree/main).

### Experiment Log

| Version | Key Change | Epochs | mAP50 | Status |
|---|---|---|---|---|
| v1 | Baseline | 50 | 0.368 | Complete |
| v2 | Clean dataset | 100 (best at 55) | 0.347 | Complete |

---

## Repository Structure

```
SambaGuard/
├── docs/
│   ├── dataset-setup-guide.md
│   ├── annotation-conversion-guide.md
│   ├── pipeline-summary.md
│   ├── problem-statement.md
│   └── pi-deployment-guide.md
├── scripts/
│   ├── inference.py
│   └── test_image.py
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
pip install ultralytics onnxruntime opencv-python-headless numpy pillow
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
| 2 | Install dependencies: onnxruntime, opencv-python-headless, picamera2 |
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
- Expansion to additional maize pest and disease classes

---

## Model on Hugging Face

Trained weights, evaluation metrics, training plots, and confusion matrix are available at:

[https://huggingface.co/ndunge23/SambaGuard-v2](https://huggingface.co/ndunge23/SambaGuard-v2/tree/main)

---

## Citation

```bibtex
@misc{sambaguard2026,
  author       = {Annastacia Ndunge},
  title        = {SambaGuard AI: Edge AI for Fall Armyworm Detection in Smallholder Maize Farms},
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
