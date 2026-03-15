# MiVOLO Skill — Age & Gender Detection

Detect age and gender of people in images using the MiVOLO v2 model.

## Overview

This skill uses MiVOLO (a state-of-the-art age and gender estimation model) to analyze images and return age and gender predictions for each detected person. It processes both face and body crops for improved accuracy.

- **Gender accuracy:** 97.3–98.0%
- **Age MAE:** ~3.65 years
- **Model:** [iitolstykh/mivolo_v2](https://huggingface.co/iitolstykh/mivolo_v2)

## Usage

Invoke this skill when the user wants to:
- Detect the age and/or gender of people in an image
- Analyze demographic information from photos
- Process multiple people in a single image

### Example prompts
- "Determine the age and gender of people in this image: /path/to/image.jpg"
- "Who is in this photo and how old are they? image.png"
- "Run age and gender detection on all images in this folder"

## How to run

```bash
python mivolo_inference.py --image <path_to_image> [--output <output_path>] [--device cpu|cuda]
```

### Arguments
- `--image` (required) — path to input image or directory of images
- `--output` — path to save annotated output image (optional)
- `--device` — `cuda` (default if available) or `cpu`
- `--draw` — save annotated image with bounding boxes and labels

### Output format
Returns a JSON list of detected persons:
```json
[
  {
    "person_id": 1,
    "gender": "female",
    "gender_confidence": 0.97,
    "age": 28.4,
    "bbox": [x1, y1, x2, y2]
  }
]
```

## Installation

Run `bash install.sh` to install all dependencies.

**System requirements:**
- Python 3.8+
- PyTorch 1.13+
- GPU recommended (works on CPU too, but slower)
