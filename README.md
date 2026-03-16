# mivolo_skill

A Claude Code skill for age and gender detection in images using MiVOLO v2.
Both models are loaded via the **Transformers API** (`trust_remote_code=True`).

## What this skill does

**mivolo_skill** lets Claude Code detect the **age and gender** of people in images — single portraits, group photos, or entire folders of images.

Under the hood it runs a two-stage pipeline:
1. **YOLOv8x detector** finds all faces and bodies in the image
2. **MiVOLO v2** estimates age and gender from the matched face + body crops

The skill returns structured JSON with per-person results and can optionally save an annotated image with bounding boxes and labels.

**Key capabilities:**
- Multiple people per image — each detected person gets their own result
- Works even without a visible face — body-only predictions supported
- Batch processing — run on an entire folder at once
- GPU + CPU support — auto-detects available hardware
- Isolated environment — installed in its own `.venv`, no system Python impact

---

## Quick Start

```bash
git clone https://github.com/sweetdream779/mivolo_skill
cd mivolo_skill
claude
```

Then type `/setup` — Claude will automatically install all dependencies, create a virtual environment, and register the skill globally.

After setup, just ask from **any** Claude Code session:
```
Determine the age and gender of people in this image: photo.jpg
```

Models (~500MB) download automatically on first run.

---

## Table of Contents

- [Installation](#installation)
- [Models](#models)
- [Pipeline](#pipeline)
- [Usage](#usage)
  - [From Claude Code](#from-claude-code)
  - [CLI](#cli)
- [Examples](#examples)
  - [Single person](#single-person)
  - [Group photo](#group-photo-multiple-people)
  - [Person without visible face](#person-without-visible-face-only-body-detected)
  - [Batch processing](#batch-processing-a-folder)
- [Output format](#output-format)
- [Requirements](#requirements)
- [References](#references)

---

## Installation

### Automatic via `/setup` (recommended)

```bash
git clone https://github.com/sweetdream779/mivolo_skill
cd mivolo_skill
claude
# Type: /setup
```

The `/setup` command will:
1. Check that **Python 3.8+** is installed
2. Create an isolated **virtual environment** at `.venv/`
3. Install all dependencies (PyTorch, Transformers, ultralytics, MiVOLO)
4. Register the skill in `~/.claude/skills/` so it's available globally
5. Verify the installation

### Manual install

```bash
git clone https://github.com/sweetdream779/mivolo_skill ~/.claude/skills/mivolo_skill
bash ~/.claude/skills/mivolo_skill/install.sh
```

Claude Code picks up the skill automatically — no restart needed.

---

## Models

### 1. Detector — [iitolstykh/YOLO-Face-Person-Detector](https://huggingface.co/iitolstykh/YOLO-Face-Person-Detector)
YOLOv8x fine-tuned on ~150k images for detecting two classes: **Face** and **Person**.
- Loaded via `AutoModel.from_pretrained`
- Conf threshold: 0.4 / IoU threshold: 0.7

### 2. Age & Gender — [iitolstykh/mivolo_v2](https://huggingface.co/iitolstykh/mivolo_v2)
MiVOLO v2 — state-of-the-art age and gender estimation from face + body crops.
- Loaded via `AutoModelForImageClassification.from_pretrained`
- **Gender accuracy:** 97.3–98.0%
- **Age MAE:** ~3.65 years
- **Input size:** 384x384 px per crop

Models are downloaded automatically on first run from HuggingFace and cached in `~/.cache/huggingface/`.

## Pipeline

```
Image → YOLO Detector → faces[] + persons[]
                      → match face↔person by IoU
                      → crop & resize to 384×384
                      → MiVOLO v2 → age + gender per person
```

## Usage

### From Claude Code

Simply ask Claude in natural language:
```
Determine the age and gender of people in this image: /path/to/photo.jpg
```
```
How old are the people in this photo? photo.jpg
```
```
Run age and gender detection on all images in ./dataset/ and save annotated results to ./results/
```
```
Process group_photo.jpg and tell me how many men and women are in the picture
```

### CLI

```bash
SKILL_DIR="$HOME/.claude/skills/mivolo_skill"

# Single image
$SKILL_DIR/.venv/bin/python $SKILL_DIR/mivolo_inference.py --image photo.jpg

# Image from URL
$SKILL_DIR/.venv/bin/python $SKILL_DIR/mivolo_inference.py --image https://example.com/photo.jpg

# Save annotated image with bounding boxes and labels
$SKILL_DIR/.venv/bin/python $SKILL_DIR/mivolo_inference.py --image photo.jpg --draw --output result.jpg

# Process entire folder
$SKILL_DIR/.venv/bin/python $SKILL_DIR/mivolo_inference.py --image ./photos/ --draw --output ./results/

# Force CPU (no GPU required)
$SKILL_DIR/.venv/bin/python $SKILL_DIR/mivolo_inference.py --image photo.jpg --device cpu
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--image` | Yes | Path to image, directory, or URL |
| `--output` | No | Path to save annotated image |
| `--device` | No | `cuda` (default if available) or `cpu` |
| `--draw` | No | Save annotated image with bounding boxes and labels |

## Examples

### Single person

```bash
$SKILL_DIR/.venv/bin/python $SKILL_DIR/mivolo_inference.py --image portrait.jpg
```
```json
[
  {
    "person_id": 1,
    "gender": "female",
    "gender_confidence": 0.981,
    "age": 34.2,
    "face_box": [210, 80, 390, 280],
    "person_box": [150, 60, 460, 720]
  }
]
```

### Group photo (multiple people)

```bash
$SKILL_DIR/.venv/bin/python $SKILL_DIR/mivolo_inference.py --image group.jpg --draw --output group_annotated.jpg
```
```json
[
  {
    "person_id": 1,
    "gender": "male",
    "gender_confidence": 0.994,
    "age": 42.7,
    "face_box": [95, 40, 220, 180],
    "person_box": [60, 30, 260, 480]
  },
  {
    "person_id": 2,
    "gender": "female",
    "gender_confidence": 0.976,
    "age": 27.1,
    "face_box": [310, 55, 430, 195],
    "person_box": [280, 45, 470, 490]
  },
  {
    "person_id": 3,
    "gender": "male",
    "gender_confidence": 0.963,
    "age": 61.4,
    "face_box": [520, 70, 650, 200],
    "person_box": [490, 60, 690, 500]
  }
]
```

### Person without visible face (only body detected)

```bash
$SKILL_DIR/.venv/bin/python $SKILL_DIR/mivolo_inference.py --image back_view.jpg
```
```json
[
  {
    "person_id": 1,
    "gender": "female",
    "gender_confidence": 0.912,
    "age": 29.8,
    "face_box": null,
    "person_box": [120, 50, 380, 700]
  }
]
```

### Batch processing a folder

```bash
$SKILL_DIR/.venv/bin/python $SKILL_DIR/mivolo_inference.py --image ./dataset/ --draw --output ./results/
```
```json
{
  "photo1.jpg": [
    {"person_id": 1, "gender": "male", "age": 35.0, "gender_confidence": 0.988, "face_box": [...], "person_box": [...]}
  ],
  "photo2.jpg": [
    {"person_id": 1, "gender": "female", "age": 22.3, "gender_confidence": 0.971, "face_box": [...], "person_box": [...]},
    {"person_id": 2, "gender": "male", "age": 48.6, "gender_confidence": 0.959, "face_box": [...], "person_box": [...]}
  ]
}
```

## Output format

```json
[
  {
    "person_id": 1,
    "gender": "female",
    "gender_confidence": 0.973,
    "age": 28.4,
    "face_box": [120, 45, 310, 200],
    "person_box": [100, 40, 320, 390]
  }
]
```

## Requirements

- Python 3.8+
- PyTorch 1.13+
- GPU recommended (CUDA), works on CPU
- ~2GB disk (venv + models)

## References

- [MiVOLO paper](https://arxiv.org/abs/2307.04616)
- [MiVOLO GitHub](https://github.com/WildChlamydia/MiVOLO)
- [MiVOLO v2 on HuggingFace](https://huggingface.co/iitolstykh/mivolo_v2)
- [YOLO Detector on HuggingFace](https://huggingface.co/iitolstykh/YOLO-Face-Person-Detector)
