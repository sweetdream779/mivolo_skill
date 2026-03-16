---
name: mivolo_skill
description: Detect the age and gender of people in images using MiVOLO v2 and a YOLOv8 face+person detector. Supports single images, folders, and URLs. Use when the user asks to detect age, gender, or demographic information from photos.
allowed-tools: Bash(mivolo:*)
---

# MiVOLO Skill — Age & Gender Detection

Detect age and gender of people in images using the MiVOLO v2 model.

- **Gender accuracy:** 97.3–98.0%
- **Age MAE:** ~3.65 years
- **Models:** [iitolstykh/mivolo_v2](https://huggingface.co/iitolstykh/mivolo_v2), [iitolstykh/YOLO-Face-Person-Detector](https://huggingface.co/iitolstykh/YOLO-Face-Person-Detector)

## Auto-setup

Before first use, check if the virtual environment exists. If not, install automatically:

```bash
SKILL_DIR="$(cd "$(dirname "$(readlink -f "$0")" 2>/dev/null || echo "$0")/../" && pwd)"
# Fallback: locate via .claude/skills
[ -d "$SKILL_DIR/.venv" ] || SKILL_DIR="$HOME/.claude/skills/mivolo_skill"
[ -d "$SKILL_DIR/.venv" ] || bash "$SKILL_DIR/install.sh"
```

If the `.venv/` directory already exists, skip installation.

Models (~500MB total) are downloaded from HuggingFace automatically on the first inference run.

## Usage

```bash
SKILL_DIR="$HOME/.claude/skills/mivolo_skill"

# Default: JSON-only output, no image saved
$SKILL_DIR/.venv/bin/python $SKILL_DIR/mivolo_inference.py --image photo.jpg

# With annotated output (only when user explicitly asks for a visual/annotated result)
$SKILL_DIR/.venv/bin/python $SKILL_DIR/mivolo_inference.py --image photo.jpg --draw --output result.jpg

# Process a directory of images
$SKILL_DIR/.venv/bin/python $SKILL_DIR/mivolo_inference.py --image ./photos/ --draw --output ./results/

# Image from URL
$SKILL_DIR/.venv/bin/python $SKILL_DIR/mivolo_inference.py --image "https://example.com/photo.jpg"
```

### When to use `--draw --output`

- **DO NOT** use `--draw` / `--output` by default. The JSON output is sufficient.
- **DO** use `--draw --output <unique_name>.jpg` only when the user explicitly asks for an annotated/visual result.
- When using `--output`, generate a **unique filename** based on the input (e.g., `photo_annotated.jpg`), never hardcode `result.jpg`.

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--image` | Yes | Path to image, directory, or URL |
| `--output` | No | Path to save annotated image (use only when user asks) |
| `--device` | No | `cuda` (default if available) or `cpu` |
| `--draw` | No | Save annotated image with bounding boxes and labels |

### Output format

Returns JSON to stdout:

```json
[
  {
    "person_id": 1,
    "gender": "female",
    "gender_confidence": 0.97,
    "age": 28.4,
    "face_box": [x1, y1, x2, y2],
    "person_box": [x1, y1, x2, y2]
  }
]
```

## Installation

### Quick install (Claude Code)

Copy this skill to your Claude Code skills directory:

```bash
cp -r /path/to/mivolo_skill ~/.claude/skills/mivolo_skill
```

The skill will auto-install its dependencies on first use.

### Manual install

```bash
cd ~/.claude/skills/mivolo_skill
bash install.sh
```

This creates an isolated `.venv/` with PyTorch, Transformers, and all dependencies. No impact on your system Python.

### System requirements

- Python 3.10
- ~2GB disk (venv + models)
- GPU recommended (works on CPU, but slower)
