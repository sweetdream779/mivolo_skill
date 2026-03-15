# mivolo_skill

A Claude Code skill for age and gender detection in images using MiVOLO v2.
Both models are loaded via the **Transformers API** (`trust_remote_code=True`).

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
- **Input size:** 384×384 px per crop

## Pipeline

```
Image → YOLO Detector → faces[] + persons[]
                      → match face↔person by IoU
                      → crop & resize to 384×384
                      → MiVOLO v2 → age + gender per person
```

## Installation

### As a Claude Code skill

```bash
npx skills add https://github.com/YOUR_USERNAME/mivolo_skill
```

### Manual

```bash
git clone https://github.com/YOUR_USERNAME/mivolo_skill
cd mivolo_skill
bash install.sh
```

Models are downloaded automatically on first run from HuggingFace.

## Usage

### From Claude Code

Simply ask Claude:
```
Determine the age and gender of people in this image: /path/to/photo.jpg
```

### CLI

```bash
# Basic inference — prints JSON to stdout
python mivolo_inference.py --image photo.jpg

# Save annotated image with bounding boxes and labels
python mivolo_inference.py --image photo.jpg --draw --output result.jpg

# Process entire folder
python mivolo_inference.py --image ./photos/ --draw --output ./results/

# Force CPU
python mivolo_inference.py --image photo.jpg --device cpu
```

## Output

```json
[
  {
    "person_id": 1,
    "gender": "female",
    "gender_confidence": 0.973,
    "age": 28.4,
    "face_box": [120, 45, 310, 200],
    "person_box": [100, 40, 320, 390]
  },
  {
    "person_id": 2,
    "gender": "male",
    "gender_confidence": 0.991,
    "age": 42.1,
    "face_box": [400, 60, 530, 180],
    "person_box": [380, 55, 590, 410]
  }
]
```

## Requirements

- Python 3.8+
- PyTorch 1.13+
- GPU recommended (CUDA), works on CPU
- `ultralytics==8.1.0` — backend dependency for the YOLO detector (not imported directly)

## References

- [MiVOLO paper](https://arxiv.org/abs/2307.04616)
- [MiVOLO GitHub](https://github.com/WildChlamydia/MiVOLO)
- [MiVOLO v2 on HuggingFace](https://huggingface.co/iitolstykh/mivolo_v2)
- [YOLO Detector on HuggingFace](https://huggingface.co/iitolstykh/YOLO-Face-Person-Detector)
