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
# Basic inference — prints JSON to stdout
python mivolo_inference.py --image photo.jpg

# Save annotated image with bounding boxes and labels
python mivolo_inference.py --image photo.jpg --draw --output result.jpg

# Process entire folder of images
python mivolo_inference.py --image ./photos/ --draw --output ./results/

# Force CPU (no GPU required)
python mivolo_inference.py --image photo.jpg --device cpu
```

## Examples

### Single person

```bash
python mivolo_inference.py --image portrait.jpg
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
python mivolo_inference.py --image group.jpg --draw --output group_annotated.jpg
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
python mivolo_inference.py --image back_view.jpg
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
python mivolo_inference.py --image ./dataset/ --draw --output ./results/
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
