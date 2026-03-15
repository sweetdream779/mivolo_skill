# mivolo_skill

A Claude Code skill for age and gender detection in images using [MiVOLO v2](https://huggingface.co/iitolstykh/mivolo_v2).

## Model

- **Model:** [iitolstykh/mivolo_v2](https://huggingface.co/iitolstykh/mivolo_v2) (28.8M parameters)
- **Library:** [WildChlamydia/MiVOLO](https://github.com/WildChlamydia/MiVOLO)
- **Gender accuracy:** 97.3–98.0%
- **Age MAE:** ~3.65 years
- **Input size:** 384×384 px (face + body crops)

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
    "bbox": [120, 45, 310, 390]
  },
  {
    "person_id": 2,
    "gender": "male",
    "gender_confidence": 0.991,
    "age": 42.1,
    "bbox": [400, 60, 590, 410]
  }
]
```

## Requirements

- Python 3.8+
- PyTorch 1.13+
- GPU recommended (CUDA), works on CPU

## References

- [MiVOLO paper](https://arxiv.org/abs/2307.04616)
- [MiVOLO GitHub](https://github.com/WildChlamydia/MiVOLO)
- [Model on HuggingFace](https://huggingface.co/iitolstykh/mivolo_v2)
