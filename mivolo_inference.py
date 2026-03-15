#!/usr/bin/env python3
"""
MiVOLO Skill — Age & Gender Detection

Pipeline:
  1. Detect faces and persons via YOLO (iitolstykh/YOLO-Face-Person-Detector)
  2. Match each face to the nearest person body
  3. Crop and preprocess face + body crops (384x384)
  4. Run MiVOLO v2 (iitolstykh/mivolo_v2) for age & gender prediction

Models:
  - Detector: https://huggingface.co/iitolstykh/YOLO-Face-Person-Detector
  - MiVOLO:   https://huggingface.co/iitolstykh/mivolo_v2
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── Constants ──────────────────────────────────────────────────────────────────

DETECTOR_MODEL_ID = "iitolstykh/YOLO-Face-Person-Detector"
MIVOLO_MODEL_ID = "iitolstykh/mivolo_v2"
CROP_SIZE = 384
DETECTOR_CONF = 0.4
DETECTOR_IOU = 0.7


# ── Model loading ──────────────────────────────────────────────────────────────

def load_detector(device: str):
    """Load YOLOv8 face+person detector from HuggingFace."""
    from transformers import AutoModel

    print(f"Loading detector {DETECTOR_MODEL_ID}...", file=sys.stderr)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        DETECTOR_MODEL_ID,
        trust_remote_code=True,
        dtype=dtype,
    ).to(device)
    return model


def load_mivolo(device: str):
    """Load MiVOLO v2 model, processor and config from HuggingFace."""
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

    print(f"Loading MiVOLO {MIVOLO_MODEL_ID}...", file=sys.stderr)
    dtype = torch.float16 if device == "cuda" else torch.float32
    config = AutoConfig.from_pretrained(MIVOLO_MODEL_ID, trust_remote_code=True)
    processor = AutoImageProcessor.from_pretrained(MIVOLO_MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageClassification.from_pretrained(
        MIVOLO_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    return model, processor, config


# ── Detection ──────────────────────────────────────────────────────────────────

def detect(detector, image: Image.Image) -> tuple[list, list]:
    """
    Run YOLO detector and return separate lists of face and person boxes.

    Returns:
        faces:   list of [x1, y1, x2, y2] for class 'Face'
        persons: list of [x1, y1, x2, y2] for class 'Person'
    """
    results = detector(image, conf=DETECTOR_CONF, iou=DETECTOR_IOU)[0]
    faces, persons = [], []
    for box in results.boxes:
        cls_name = results.names[int(box.cls)]
        coords = [round(v) for v in box.xyxy[0].tolist()]
        if cls_name == "Face":
            faces.append(coords)
        elif cls_name == "Person":
            persons.append(coords)
    return faces, persons


def iou(box_a: list, box_b: list) -> float:
    """Compute Intersection over Union between two boxes [x1,y1,x2,y2]."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_faces_to_persons(faces: list, persons: list) -> list[dict]:
    """
    Match each face to the best overlapping person body.
    A face is matched to a person if the face bbox overlaps with the person bbox
    (face center is inside person box, or highest IoU).

    Returns list of dicts: {face_box, person_box or None}
    """
    matched = []
    used_faces = set()

    for person_box in persons:
        best_face, best_score = None, -1.0
        for i, face_box in enumerate(faces):
            if i in used_faces:
                continue
            # Check if face center is inside person box
            cx = (face_box[0] + face_box[2]) / 2
            cy = (face_box[1] + face_box[3]) / 2
            inside = (person_box[0] <= cx <= person_box[2] and
                      person_box[1] <= cy <= person_box[3])
            score = iou(face_box, person_box) + (1.0 if inside else 0.0)
            if score > best_score:
                best_score, best_face = score, i
        if best_face is not None and best_score > 0:
            matched.append({"face_box": faces[best_face], "person_box": person_box})
            used_faces.add(best_face)
        else:
            matched.append({"face_box": None, "person_box": person_box})

    # Add unmatched faces (no corresponding body detected)
    for i, face_box in enumerate(faces):
        if i not in used_faces:
            matched.append({"face_box": face_box, "person_box": None})

    return matched


# ── Cropping ───────────────────────────────────────────────────────────────────

def crop_and_resize(image: Image.Image, box: list | None) -> Image.Image | None:
    """Crop region from image and resize to CROP_SIZE x CROP_SIZE."""
    if box is None:
        return None
    w, h = image.size
    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image.crop((x1, y1, x2, y2))
    return crop.resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)


# ── MiVOLO inference ───────────────────────────────────────────────────────────

def predict_age_gender(mivolo_model, processor, config, device: str,
                       face_crop: Image.Image | None,
                       body_crop: Image.Image | None) -> dict:
    """Run MiVOLO inference for a single person."""
    def preprocess(crop):
        if crop is None:
            return None
        inputs = processor(images=crop, return_tensors="pt")
        return inputs["pixel_values"].to(device)

    faces_input = preprocess(face_crop)
    body_input = preprocess(body_crop)

    if faces_input is None and body_input is None:
        return {"gender": None, "gender_confidence": None, "age": None}

    with torch.no_grad():
        output = mivolo_model(faces_input=faces_input, body_input=body_input)

    age = round(float(output.age_output[0].item()), 1)
    gender_idx = output.gender_class_idx[0].item()
    gender = config.gender_id2label[gender_idx]

    # Gender confidence from softmax logits if available
    gender_conf = None
    if hasattr(output, "gender_output"):
        probs = torch.softmax(output.gender_output[0], dim=-1)
        gender_conf = round(float(probs[gender_idx].item()), 4)

    return {"gender": gender, "gender_confidence": gender_conf, "age": age}


# ── Annotation ─────────────────────────────────────────────────────────────────

def annotate_image(image: Image.Image, results: list) -> Image.Image:
    """Draw bounding boxes and age/gender labels on image."""
    draw = ImageDraw.Draw(image)
    for person in results:
        label = f"{person.get('gender', '?')} {person.get('age', '?')}"
        box = person.get("face_box") or person.get("person_box")
        if box:
            draw.rectangle(box, outline="lime", width=3)
            draw.text((box[0], max(0, box[1] - 16)), label, fill="lime")
        if person.get("person_box") and person.get("face_box"):
            draw.rectangle(person["person_box"], outline="cyan", width=2)
    return image


# ── Main pipeline ──────────────────────────────────────────────────────────────

def process_image(image_path: str, output_path: str | None,
                  device: str, draw: bool,
                  detector, mivolo_model, processor, config) -> list:
    """Full pipeline for a single image."""
    image = Image.open(image_path).convert("RGB")

    # Step 1: Detect faces and persons
    faces, persons = detect(detector, image)
    print(f"  Detected: {len(faces)} face(s), {len(persons)} person(s)", file=sys.stderr)

    # Step 2: Match faces to persons
    pairs = match_faces_to_persons(faces, persons)

    # Step 3: For each pair, crop and run MiVOLO
    results = []
    for i, pair in enumerate(pairs):
        face_crop = crop_and_resize(image, pair["face_box"])
        body_crop = crop_and_resize(image, pair["person_box"])
        prediction = predict_age_gender(mivolo_model, processor, config, device, face_crop, body_crop)
        results.append({
            "person_id": i + 1,
            "gender": prediction["gender"],
            "gender_confidence": prediction["gender_confidence"],
            "age": prediction["age"],
            "face_box": pair["face_box"],
            "person_box": pair["person_box"],
        })

    # Step 4: Optionally save annotated image
    if draw and output_path:
        annotated = annotate_image(image.copy(), results)
        annotated.save(output_path)
        print(f"  Annotated image saved to: {output_path}", file=sys.stderr)

    return results


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MiVOLO — Age & Gender Detection Skill")
    parser.add_argument("--image", required=True,
                        help="Path to input image or directory of images")
    parser.add_argument("--output", default=None,
                        help="Path to save annotated output image (or directory if --image is a dir)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device: cuda or cpu (default: auto-detect)")
    parser.add_argument("--draw", action="store_true",
                        help="Save annotated image with bounding boxes and labels")
    args = parser.parse_args()

    # Load models once
    detector = load_detector(args.device)
    mivolo_model, processor, config = load_mivolo(args.device)

    image_path = Path(args.image)

    if image_path.is_dir():
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = sorted(f for f in image_path.iterdir() if f.suffix.lower() in extensions)
        if args.output:
            Path(args.output).mkdir(parents=True, exist_ok=True)
        all_results = {}
        for img in images:
            out = str(Path(args.output) / f"{img.stem}_annotated{img.suffix}") if args.output else None
            print(f"Processing {img.name}...", file=sys.stderr)
            results = process_image(str(img), out, args.device, args.draw,
                                    detector, mivolo_model, processor, config)
            all_results[img.name] = results
        print(json.dumps(all_results, ensure_ascii=False, indent=2))

    elif image_path.is_file():
        results = process_image(str(image_path), args.output, args.device, args.draw,
                                detector, mivolo_model, processor, config)
        print(json.dumps(results, ensure_ascii=False, indent=2))

    else:
        print(f"Error: '{image_path}' does not exist.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
