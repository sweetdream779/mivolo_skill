#!/usr/bin/env python3
"""
MiVOLO Skill — Age & Gender Detection

Models are loaded via the Transformers API (trust_remote_code=True).
Face-person matching, cropping and annotation are delegated to the mivolo package
(PersonAndFaceResult) to avoid duplicating logic.

Pipeline:
  1. Detect faces and persons via YOLO (iitolstykh/YOLO-Face-Person-Detector)
  2. Wrap results in PersonAndFaceResult (handles matching + cropping)
  3. Run MiVOLO v2 (iitolstykh/mivolo_v2) for age & gender prediction
  4. Optionally annotate with PersonAndFaceResult.plot()

Models:
  - Detector: https://huggingface.co/iitolstykh/YOLO-Face-Person-Detector
  - MiVOLO:   https://huggingface.co/iitolstykh/mivolo_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageClassification,
)

from mivolo.structures import PersonAndFaceResult

# ── Constants ──────────────────────────────────────────────────────────────────

DETECTOR_MODEL_ID = "iitolstykh/YOLO-Face-Person-Detector"
MIVOLO_MODEL_ID = "iitolstykh/mivolo_v2"
DETECTOR_CONF = 0.4
DETECTOR_IOU = 0.7


# ── Model loading ──────────────────────────────────────────────────────────────

def load_detector(device: str):
    """Load YOLOv8 face+person detector via Transformers API."""
    print(f"Loading detector {DETECTOR_MODEL_ID}...", file=sys.stderr)
    model = AutoModel.from_pretrained(
        DETECTOR_MODEL_ID,
        trust_remote_code=True,
    ).to(device)
    return model


def load_mivolo(device: str):
    """Load MiVOLO v2 model, processor and config via Transformers API."""
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

def detect(detector, image: np.ndarray) -> PersonAndFaceResult:
    """Run YOLO detector and wrap raw results in PersonAndFaceResult."""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    yolo_results = detector(pil_image, conf=DETECTOR_CONF, iou=DETECTOR_IOU)[0]
    return PersonAndFaceResult(yolo_results)


# ── MiVOLO inference ───────────────────────────────────────────────────────────

def predict_age_gender(
    mivolo_model,
    processor,
    config,
    device: str,
    detected: PersonAndFaceResult,
    image: np.ndarray,
):
    """Run MiVOLO on all detected face-person pairs via PersonAndFaceResult."""
    if detected.n_objects == 0:
        return

    detected.associate_faces_with_persons()
    crops = detected.collect_crops(image)

    (bodies_inds, bodies_crops), (faces_inds, faces_crops) = crops.get_faces_with_bodies(
        use_persons=True, use_faces=True,
    )

    for i in range(len(faces_inds)):
        face_crop = faces_crops[i]
        body_crop = bodies_crops[i]

        face_input = _preprocess_crop(processor, face_crop, device)
        body_input = _preprocess_crop(processor, body_crop, device)

        if face_input is None and body_input is None:
            continue

        with torch.no_grad():
            output = mivolo_model(faces_input=face_input, body_input=body_input)

        age = round(float(output.age_output[0].item()), 1)
        gender_idx = output.gender_class_idx[0].item()
        gender = config.gender_id2label[gender_idx]

        gender_score = None
        if output.gender_probs is not None:
            gender_score = round(float(output.gender_probs[0].max().item()), 4)

        face_ind = faces_inds[i]
        body_ind = bodies_inds[i]
        detected.set_age(face_ind, age)
        detected.set_age(body_ind, age)
        if gender_score is not None:
            detected.set_gender(face_ind, gender, gender_score)
            detected.set_gender(body_ind, gender, gender_score)


def _preprocess_crop(processor, crop: np.ndarray | None, device: str) -> torch.Tensor | None:
    """Preprocess a single crop (numpy BGR) for MiVOLO."""
    if crop is None:
        return None
    if crop.size == 0:
        return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    inputs = processor.preprocess([rgb])
    return inputs["pixel_values"].to(device)


# ── Results extraction ─────────────────────────────────────────────────────────

def extract_results(detected: PersonAndFaceResult) -> list[dict]:
    """Convert PersonAndFaceResult into a JSON-serializable list."""
    results = []
    person_id = 0
    boxes = detected.yolo_results.boxes
    names = detected.yolo_results.names

    for face_ind, person_ind in detected.face_to_person_map.items():
        person_id += 1
        entry = _make_entry(person_id, detected, boxes, names, face_ind, person_ind)
        results.append(entry)

    for person_ind in detected.unassigned_persons_inds:
        person_id += 1
        entry = _make_entry(person_id, detected, boxes, names, None, person_ind)
        results.append(entry)

    return results


def _make_entry(
    person_id: int,
    detected: PersonAndFaceResult,
    boxes,
    names,
    face_ind: int | None,
    person_ind: int | None,
) -> dict:
    """Build a single result dict from detection indices."""
    entry: dict = {"person_id": person_id}

    age = None
    gender = None
    gender_score = None
    for ind in (face_ind, person_ind):
        if ind is not None:
            if age is None and detected.ages[ind] is not None:
                age = detected.ages[ind]
            if gender is None and detected.genders[ind] is not None:
                gender = detected.genders[ind]
                gender_score = detected.gender_scores[ind]

    entry["gender"] = gender
    entry["gender_confidence"] = round(gender_score, 4) if gender_score is not None else None
    entry["age"] = age

    if face_ind is not None:
        bb = boxes[face_ind]
        entry["face_box"] = [round(v) for v in bb.xyxy[0].tolist()]
        entry["face_detection_conf"] = round(float(bb.conf), 4)
    else:
        entry["face_box"] = None
        entry["face_detection_conf"] = None

    if person_ind is not None:
        bb = boxes[person_ind]
        entry["person_box"] = [round(v) for v in bb.xyxy[0].tolist()]
        entry["person_detection_conf"] = round(float(bb.conf), 4)
    else:
        entry["person_box"] = None
        entry["person_detection_conf"] = None

    return entry


def log_results(results: list[dict]):
    """Print per-person summary to stderr."""
    for r in results:
        parts = [f"  Person {r['person_id']}: {r['gender']}, age {r['age']}"]
        if r["gender_confidence"] is not None:
            parts.append(f"gender_conf={r['gender_confidence']}")
        if r["face_box"] is not None:
            parts.append(f"face={r['face_box']} (conf={r['face_detection_conf']})")
        if r["person_box"] is not None:
            parts.append(f"body={r['person_box']} (conf={r['person_detection_conf']})")
        print(" | ".join(parts), file=sys.stderr)


# ── Image loading ──────────────────────────────────────────────────────────────

def load_image(source: str) -> np.ndarray:
    """Load image from a local path or URL, return as BGR numpy array."""
    if source.startswith("http://") or source.startswith("https://"):
        print(f"  Downloading image from {source} ...", file=sys.stderr)
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MiVOLO/1.0)"}
        response = requests.get(source, timeout=15, headers=headers)
        response.raise_for_status()
        pil_img = Image.open(BytesIO(response.content)).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return cv2.imread(source)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def process_image(
    image_path: str,
    output_path: str | None,
    device: str,
    draw: bool,
    detector,
    mivolo_model,
    processor,
    config,
) -> list:
    """Full pipeline for a single image (local path or URL)."""
    image = load_image(image_path)

    detected = detect(detector, image)
    print(
        f"  Detected: {detected.n_faces} face(s), {detected.n_persons} person(s)",
        file=sys.stderr,
    )

    predict_age_gender(mivolo_model, processor, config, device, detected, image)

    results = extract_results(detected)
    log_results(results)

    if draw and output_path:
        annotated = detected.plot(conf=True, ages=True, genders=True, gender_probs=True)
        cv2.imwrite(output_path, annotated)
        print(f"  Annotated image saved to: {output_path}", file=sys.stderr)

    return results


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MiVOLO — Age & Gender Detection Skill")
    parser.add_argument(
        "--image", required=True,
        help="Path to input image, directory of images, or http(s):// URL",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save annotated output image (or directory if --image is a dir)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cuda or cpu (default: auto-detect)",
    )
    parser.add_argument(
        "--draw", action="store_true",
        help="Save annotated image with bounding boxes and labels",
    )
    args = parser.parse_args()

    detector = load_detector(args.device)
    mivolo_model, processor, config = load_mivolo(args.device)

    if args.image.startswith("http://") or args.image.startswith("https://"):
        print(f"Processing URL: {args.image}", file=sys.stderr)
        results = process_image(
            args.image, args.output, args.device, args.draw,
            detector, mivolo_model, processor, config,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

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
            results = process_image(
                str(img), out, args.device, args.draw,
                detector, mivolo_model, processor, config,
            )
            all_results[img.name] = results
        print(json.dumps(all_results, ensure_ascii=False, indent=2))

    elif image_path.is_file():
        results = process_image(
            str(image_path), args.output, args.device, args.draw,
            detector, mivolo_model, processor, config,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))

    else:
        print(f"Error: '{image_path}' does not exist.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
