#!/usr/bin/env python3
"""
MiVOLO Skill — Age & Gender Detection
Uses MiVOLO v2 from https://huggingface.co/iitolstykh/mivolo_v2
Library: https://github.com/WildChlamydia/MiVOLO
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image


def load_model(device: str):
    """Load MiVOLO v2 model and processor from HuggingFace."""
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

    model_id = "iitolstykh/mivolo_v2"
    print(f"Loading model {model_id}...", file=sys.stderr)

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageClassification.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()

    return model, processor, config


def detect_persons(image_path: str):
    """
    Detect person bounding boxes (faces + bodies) using MiVOLO's built-in detector.
    Returns list of (face_crop, body_crop, bbox) tuples.
    """
    from mivolo.predictor import Predictor
    from mivolo.structures import PersonAndFaceResult

    # MiVOLO uses YOLO-based detector internally
    # The Predictor class handles detection + crops automatically
    return None  # handled via Predictor below


def run_inference_mivolo_library(image_path: str, output_path: str | None, device: str, draw: bool):
    """
    Run inference using the MiVOLO library (Predictor class).
    This is the recommended approach as it handles detection automatically.
    """
    from mivolo.predictor import Predictor

    # Predictor uses the HuggingFace model under the hood
    predictor = Predictor(
        detector_weights=None,  # auto-download or use default
        checkpoint="iitolstykh/mivolo_v2",
        device=device,
        with_persons=True,
        disable_faces=False,
    )

    image = Image.open(image_path).convert("RGB")
    detected_objects, output_image = predictor.recognize(image)

    results = []
    for i, person in enumerate(detected_objects.get_results_for_image(0)):
        results.append({
            "person_id": i + 1,
            "gender": person.gender,
            "gender_confidence": round(float(person.gender_score), 4) if person.gender_score else None,
            "age": round(float(person.age), 1) if person.age else None,
            "bbox": [round(x) for x in person.bbox] if person.bbox else None,
        })

    if draw and output_path and output_image:
        output_image.save(output_path)
        print(f"Annotated image saved to: {output_path}", file=sys.stderr)

    return results


def run_inference_transformers(image_path: str, device: str):
    """
    Run inference using the Transformers API directly (without MiVOLO library).
    NOTE: Requires manual face/body crops — use run_inference_mivolo_library for full pipeline.
    """
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

    model, processor, config = load_model(device)

    image = Image.open(image_path).convert("RGB")

    # Process the full image as a single crop (simplified — no detection step)
    inputs = processor(images=image, return_tensors="pt")
    faces_input = inputs["pixel_values"].to(device)

    with torch.no_grad():
        output = model(faces_input=faces_input, body_input=None)

    age = round(output.age_output[0].item(), 1)
    gender = config.gender_id2label[output.gender_class_idx[0].item()]

    return [{"person_id": 1, "gender": gender, "age": age, "bbox": None}]


def process_image(image_path: str, output_path: str | None, device: str, draw: bool) -> list:
    """Process a single image and return results."""
    try:
        results = run_inference_mivolo_library(image_path, output_path, device, draw)
    except ImportError:
        print("MiVOLO library not found, falling back to Transformers API (no detection).", file=sys.stderr)
        results = run_inference_transformers(image_path, device)
    return results


def main():
    parser = argparse.ArgumentParser(description="MiVOLO — Age & Gender Detection Skill")
    parser.add_argument("--image", required=True, help="Path to input image or directory of images")
    parser.add_argument("--output", default=None, help="Path to save annotated output image")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device: cuda or cpu (default: auto-detect)")
    parser.add_argument("--draw", action="store_true", help="Save annotated image with bounding boxes")
    args = parser.parse_args()

    image_path = Path(args.image)

    if image_path.is_dir():
        # Process all images in directory
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = [f for f in image_path.iterdir() if f.suffix.lower() in extensions]
        all_results = {}
        for img in sorted(images):
            out = Path(args.output) / f"{img.stem}_annotated{img.suffix}" if args.output else None
            results = process_image(str(img), str(out) if out else None, args.device, args.draw)
            all_results[img.name] = results
            print(f"{img.name}: {results}", file=sys.stderr)
        print(json.dumps(all_results, ensure_ascii=False, indent=2))

    elif image_path.is_file():
        results = process_image(str(image_path), args.output, args.device, args.draw)
        print(json.dumps(results, ensure_ascii=False, indent=2))

    else:
        print(f"Error: {image_path} does not exist.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
