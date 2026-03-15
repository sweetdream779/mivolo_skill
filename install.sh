#!/bin/bash
set -e

echo "Installing MiVOLO Skill dependencies..."

# Check Python version
python3 -c "import sys; assert sys.version_info >= (3, 8), 'Python 3.8+ required'" || {
    echo "Error: Python 3.8+ is required"
    exit 1
}

# Install PyTorch (with CUDA if available, fallback to CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || \
pip install torch torchvision

# Install remaining dependencies
pip install \
    "transformers==4.51.0" \
    "accelerate==1.8.1" \
    "ultralytics==8.1.0" \
    "huggingface_hub" \
    "Pillow" \
    "numpy"

echo ""
echo "✅ MiVOLO Skill installed successfully!"
echo ""
echo "Models will be downloaded automatically on first run from HuggingFace:"
echo "  - iitolstykh/YOLO-Face-Person-Detector"
echo "  - iitolstykh/mivolo_v2"
echo ""
echo "Usage:"
echo "  python mivolo_inference.py --image photo.jpg"
echo "  python mivolo_inference.py --image photo.jpg --draw --output result.jpg"
echo "  python mivolo_inference.py --image ./photos/ --draw --output ./results/"
